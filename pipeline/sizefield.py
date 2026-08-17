"""
Declarative size field. PURE -- no gmsh.

A SizeFieldSpec is DATA, not code. That is the central decision in this module and
it buys three things:

  * a future agent edits a typed, validated structure instead of emitting mesher
    code -- the same reasoning that made Stage 1's tools typed rather than
    code-generating
  * successive specs can be DIFFED, so a run record shows exactly what changed
    between iterations rather than "we remeshed and it got better"
  * a spec HASHES, which is how the loop detects that it is oscillating between
    two parameter sets instead of converging

Rules are scoped by persistent entity id (PID), never by backend tag, so a rule
written against evidence from iteration 3 still refers to the same face in
iteration 7 even after a re-import renumbered everything.

Curvature sizing
----------------
Two independent bounds, and the spec takes the MINIMUM:

    chordal:  delta = h^2 / (8R)   ->   h <= sqrt(8 R delta)
    normal:   theta = asin(h / 2R) ->   h <= 2 R sin(theta)

Both matter because they scale differently. Chordal error is O(h^2/R) and normal
error is O(h/R), so halving element size quarters the first but only halves the
second. Normal deviation is therefore the binding constraint at coarse sizes, and a
size field driven by sagitta alone systematically under-refines curved faces. This
was measured during the research phase, not assumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum


class RuleKind(str, Enum):
    BASE = "base"                # global default size
    CURVATURE = "curvature"      # chordal + normal driven
    THICKNESS = "thickness"      # N elements through a thin wall
    FEATURE = "feature"          # explicit local size on named entities
    CLAMP = "clamp"              # absolute floor/ceiling
    GRADATION = "gradation"      # max neighbour size ratio


@dataclass(frozen=True)
class Rule:
    """One sizing rule. `scope` is a tuple of PIDs, or empty meaning global."""
    kind: RuleKind
    params: dict
    scope: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "params": dict(self.params),
                "scope": list(self.scope), "note": self.note}


def base(h: float, note: str = "") -> Rule:
    return Rule(RuleKind.BASE, {"h": float(h)}, (), note)


def curvature(chordal_tol: float, normal_tol_deg: float,
              min_per_2pi: float = 12.0, scope=(), note: str = "") -> Rule:
    """Chordal tolerance is ABSOLUTE (model units); normal tolerance is degrees."""
    return Rule(RuleKind.CURVATURE,
                {"chordal_tol": float(chordal_tol),
                 "normal_tol_deg": float(normal_tol_deg),
                 "min_per_2pi": float(min_per_2pi)}, tuple(scope), note)


def thickness(min_elements_through: int = 3, scope=(), note: str = "") -> Rule:
    return Rule(RuleKind.THICKNESS,
                {"min_elements_through": int(min_elements_through)},
                tuple(scope), note)


def feature(h: float, scope=(), note: str = "") -> Rule:
    return Rule(RuleKind.FEATURE, {"h": float(h)}, tuple(scope), note)


def clamp(h_min: float, h_max: float, note: str = "") -> Rule:
    return Rule(RuleKind.CLAMP, {"h_min": float(h_min), "h_max": float(h_max)},
                (), note)


def gradation(max_growth: float = 1.4, note: str = "") -> Rule:
    return Rule(RuleKind.GRADATION, {"max_growth": float(max_growth)}, (), note)


# ---------------------------------------------------------------------------

def size_from_curvature(radius: float, chordal_tol: float, normal_tol_deg: float,
                        min_per_2pi: float = 12.0) -> float:
    """Largest element size satisfying BOTH the chordal and normal bounds.

    radius is the local radius of curvature (1/max|kappa|). Returns inf for a
    planar region, which the caller clamps against the base size.
    """
    if not math.isfinite(radius) or radius <= 0:
        return math.inf

    h_chord = (math.sqrt(8.0 * radius * chordal_tol)
               if 0 < chordal_tol < math.inf else math.inf)

    # Clamp the half-angle at 90 degrees. sin() is only monotone on [0, pi/2]; past
    # that the "bound" turns around and eventually goes NEGATIVE, so a large value
    # meant as "disable this term" silently becomes the tightest constraint of all
    # and drives the requested size negative. 2R*sin(90) = the diameter, which is
    # the correct meaning of an unconstrained normal tolerance.
    theta = min(math.radians(max(normal_tol_deg, 0.0)), 0.5 * math.pi)
    h_normal = 2.0 * radius * math.sin(theta) if theta > 0 else math.inf

    h_count = (2.0 * math.pi * radius / min_per_2pi
               if 0 < min_per_2pi < math.inf else math.inf)
    return min(h_chord, h_normal, h_count)


def smooth_sizes(sizes: dict[str, float], adjacency: dict[str, set[str]],
                 max_growth: float, extents: dict[str, float] | None = None,
                 max_passes: int = 50) -> dict[str, float]:
    """Enforce a maximum size growth rate between neighbouring faces.

    Requested sizes are computed per face independently, so a small filleted face
    beside a large planar one can ask for a 20x jump. Meshers smooth this
    internally by unspecified amounts, which makes the `gradation` objective
    unattributable -- you cannot tell a mesher that ignored the field from a field
    that was never gradable. Smoothing the REQUEST first makes it measurable.

    GRADATION IS A SPATIAL LIMIT, NOT A TOPOLOGICAL ONE.

    `extents` maps face PID -> characteristic length (sqrt of area). Without it,
    a neighbour is capped at h * max_growth: one hop, one factor. But a single hop
    can cross a 50 mm face, so a tiny local size propagates across the whole model
    at full strength. Measured on sliver_block.step: a 0.005 mm rail forced a
    0.077 mm size that walked outward over the face graph and produced 1,668,436
    triangles in 16 s, against ~1900 before.

    With extents, a face of length L carrying size h can grow internally by
    max_growth ** (L / h) before reaching its far edge, so that is the ceiling it
    imposes on its neighbours. A large face therefore keeps its own coarse size
    next to a small one, and the mesher grades across it -- which is what actually
    happens physically.
    """
    if max_growth <= 1.0:
        return dict(sizes)
    out = dict(sizes)
    for _ in range(max_passes):
        changed = False
        for pid, h in list(out.items()):
            if h <= 0:
                continue
            if extents:
                L = float(extents.get(pid, 0.0))
                # Cap the exponent: a wide face imposes no practical limit, and
                # 1.4**1e6 overflows.
                steps = min(L / h, 40.0) if h > 0 else 0.0
                limit = h * (max_growth ** max(steps, 1.0))
            else:
                limit = h * max_growth
            for nb in adjacency.get(pid, ()):
                hn = out.get(nb)
                if hn is not None and hn > limit:
                    out[nb] = limit
                    changed = True
        if not changed:
            break
    return out


def ramp_distance(h_near: float, h_far: float, max_growth: float) -> float:
    """Physical distance needed to grow from h_near to h_far at max_growth per element.

    Elements growing geometrically from h_near by ratio g cover, after n elements,
    a distance h_near*(g^n - 1)/(g - 1). Setting g^n = h_far/h_near gives

        distance = (h_far - h_near) / (g - 1)

    This is the number a Threshold field needs for DistMax. It is what makes the
    gradation objective ACHIEVABLE rather than merely measured: a constant per-face
    size has no ramp inside a face, so elements of both sizes meet at the shared
    edge and the observed gradation is the raw face-size ratio. On block_hole that
    ratio is 2.44 against a 1.5 limit, and no per-face smoothing can fix it without
    coarsening the fillets or refining the whole part.
    """
    if h_near <= 0 or h_far <= h_near or max_growth <= 1.0:
        return 0.0
    return (h_far - h_near) / (max_growth - 1.0)


def estimate_element_count(sizes: dict[str, float],
                           areas: dict[str, float]) -> int:
    """Triangles implied by a size field: sum(area / (sqrt(3)/4 * h^2)).

    Cheap, and it is the guard the adaptive loop needs BEFORE meshing. A refinement
    action that would produce two million elements is not an improvement to be
    discovered after 16 seconds of meshing; it is a decision that has to be taken
    deliberately.

    LOWER BOUND when the backend compiles spatial ramps. This counts each face at
    its own requested size; a Threshold ramp also refines a band of the NEIGHBOURING
    coarse faces, which is not counted here. Against constant per-face fields the
    estimate ran 1.01-1.27x low; with ramps expect a larger factor, and calibrate it
    from tools/verify_backend.py's est-vs-actual column before tightening a budget.
    """
    total = 0.0
    for pid, h in sizes.items():
        a = float(areas.get(pid, 0.0))
        if h > 0 and a > 0 and math.isfinite(h):
            total += a / (math.sqrt(3.0) / 4.0 * h * h)
    return int(round(total))


@dataclass
class SizeFieldSpec:
    """An ordered list of rules plus the resolved per-entity sizes they imply."""
    rules: list[Rule] = field(default_factory=list)
    model_scale: float = 1.0

    # -- construction --------------------------------------------------------
    def with_rule(self, rule: Rule) -> "SizeFieldSpec":
        return SizeFieldSpec(rules=list(self.rules) + [rule],
                             model_scale=self.model_scale)

    def replace_kind(self, kind: RuleKind, rule: Rule) -> "SizeFieldSpec":
        """Swap the first rule of a kind, or append. The loop's main edit."""
        rules = list(self.rules)
        for i, r in enumerate(rules):
            if r.kind is kind and r.scope == rule.scope:
                rules[i] = rule
                return SizeFieldSpec(rules, self.model_scale)
        rules.append(rule)
        return SizeFieldSpec(rules, self.model_scale)

    def get(self, kind: RuleKind, scope: tuple = ()) -> Rule | None:
        for r in self.rules:
            if r.kind is kind and r.scope == scope:
                return r
        return None

    @staticmethod
    def default(model_scale: float, base_frac: float = 0.05) -> "SizeFieldSpec":
        """A defensible starting point, expressed as fractions of model scale.

        Absolute defaults would be wrong for every part but one; fractions of the
        bounding-box diagonal transfer across a 5 mm bracket and a 5 m weldment,
        the same convention Stage 0 uses for its thresholds.
        """
        h0 = base_frac * model_scale
        return SizeFieldSpec(rules=[
            base(h0, "default: 5% of model scale"),
            curvature(chordal_tol=0.02 * h0, normal_tol_deg=12.0,
                      note="chordal 2% of base size; normal bound usually binds"),
            thickness(3, note="3 elements through the thinnest wall"),
            clamp(h_min=0.02 * h0, h_max=4.0 * h0,
                  note="floor stops curvature from exploding element count"),
            gradation(1.4),
        ], model_scale=model_scale)

    # -- resolution ----------------------------------------------------------
    def resolve(self, faces: dict[str, dict],
                adjacency: dict[str, set[str]] | None = None,
                smooth: bool = True) -> dict[str, float]:
        """Compute a target size per face PID.

        `faces` maps PID -> {"curvature_max": float, "thickness": float|None}.
        curvature_max is max |principal curvature| on that face (0 for planar).

        smooth=False skips per-face gradation smoothing. A backend that compiles
        SPATIAL ramps (Distance + Threshold) should pass False: the ramp enforces
        gradation properly, and pre-smoothing on top of it would coarsen the fine
        faces for no benefit. Per-face smoothing is only a stand-in for backends
        that can express nothing but a constant size per face.

        Order is deliberate: base, then each restriction takes the MINIMUM, then
        clamp, then gradation smoothing. Every rule can only shrink the size, so
        resolution is order-independent among the restricting rules and therefore
        reproducible.
        """
        b = self.get(RuleKind.BASE)
        h0 = float(b.params["h"]) if b else 0.05 * self.model_scale
        sizes = {pid: h0 for pid in faces}

        for rule in self.rules:
            scope = set(rule.scope) if rule.scope else set(faces)

            if rule.kind is RuleKind.CURVATURE:
                for pid in scope & set(faces):
                    k = float(faces[pid].get("curvature_max") or 0.0)
                    if k <= 0:
                        continue
                    h = size_from_curvature(1.0 / k, rule.params["chordal_tol"],
                                            rule.params["normal_tol_deg"],
                                            rule.params["min_per_2pi"])
                    sizes[pid] = min(sizes[pid], h)

            elif rule.kind is RuleKind.THICKNESS:
                n = max(1, int(rule.params["min_elements_through"]))
                for pid in scope & set(faces):
                    t = faces[pid].get("thickness")
                    if t:
                        sizes[pid] = min(sizes[pid], float(t) / n)

            elif rule.kind is RuleKind.FEATURE:
                for pid in scope & set(faces):
                    sizes[pid] = min(sizes[pid], float(rule.params["h"]))

        c = self.get(RuleKind.CLAMP)
        if c:
            lo, hi = float(c.params["h_min"]), float(c.params["h_max"])
            sizes = {p: min(max(h, lo), hi) for p, h in sizes.items()}

        g = self.get(RuleKind.GRADATION)
        if g and adjacency and smooth:
            extents = {pid: math.sqrt(max(float(f.get("area") or 0.0), 0.0))
                       for pid, f in faces.items()}
            sizes = smooth_sizes(sizes, adjacency, float(g.params["max_growth"]),
                                 extents=extents)
        return sizes

    # -- identity and diffing ------------------------------------------------
    def to_dict(self) -> dict:
        return {"model_scale": self.model_scale,
                "rules": [r.to_dict() for r in self.rules]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def fingerprint(self) -> str:
        """Stable hash. The loop's oscillation guard compares these."""
        payload = json.dumps(self.to_dict(), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def diff(self, other: "SizeFieldSpec") -> list[str]:
        """Human- and agent-readable description of what changed."""
        out: list[str] = []
        mine = {(r.kind.value, r.scope): r for r in self.rules}
        theirs = {(r.kind.value, r.scope): r for r in other.rules}
        for key in sorted(set(mine) | set(theirs), key=lambda k: (k[0], k[1])):
            a, b = mine.get(key), theirs.get(key)
            scope = f" on {len(key[1])} entities" if key[1] else ""
            if a is None:
                out.append(f"+ {key[0]}{scope}: {b.params}")
            elif b is None:
                out.append(f"- {key[0]}{scope}")
            elif a.params != b.params:
                changed = {k: (a.params.get(k), b.params.get(k))
                           for k in set(a.params) | set(b.params)
                           if a.params.get(k) != b.params.get(k)}
                out.append(f"~ {key[0]}{scope}: " + ", ".join(
                    f"{k} {v[0]!r} -> {v[1]!r}" for k, v in sorted(changed.items())))
        return out


def spec_from_dict(d: dict) -> SizeFieldSpec:
    return SizeFieldSpec(
        rules=[Rule(RuleKind(r["kind"]), dict(r["params"]),
                    tuple(r.get("scope", ())), r.get("note", ""))
               for r in d.get("rules", [])],
        model_scale=float(d.get("model_scale", 1.0)))
