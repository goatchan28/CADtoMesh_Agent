"""
Simulation-readiness criteria: the VALIDITY GATE and the OBJECTIVE SCORE.

The distinction this module exists to enforce
---------------------------------------------
There are two fundamentally different kinds of mesh problem, and an adaptive loop
that conflates them cannot work.

HARD VALIDITY FAILURES make the mesh unusable. An inverted element gives a negative
Jacobian, so the element stiffness matrix is garbage and the solve either diverges
or returns confident nonsense. A non-manifold edge means the topology is not a
surface. These are not "low quality" -- there is no score that redeems them, and no
amount of further refinement fixes them. The correct response is to REJECT the mesh
and change strategy: different algorithm, different tolerance, repaired geometry.

OPTIMIZATION OBJECTIVES are continuous. A minimum angle of 22 degrees is worse than
28 and better than 15; a chordal error of 0.03h may be fine for a stiffness study
and unacceptable for a contact analysis. These are what an iterative loop should
push on, by adjusting sizing and remeshing.

Collapsing the two produces both failure modes. Treat everything as a soft score
and the loop happily converges on a mesh with three inverted elements because the
average looks good. Treat everything as a hard gate and the loop never terminates,
because "minimum angle above 30 degrees everywhere" is not achievable on real CAD.

So: `Verdict.is_valid` is a boolean gate that ignores scores entirely, and
`Verdict.score` is only meaningful once the gate passes. The loop's stopping rule
is "valid AND every objective satisfied", with the objectives carrying explicit
targets so "satisfied" is defined rather than judged.

Everything here is backend-independent. The same Criteria evaluate a gmsh mesh and a
research mesh, which is the point -- otherwise the comparison that motivated the
pivot to gmsh could not be repeated as gmsh is tuned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import numpy as np

from . import core, normals, size


class Kind(str, Enum):
    VALIDITY = "validity"      # hard gate; failure means reject
    OBJECTIVE = "objective"    # continuous; drives the loop


@dataclass
class Failure:
    """A hard validity failure. Presence of any means the mesh is unusable."""
    code: str
    message: str
    count: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Objective:
    """A continuous objective with an explicit target."""
    name: str
    value: float
    target: float
    satisfied: bool
    direction: str            # "max" -> higher is better; "min" -> lower is better
    weight: float = 1.0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        """0..1 attainment against the target. 1.0 means the target is met."""
        if self.direction == "max":
            if self.target <= 0:
                return 1.0
            return float(min(1.0, max(0.0, self.value / self.target)))
        if self.value <= 0:
            return 1.0
        if self.target <= 0:
            return 0.0
        return float(min(1.0, max(0.0, self.target / self.value)))


@dataclass
class Criteria:
    """Thresholds defining simulation-readiness. Tune per analysis type.

    Defaults target linear-triangle FEM on chunky solid geometry. A contact or
    thin-shell analysis wants tighter fidelity; a coarse stiffness screen can relax
    it. The point of having them in one dataclass is that the adaptive loop can be
    handed a Criteria and needs no other definition of "done".
    """

    # ---- hard validity ----
    allow_inverted: bool = False
    allow_nonmanifold: bool = False
    allow_inconsistent_normals: bool = False
    require_watertight: bool = False
    """Only meaningful for a CLOSED surface. A single trimmed face is legitimately
    open, so this defaults off and the caller turns it on when meshing a full solid
    boundary."""
    min_shape_floor: float = 0.05
    """Absolute shape floor. Below this an element is numerically degenerate rather
    than merely poor, and produces a near-singular element matrix regardless of how
    good the rest of the mesh is. Distinct from the min_shape OBJECTIVE below,
    which is the quality one actually wants.

    Raised from 0.01 after sliver_block.step passed the gate at shape 0.018.
    Measured correspondence for a thin isoceles triangle (apex angle):

        shape 0.003 -> 0.3 deg      shape 0.017 -> 1.7 deg
        shape 0.006 -> 0.6 deg      shape 0.033 -> 3.3 deg

    so shape is roughly (apex degrees)/100 in this regime, and the old 0.01 floor
    admitted anything above about 1 degree. Note the mesh's worst-SHAPE element and
    its worst-ANGLE element need not be the same triangle, which is why the table
    showed shp_min 0.018 alongside a 0.6 degree minimum angle.

    0.05 is a calibration choice, not a law. It is deliberately far below the
    min_shape objective (0.30): the floor answers "will the solve fail", the
    objective answers "is this a good mesh"."""
    max_duplicate_node_tolerance: float = 0.0
    """If > 0, nodes closer than this are treated as an unmerged-duplicate failure.
    The silent-crack failure mode: coincident but distinct nodes look perfect and
    transmit no load."""

    # ---- objectives ----
    min_shape: float = 0.30
    min_angle_deg: float = 20.0
    max_angle_deg: float = 130.0
    size_tolerance: float = 1.5
    min_size_in_band: float = 0.90
    max_gradation: float = 1.5
    max_chordal_relative: float = 0.05
    max_normal_deg: float = 15.0

    weights: dict = field(default_factory=lambda: {
        "min_shape": 2.0, "min_angle": 2.0, "max_angle": 1.0,
        "size_in_band": 1.5, "gradation": 1.0,
        "chordal": 1.5, "normal_deviation": 1.5,
    })

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    """The evaluation result. Read `is_valid` first; `score` only after it passes."""
    is_valid: bool
    failures: list[Failure] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    n_triangles: int = 0
    n_vertices: int = 0
    method: str = ""
    criteria: dict = field(default_factory=dict)

    @property
    def all_objectives_satisfied(self) -> bool:
        return all(o.satisfied for o in self.objectives)

    @property
    def ready(self) -> bool:
        """The adaptive loop's stopping condition."""
        return self.is_valid and self.all_objectives_satisfied

    @property
    def score(self) -> float:
        """Weighted attainment in 0..1. Zero if invalid -- deliberately.

        An invalid mesh must not be comparable to a valid one on a continuous scale,
        or a search will trade validity for a better average.
        """
        if not self.is_valid or not self.objectives:
            return 0.0
        tw = sum(o.weight for o in self.objectives)
        if tw <= 0:
            return 0.0
        return float(sum(o.normalized * o.weight for o in self.objectives) / tw)

    def unsatisfied(self) -> list[Objective]:
        """Objectives to act on, worst attainment first -- the loop's work list."""
        return sorted((o for o in self.objectives if not o.satisfied),
                      key=lambda o: o.normalized)

    def summary(self) -> str:
        lines = [f"{self.method or 'mesh'}: {self.n_triangles} tris, "
                 f"{self.n_vertices} verts"]
        if not self.is_valid:
            lines.append(f"INVALID -- {len(self.failures)} hard failure(s), "
                         "no score (reject and change strategy)")
            for f in self.failures:
                lines.append(f"  [{f.code}] {f.message}")
            return "\n".join(lines)
        lines.append(f"VALID | score {self.score:.3f} | "
                     f"{'READY' if self.ready else 'not ready'}")
        for o in self.objectives:
            mark = "ok  " if o.satisfied else "MISS"
            arrow = ">=" if o.direction == "max" else "<="
            lines.append(f"  {mark} {o.name:<18}{o.value:10.4g} {arrow} "
                         f"{o.target:<10.4g} ({o.normalized:.2f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------

def _duplicate_node_count(mesh: core.SurfaceMesh, tol: float) -> int:
    """Coincident-but-distinct node pairs, via a grid hash (never O(n^2))."""
    if tol <= 0 or mesh.n_vertices == 0:
        return 0
    v = mesh.vertices
    cell = max(tol, 1e-12)
    grid: dict[tuple, list[int]] = {}
    for i, p in enumerate(v):
        grid.setdefault(tuple((p / cell).astype(np.int64)), []).append(i)
    dup = 0
    for key, bucket in grid.items():
        neigh: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neigh.extend(grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
        for i in bucket:
            for j in neigh:
                if j > i and float(np.linalg.norm(v[i] - v[j])) <= tol:
                    dup += 1
    return dup


def evaluate(mesh: core.SurfaceMesh, *, criteria: Criteria | None = None,
             target_size=None, surface=None,
             interface_triangles=None,
             max_fidelity_samples: int | None = 1500) -> Verdict:
    """Full simulation-readiness evaluation of any surface mesh, any backend.

    Args:
        target_size: scalar or callable size field. Omit to skip size adequation
            (mesh.meta["target_size"] is used when present).
        interface_triangles: indices of triangles on internal faces of an
            imprinted assembly. Topology is then checked on the EXTERIOR SKIN. An
            imprinted two-body assembly otherwise reports ~30 non-manifold edges
            and fails the validity gate on a mesh that is exactly correct.
        surface: needed for the fidelity objectives (chordal, normal deviation).
            Omit them and those objectives are simply not reported -- a topology and
            shape verdict is still returned, which is what you get when meshing
            something with no analytic surface to compare against.

    Hard failures are collected first and independently of every score.
    """
    cr = criteria or Criteria()
    v = Verdict(is_valid=True, n_triangles=mesh.n_triangles,
                n_vertices=mesh.n_vertices, method=mesh.method,
                criteria=cr.to_dict())

    if mesh.n_triangles == 0:
        v.is_valid = False
        v.failures.append(Failure("mesh.empty", "no triangles"))
        return v

    rep = core.evaluate(mesh)
    topo = core.topology_checks(mesh, interface_triangles)
    a, b, c = mesh.corners()

    # ---------------- hard validity gate ----------------
    if not cr.allow_inverted and rep.n_inverted > 0:
        v.failures.append(Failure(
            "validity.inverted", f"{rep.n_inverted} inverted/zero-area element(s); "
            "negative Jacobian makes the element matrix meaningless",
            count=rep.n_inverted))

    if not cr.allow_nonmanifold and topo["n_nonmanifold_edges"] > 0:
        v.failures.append(Failure(
            "validity.nonmanifold",
            f"{topo['n_nonmanifold_edges']} edge(s) shared by 3+ triangles; "
            "the mesh is not a surface", count=topo["n_nonmanifold_edges"]))

    if not cr.allow_inconsistent_normals and topo["n_inconsistent_normals"] > 0:
        v.failures.append(Failure(
            "validity.orientation",
            f"{topo['n_inconsistent_normals']} adjacent pair(s) with opposing "
            "winding; surface integrals and any outward-normal convention break",
            count=topo["n_inconsistent_normals"]))

    if cr.require_watertight and not topo["is_watertight"]:
        v.failures.append(Failure(
            "validity.not_watertight",
            f"{topo['n_boundary_edges']} boundary edge(s) on a surface required to "
            "be closed", count=topo["n_boundary_edges"]))

    shape = core.shape_quality(a, b, c)
    n_degenerate = int(np.sum(shape < cr.min_shape_floor))
    if n_degenerate > 0:
        v.failures.append(Failure(
            "validity.degenerate_shape",
            f"{n_degenerate} element(s) below the absolute shape floor "
            f"{cr.min_shape_floor}; numerically degenerate, not merely poor",
            count=n_degenerate,
            detail={"worst": float(shape.min())}))

    if cr.max_duplicate_node_tolerance > 0:
        dup = _duplicate_node_count(mesh, cr.max_duplicate_node_tolerance)
        if dup > 0:
            v.failures.append(Failure(
                "validity.duplicate_nodes",
                f"{dup} coincident-but-distinct node pair(s); the mesh has a crack "
                "that looks perfect and transmits no load", count=dup))

    v.is_valid = not v.failures

    # ---------------- objectives ----------------
    w = cr.weights

    def add(name, value, target, direction):
        ok = value >= target if direction == "max" else value <= target
        v.objectives.append(Objective(name=name, value=float(value),
                                      target=float(target), satisfied=bool(ok),
                                      direction=direction,
                                      weight=float(w.get(name, 1.0))))

    add("min_shape", rep.shape_min, cr.min_shape, "max")
    add("min_angle", rep.min_angle, cr.min_angle_deg, "max")
    add("max_angle", rep.max_angle, cr.max_angle_deg, "min")

    ts = target_size if target_size is not None else mesh.meta.get("target_size")
    if ts is not None:
        sr = size.size_adequation(mesh, ts, tolerance=cr.size_tolerance)
        add("size_in_band", sr.frac_in_band, cr.min_size_in_band, "max")
        v.objectives[-1].detail = {"ratio_min": sr.ratio_min,
                                   "ratio_max": sr.ratio_max,
                                   "frac_oversized": sr.frac_oversized,
                                   "frac_undersized": sr.frac_undersized}

    gr = size.gradation(mesh, limit=cr.max_gradation)
    add("gradation", gr.p99_ratio, cr.max_gradation, "min")
    v.objectives[-1].detail = {"max_ratio": gr.max_ratio, "n_pairs": gr.n_pairs}

    if surface is not None:
        ch = core.chordal_deviation(mesh, surface, target_size=ts,
                                    max_triangles=max_fidelity_samples)
        if ts is not None and math.isfinite(ch.max_relative):
            add("chordal", ch.max_relative, cr.max_chordal_relative, "min")
            v.objectives[-1].detail = {"max_deviation": ch.max_deviation}
        nd = normals.normal_deviation(mesh, surface, limit_deg=cr.max_normal_deg,
                                      max_triangles=max_fidelity_samples)
        add("normal_deviation", nd.p99_deg, cr.max_normal_deg, "min")
        v.objectives[-1].detail = {"max_deg": nd.max_deg,
                                   "area_weighted_mean_deg":
                                       nd.area_weighted_mean_deg}
    return v
