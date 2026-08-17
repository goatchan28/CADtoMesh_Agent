"""
Adaptation policy. PURE -- no gmsh, no backend, no LLM.

Maps evidence (a Verdict plus per-face scores) onto a CLOSED SET of actions. Closed
is the point: a future agent chooses among candidates that this module generated and
validated, it does not invent parameters. Every action is fully serializable and
carries the evidence that produced it, so a run record explains itself.

The central rule, from quality/criteria.py
------------------------------------------
HARD FAILURES AND UNSATISFIED OBJECTIVES ARE REACHED BY DIFFERENT CODE PATHS.

A hard failure means the mesh is unusable -- inverted elements, non-manifold
topology, degenerate shape. No amount of refinement fixes it, so the response is a
STRATEGY change: heal, re-imprint, change algorithm, or escalate. Proposing "refine
those faces" against an inverted element is how a loop burns its whole iteration
budget without ever addressing the cause.

An unsatisfied objective means the mesh works but could be better, and the response
is a SIZE FIELD edit, scoped as tightly as the evidence supports.

Three constraints keep the search bounded
-----------------------------------------
  LOCALIZED BEFORE GLOBAL  A global change is only proposed after a localized one
                           has been tried and failed to move the objective.
  BUDGET                   Any action that would refine must declare its estimated
                           element count, and is dropped if it exceeds the budget.
  NO REPEATS               The loop rejects an action whose resulting parameter set
                           has already been meshed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum

from pipeline.sizefield import (RuleKind, SizeFieldSpec, base, clamp, curvature,
                                estimate_element_count, feature, gradation)


class ActionKind(str, Enum):
    # --- strategy changes: the response to a HARD failure ---
    ENABLE_HEALING = "enable_healing"
    INCREASE_HEAL_TOLERANCE = "increase_heal_tolerance"
    TIGHTEN_TOLERANCE = "tighten_tolerance"
    REIMPRINT = "reimprint"
    SWITCH_ALGORITHM = "switch_algorithm"
    INCREASE_OPTIMIZATION = "increase_optimization"
    ESCALATE = "escalate"
    # --- size field edits: the response to an unsatisfied OBJECTIVE ---
    REFINE_FACES = "refine_faces"
    TIGHTEN_CURVATURE = "tighten_curvature"
    TIGHTEN_GRADATION = "tighten_gradation"
    COARSEN_GLOBAL = "coarsen_global"
    REFINE_GLOBAL = "refine_global"
    RAISE_FLOOR = "raise_floor"


STRATEGY_ACTIONS = frozenset({
    ActionKind.ENABLE_HEALING, ActionKind.INCREASE_HEAL_TOLERANCE,
    ActionKind.TIGHTEN_TOLERANCE, ActionKind.REIMPRINT,
    ActionKind.SWITCH_ALGORITHM, ActionKind.INCREASE_OPTIMIZATION,
    ActionKind.ESCALATE,
})

ALGORITHM_LADDER = ("frontal", "delaunay", "meshadapt")

MAX_HEAL_TOLERANCE_FRAC = 1e-2
"""Ceiling on healing aggression, as a fraction of model scale.

Above roughly 1% of model scale healing stops removing artifacts and starts
eating real features -- a 0.7 mm tolerance on a 76 mm part would take genuine
fillets with it. Past this point the decision belongs to a human, because removing
a real feature changes the feature contract."""


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    params: dict = field(default_factory=dict)
    scope: tuple[str, ...] = ()
    rationale: str = ""
    evidence: dict = field(default_factory=dict)
    priority: int = 100
    """Lower runs first. Hard failures outrank objectives by construction."""

    @property
    def is_strategy(self) -> bool:
        return self.kind in STRATEGY_ACTIONS

    @property
    def is_terminal(self) -> bool:
        return self.kind is ActionKind.ESCALATE

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "params": dict(self.params),
                "scope": list(self.scope), "rationale": self.rationale,
                "evidence": dict(self.evidence), "priority": self.priority}

    def describe(self) -> str:
        where = f" on {len(self.scope)} face(s)" if self.scope else " globally"
        return f"{self.kind.value}{where}: {self.rationale}"


# ---------------------------------------------------------------------------
# Applying an action -- pure transform
# ---------------------------------------------------------------------------

def apply_action(action: Action, spec: SizeFieldSpec, strategy):
    """Return (new_spec, new_strategy). Neither input is mutated.

    Immutability matters for the run record: every iteration keeps the exact spec
    it meshed with, so a diff between iterations is meaningful after the fact.
    """
    import dataclasses
    k, p = action.kind, action.params

    if k is ActionKind.ENABLE_HEALING:
        return spec, dataclasses.replace(strategy, heal=True)
    if k is ActionKind.INCREASE_HEAL_TOLERANCE:
        return spec, dataclasses.replace(
            strategy, heal=True,
            heal_tolerance_frac=float(p["heal_tolerance_frac"]))
    if k is ActionKind.TIGHTEN_TOLERANCE:
        return spec, dataclasses.replace(strategy,
                                         geometry_tolerance=float(p["tolerance"]))
    if k is ActionKind.REIMPRINT:
        return spec, dataclasses.replace(strategy, imprint=True)
    if k is ActionKind.SWITCH_ALGORITHM:
        return spec, dataclasses.replace(strategy, algorithm=str(p["algorithm"]))
    if k is ActionKind.INCREASE_OPTIMIZATION:
        return spec, dataclasses.replace(
            strategy, optimize_passes=strategy.optimize_passes + 1,
            smooth_passes=strategy.smooth_passes + 1)
    if k is ActionKind.ESCALATE:
        return spec, strategy

    if k is ActionKind.REFINE_FACES:
        return spec.with_rule(feature(float(p["h"]), scope=action.scope,
                                      note=action.rationale)), strategy
    if k is ActionKind.TIGHTEN_CURVATURE:
        return spec.replace_kind(RuleKind.CURVATURE, curvature(
            chordal_tol=float(p["chordal_tol"]),
            normal_tol_deg=float(p["normal_tol_deg"]),
            min_per_2pi=float(p.get("min_per_2pi", 12.0)),
            scope=action.scope, note=action.rationale)), strategy
    if k is ActionKind.TIGHTEN_GRADATION:
        return spec.replace_kind(RuleKind.GRADATION,
                                 gradation(float(p["max_growth"]),
                                           note=action.rationale)), strategy
    if k in (ActionKind.COARSEN_GLOBAL, ActionKind.REFINE_GLOBAL):
        return spec.replace_kind(RuleKind.BASE,
                                 base(float(p["h"]), note=action.rationale)), strategy
    if k is ActionKind.RAISE_FLOOR:
        cur = spec.get(RuleKind.CLAMP)
        h_max = float(cur.params["h_max"]) if cur else float(p["h_min"]) * 100
        return spec.replace_kind(RuleKind.CLAMP,
                                 clamp(float(p["h_min"]), h_max,
                                       note=action.rationale)), strategy
    raise ValueError(f"unhandled action kind {k}")


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------

CONTRACT_CODES = frozenset({
    "contract.missing_face", "contract.unapproved_removal",
    "contract.unexpected_boundary", "contract.boundary_mismatch",
    "contract.interface_not_conformal", "contract.nonmanifold_skin",
    "contract.inconsistent_orientation",
})

FAILURE_RESPONSES = {
    "validity.nonmanifold": "geometry or imprinting",
    "validity.orientation": "geometry",
    "validity.inverted": "mesher",
    "validity.degenerate_shape": "geometry",
    "validity.duplicate_nodes": "imprinting",
    "validity.not_watertight": "geometry",
    "topology.no_volume": "import",
    "mesh.empty": "mesher",
}


def propose(verdict, face_scores, spec: SizeFieldSpec, strategy, *,
            model_scale: float, face_areas: dict[str, float],
            current_sizes: dict[str, float], budget: int,
            history=None) -> list[Action]:
    """Ranked candidate actions. Never mutates anything.

    history is a list of previously applied Action objects; it is used only to
    escalate from localized to global, never to filter (that is the selector's job).
    """
    history = list(history or [])
    if not verdict.is_valid:
        return _strategy_actions(verdict, strategy, model_scale, history)
    return _objective_actions(verdict, face_scores, spec, strategy,
                              model_scale=model_scale, face_areas=face_areas,
                              current_sizes=current_sizes, budget=budget,
                              history=history)


def _tried(history, kind: ActionKind) -> int:
    return sum(1 for a in history if a.kind is kind)


def _strategy_actions(verdict, strategy, model_scale: float,
                      history) -> list[Action]:
    """Hard failures. Deliberately never returns a sizing action."""
    codes = {f.code: f for f in verdict.failures}
    out: list[Action] = []

    if "validity.nonmanifold" in codes or "validity.duplicate_nodes" in codes:
        f = codes.get("validity.nonmanifold") or codes["validity.duplicate_nodes"]
        if not strategy.imprint:
            out.append(Action(
                ActionKind.REIMPRINT, priority=0,
                rationale="non-manifold or duplicate nodes with imprinting off; "
                          "touching solids are meshing as disconnected parts",
                evidence={"code": f.code, "count": f.count}))
        elif _tried(history, ActionKind.TIGHTEN_TOLERANCE) < 2:
            tol = (strategy.geometry_tolerance or 1e-4 * model_scale) * 0.1
            out.append(Action(
                ActionKind.TIGHTEN_TOLERANCE, {"tolerance": tol}, priority=1,
                rationale="already imprinted, so coincident faces are probably "
                          "outside tolerance and failing to merge",
                evidence={"code": f.code, "count": f.count}))

    if "validity.degenerate_shape" in codes:
        f = codes["validity.degenerate_shape"]
        if not strategy.heal:
            out.append(Action(
                ActionKind.ENABLE_HEALING, priority=0,
                rationale="degenerate elements from sub-tolerance CAD features; "
                          "healing removes slivers that no size field can mesh",
                evidence={"count": f.count, "worst": f.detail.get("worst")}))
        else:
            # "Heal harder" before giving up. Healing is a TOLERANCE, not a switch:
            # the default removes nothing on a part whose artifacts are larger than
            # it, and toggling a boolean cannot express that. Escalating straight to
            # a human after one ineffective attempt wastes the remedy.
            nxt = strategy.heal_tolerance_frac * 10.0
            if nxt <= MAX_HEAL_TOLERANCE_FRAC:
                out.append(Action(
                    ActionKind.INCREASE_HEAL_TOLERANCE,
                    {"heal_tolerance_frac": nxt}, priority=1,
                    rationale=f"degenerate elements persist at a healing tolerance "
                              f"of {strategy.heal_tolerance_frac:.0e} of model "
                              f"scale; raise it to {nxt:.0e}",
                    evidence={"count": f.count,
                              "from": strategy.heal_tolerance_frac, "to": nxt}))
            else:
                out.append(Action(
                    ActionKind.ESCALATE, priority=90,
                    rationale=f"degenerate elements persist at the healing ceiling "
                              f"({MAX_HEAL_TOLERANCE_FRAC:.0e} of model scale). "
                              "Healing harder would start removing real features, "
                              "so defeaturing is a human decision that changes the "
                              "feature contract",
                    evidence={"count": f.count, "worst": f.detail.get("worst"),
                              "heal_tolerance_frac":
                                  strategy.heal_tolerance_frac}))

    if "validity.inverted" in codes:
        f = codes["validity.inverted"]
        if strategy.optimize_passes < 3:
            out.append(Action(
                ActionKind.INCREASE_OPTIMIZATION, priority=2,
                rationale="inverted elements are often recoverable by the mesher's "
                          "own optimization passes",
                evidence={"count": f.count}))
        else:
            nxt = _next_algorithm(strategy.algorithm, history)
            if nxt:
                out.append(Action(
                    ActionKind.SWITCH_ALGORITHM, {"algorithm": nxt}, priority=3,
                    rationale="inverted elements survive optimization; try a "
                              "different meshing algorithm",
                    evidence={"count": f.count, "from": strategy.algorithm}))

    # --- the mesher itself raised ---
    if "mesher.failed" in codes:
        f = codes["mesher.failed"]
        msg = str(f.detail.get("exception", "")).lower()

        if "periodic surface" in msg:
            # Stage 0 already prescribes this: a degenerate or seamed
            # parametrization is where Frontal-Delaunay gives up and MeshAdapt is
            # the robust fallback. Go straight there rather than walking the ladder
            # through Delaunay, which fails the same way.
            if strategy.algorithm != "meshadapt":
                out.append(Action(
                    ActionKind.SWITCH_ALGORITHM, {"algorithm": "meshadapt"},
                    priority=0,
                    rationale="the mesher cannot handle a periodic or degenerate "
                              "parametrization; MeshAdapt is the robust fallback "
                              "for exactly this case",
                    evidence={"exception": f.detail.get("exception")}))
            elif not strategy.heal:
                out.append(Action(
                    ActionKind.ENABLE_HEALING, priority=1,
                    rationale="MeshAdapt also failed on the parametrization; try "
                              "repairing the seam or pole first",
                    evidence={"exception": f.detail.get("exception")}))
            else:
                out.append(Action(
                    ActionKind.ESCALATE, priority=94,
                    rationale="no algorithm can mesh this parametrization even "
                              "after healing; the face needs reparametrization or "
                              "splitting, which changes the geometry",
                    evidence={"exception": f.detail.get("exception")}))

        elif "closed loop" in msg or "1d mesh" in msg:
            # The 1D discretization did not close, which is a boundary/tolerance
            # defect rather than a 2D algorithm choice -- the same free-edge versus
            # seam-edge distinction Stage 0 had to make.
            if not strategy.heal:
                out.append(Action(
                    ActionKind.ENABLE_HEALING, priority=0,
                    rationale="the 1D mesh did not form a closed loop, so curve "
                              "endpoints are outside tolerance; healing sews them",
                    evidence={"exception": f.detail.get("exception")}))
            elif _tried(history, ActionKind.TIGHTEN_TOLERANCE) < 2:
                tol = (strategy.geometry_tolerance or 1e-4 * model_scale) * 10.0
                out.append(Action(
                    ActionKind.TIGHTEN_TOLERANCE, {"tolerance": tol}, priority=1,
                    rationale="loops still do not close after healing; widen the "
                              "import tolerance so near-coincident endpoints merge",
                    evidence={"exception": f.detail.get("exception")}))
            else:
                out.append(Action(
                    ActionKind.ESCALATE, priority=94,
                    rationale="boundary loops do not close even after healing and "
                              "tolerance changes; the CAD topology is broken",
                    evidence={"exception": f.detail.get("exception")}))

        else:
            nxt = _next_algorithm(strategy.algorithm, history)
            if nxt:
                out.append(Action(
                    ActionKind.SWITCH_ALGORITHM, {"algorithm": nxt}, priority=2,
                    rationale="the mesher raised an unrecognized error; try a "
                              "different algorithm before giving up",
                    evidence={"exception": f.detail.get("exception")}))
            else:
                out.append(Action(
                    ActionKind.ESCALATE, priority=95,
                    rationale="every algorithm raised; this geometry needs "
                              "inspection",
                    evidence={"exception": f.detail.get("exception")}))

    if "validity.orientation" in codes or "validity.not_watertight" in codes:
        code = ("validity.orientation" if "validity.orientation" in codes
                else "validity.not_watertight")
        out.append(Action(
            ActionKind.ESCALATE, priority=95,
            rationale=f"{code} is a geometry defect, not a meshing parameter. "
                      "Re-check the topology contract and the CAD itself",
            evidence={"code": code, "count": codes[code].count}))

    # --- contract violations: the mesh disagrees with the CAD ---
    if "contract.missing_face" in codes:
        f = codes["contract.missing_face"]
        lost = tuple(f.detail.get("pids", ()))[:5]
        if lost:
            out.append(Action(
                ActionKind.REFINE_FACES, {"h": model_scale * 1e-3},
                scope=lost, priority=4,
                rationale="required CAD face(s) produced no elements; the "
                          "requested size is larger than the face itself",
                evidence={"code": f.code, "pids": list(lost)}))
        out.append(Action(
            ActionKind.ESCALATE, priority=92,
            rationale="required CAD faces are absent from the mesh. That is lost "
                      "geometry, so it must be resolved before the mesh is used",
            evidence={"code": f.code, "count": f.count}))

    if "contract.unapproved_removal" in codes:
        f = codes["contract.unapproved_removal"]
        out.append(Action(
            ActionKind.ESCALATE, priority=91,
            rationale="healing removed CAD faces that are too large to auto-approve "
                      "as artifacts. Approving a removal changes the feature "
                      "contract, so it is a human decision, not a parameter",
            evidence={"code": f.code, "count": f.count,
                      "pids": f.detail.get("pids", [])}))

    if "contract.interface_not_conformal" in codes:
        f = codes["contract.interface_not_conformal"]
        out.append(Action(
            ActionKind.REIMPRINT, priority=1,
            rationale="an imprinted interface produced no elements; the bodies are "
                      "not sharing nodes and the joint transmits no load",
            evidence={"code": f.code, "pids": f.detail.get("pids", [])}))

    for code in ("contract.unexpected_boundary", "contract.boundary_mismatch",
                 "contract.nonmanifold_skin", "contract.inconsistent_orientation"):
        if code in codes:
            out.append(Action(
                ActionKind.ESCALATE, priority=93,
                rationale=f"{code}: the mesh topology does not match what the CAD "
                          "declares. No sizing or algorithm change fixes a topology "
                          "mismatch",
                evidence={"code": code, "count": codes[code].count}))

    if not out:
        out.append(Action(
            ActionKind.ESCALATE, priority=99,
            rationale="hard failure with no mapped remedy",
            evidence={"codes": sorted(codes)}))
    out.sort(key=lambda a: a.priority)
    return out


def _next_algorithm(current: str, history) -> str | None:
    used = {a.params.get("algorithm") for a in history
            if a.kind is ActionKind.SWITCH_ALGORITHM}
    used.add(current)
    for cand in ALGORITHM_LADDER:
        if cand not in used:
            return cand
    return None


def _objective_actions(verdict, face_scores, spec, strategy, *, model_scale,
                       face_areas, current_sizes, budget, history) -> list[Action]:
    """Unsatisfied objectives, worst attainment first."""
    from backends.base import worst_faces

    out: list[Action] = []
    unsatisfied = verdict.unsatisfied()

    for rank, obj in enumerate(unsatisfied):
        pri = 10 + rank
        name = obj.name

        if name in ("chordal", "normal_deviation"):
            metric = ("chordal_relative" if name == "chordal"
                      else "normal_p99_deg")
            scope = tuple(worst_faces(face_scores, metric, 5))
            cur = spec.get(RuleKind.CURVATURE)
            chord = float(cur.params["chordal_tol"]) if cur else 0.02 * model_scale
            normal = float(cur.params["normal_tol_deg"]) if cur else 12.0
            # Tighten the binding bound only. Normal error is O(h/R) and chordal is
            # O(h^2/R), so halving h halves one and quarters the other; tightening
            # both together over-refines.
            if name == "normal_deviation":
                normal = max(normal * 0.6, 1.0)
            else:
                chord = chord * 0.5
            act = Action(
                ActionKind.TIGHTEN_CURVATURE,
                {"chordal_tol": chord, "normal_tol_deg": normal},
                scope=scope if scope else (), priority=pri,
                rationale=f"{name} {obj.value:.4g} misses {obj.target:.4g}; "
                          f"tighten the {'normal' if name == 'normal_deviation' else 'chordal'} "
                          "bound, which is the one binding here",
                evidence={"objective": name, "value": obj.value,
                          "target": obj.target, "worst_faces": list(scope)})
            out.append(act)

        elif name == "gradation":
            cur = spec.get(RuleKind.GRADATION)
            g = float(cur.params["max_growth"]) if cur else 1.4
            out.append(Action(
                ActionKind.TIGHTEN_GRADATION,
                {"max_growth": max(1.05, 1.0 + (g - 1.0) * 0.6)}, priority=pri,
                rationale=f"gradation {obj.value:.3g} exceeds {obj.target:.3g}; "
                          "a lower growth rate lengthens the ramps",
                evidence={"objective": name, "value": obj.value,
                          "target": obj.target}))

        elif name in ("min_shape", "min_angle", "max_angle"):
            if strategy.optimize_passes < 3:
                out.append(Action(
                    ActionKind.INCREASE_OPTIMIZATION, priority=pri,
                    rationale=f"{name} {obj.value:.3g} misses {obj.target:.3g}; "
                              "try more mesher optimization before changing sizing",
                    evidence={"objective": name, "value": obj.value}))
            scope = tuple(worst_faces(face_scores, "shape_min", 5))
            n_local = _tried(history, ActionKind.REFINE_FACES)
            if n_local >= 1:
                # LOCALIZED BEFORE GLOBAL, and the escalation that makes the rule
                # meaningful. Refining only the worst faces cannot help when the
                # problem is spread across the model, and without this the loop
                # re-proposes a slightly different local refinement every iteration
                # and plateaus while looking busy.
                cur = spec.get(RuleKind.BASE)
                h0 = float(cur.params["h"]) if cur else 0.05 * model_scale
                out.append(_budgeted_refine(
                    tuple(face_areas), h0 * 0.7, face_areas, current_sizes,
                    budget, pri,
                    f"{name} {obj.value:.3g} still misses {obj.target:.3g} after "
                    f"{n_local} localized refinement(s); reduce the base size",
                    {"objective": name, "value": obj.value,
                     "localized_attempts": n_local},
                    kind=ActionKind.REFINE_GLOBAL, h_param=h0 * 0.7))
            if scope:
                h = min((current_sizes.get(p, model_scale) for p in scope),
                        default=model_scale) * 0.6
                out.append(_budgeted_refine(
                    scope, h, face_areas, current_sizes, budget, pri + 1,
                    f"{name} {obj.value:.3g} misses {obj.target:.3g}; refine the "
                    "worst-shaped faces",
                    {"objective": name, "value": obj.value}))

        elif name == "size_in_band":
            d = obj.detail or {}
            if d.get("frac_oversized", 0) > d.get("frac_undersized", 0):
                cur = spec.get(RuleKind.BASE)
                h = float(cur.params["h"]) if cur else 0.05 * model_scale
                out.append(Action(
                    ActionKind.COARSEN_GLOBAL, {"h": h * 0.8}, priority=pri,
                    rationale="elements are systematically larger than requested; "
                              "lower the base so the request is attainable",
                    evidence={"objective": name, **d}))
            else:
                cur = spec.get(RuleKind.CLAMP)
                lo = float(cur.params["h_min"]) if cur else 0.001 * model_scale
                out.append(Action(
                    ActionKind.RAISE_FLOOR, {"h_min": lo * 2.0}, priority=pri,
                    rationale="elements are systematically finer than requested; "
                              "raise the floor rather than pay for over-refinement",
                    evidence={"objective": name, **d}))

    out = [a for a in out if a is not None]
    out.sort(key=lambda a: a.priority)
    return out


def _budgeted_refine(scope, h, face_areas, current_sizes, budget, priority,
                     rationale, evidence, *, kind=ActionKind.REFINE_FACES,
                     h_param=None) -> Action | None:
    """A refinement action must declare its cost, and is dropped if unaffordable.

    Discovering that a refinement produced two million elements after the fact is
    not an outcome to iterate on; it is a decision that has to be taken up front.
    """
    projected = dict(current_sizes)
    for pid in scope:
        projected[pid] = h
    est = estimate_element_count(projected, face_areas)
    if est > budget:
        return None
    is_global = kind is ActionKind.REFINE_GLOBAL
    return Action(kind, {"h": float(h_param if h_param is not None else h)},
                  scope=() if is_global else tuple(scope),
                  priority=priority, rationale=rationale,
                  evidence={**evidence, "estimated_elements": est,
                            "budget": budget})


# ---------------------------------------------------------------------------
# Selection -- the agent seam
# ---------------------------------------------------------------------------

class Selector:
    """Chooses one action from ranked candidates. Deterministic by default.

    An LLM would subclass this and choose AMONG the candidates. It would never
    propose raw parameters, and it never decides whether a mesh is acceptable --
    that stays with quality.criteria, which is deterministic.
    """
    name = "selector"

    def select(self, candidates: list[Action], history) -> Action | None:
        raise NotImplementedError


class DeterministicSelector(Selector):
    name = "deterministic"

    def select(self, candidates: list[Action], history) -> Action | None:
        """Highest priority candidate that has not already been applied."""
        seen = {(a.kind, tuple(sorted(a.scope)),
                 tuple(sorted(a.params.items()))) for a in history}
        for a in sorted(candidates, key=lambda x: x.priority):
            key = (a.kind, tuple(sorted(a.scope)), tuple(sorted(a.params.items())))
            if key not in seen:
                return a
        return None


class LLMSelector(Selector):
    """Placeholder for the agent seam. Intentionally not implemented.

    The interface is fixed now so the loop never has to change to accept one: same
    signature, same closed action set, same deterministic acceptance test.
    """
    name = "llm"

    def __init__(self, client=None):
        self.client = client

    def select(self, candidates: list[Action], history) -> Action | None:
        raise NotImplementedError(
            "LLM selection is not wired up. The pipeline is deterministic by "
            "design; use DeterministicSelector.")
