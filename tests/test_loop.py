"""
Adaptive loop and policy tests. PURE -- no gmsh.

A scripted fake backend lets the control logic be tested where it actually breaks:
meshes that DEGRADE, failures that never clear, parameter sets that come back
around, and budgets that make every remaining action unaffordable. None of those
can be provoked reliably from real CAD, and a loop only ever run on cases that
converge is untested in exactly the situations it exists to handle.
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backends.base import (MeshProvenance, MeshResult, MeshStats,
                           MeshingStrategy)
from pipeline import policy as P
from pipeline.loop import AdaptiveMesher
from pipeline.session import Iteration, Outcome, RunRecord
from pipeline.sizefield import RuleKind, SizeFieldSpec
from quality.core import SurfaceMesh
from quality.criteria import Criteria


# --------------------------------------------------------------------------
# Fake backend
# --------------------------------------------------------------------------

def _grid_mesh(n: int, size: float = 1.0, skew: float = 0.0):
    """An n x n grid of quads split into triangles, optionally skewed.

    skew shears the interior so shape quality degrades predictably, which is how a
    'this got worse' scenario is scripted.
    """
    xs = np.linspace(0, size, n + 1)
    pts, idx = [], {}
    for j, y in enumerate(xs):
        for i, x in enumerate(xs):
            dx = skew * size * (0.5 - abs(i / n - 0.5)) if 0 < i < n else 0.0
            idx[(i, j)] = len(pts)
            pts.append([x + dx, y, 0.0])
    tris = []
    for j in range(n):
        for i in range(n):
            a, b = idx[(i, j)], idx[(i + 1, j)]
            c, d = idx[(i + 1, j + 1)], idx[(i, j + 1)]
            tris += [[a, b, c], [a, c, d]]
    return np.asarray(pts, float), np.asarray(tris, np.int64)


class FakeBackend:
    """MeshingBackend with scripted per-iteration behaviour."""

    name = "fake"

    def __init__(self, script=None, *, n_faces: int = 2, scale: float = 100.0,
                 fail_load: bool = False):
        self.script = list(script or [])
        self.n_faces = n_faces
        self.scale = scale
        self.fail_load = fail_load
        self.calls = 0
        self.loads = 0
        self.strategies: list[MeshingStrategy] = []
        self.requests: list = []

    # -- protocol ---
    def load(self, path, strategy):
        if self.fail_load:
            raise RuntimeError("synthetic load failure")
        self.loads += 1
        self.strategies.append(strategy)
        return {"imported_faces": self.n_faces, "imprinted": strategy.imprint}

    def model_scale(self):
        return self.scale

    def measure_thickness(self):
        return {}

    def face_info(self):
        return {f"F{i}": {"curvature_max": 0.0, "thickness": None,
                          "area": 100.0} for i in range(self.n_faces)}

    def face_adjacency(self):
        pids = [f"F{i}" for i in range(self.n_faces)]
        return {p: set(pids) - {p} for p in pids}

    def surface_for(self, pid):
        return None

    # NOTE: FakeBackend deliberately does NOT implement body_kind_inputs.
    # The loop then skips contract checks, which keeps these tests focused on
    # control logic. Contracts are covered directly in tests/test_contracts.py and
    # end to end by ContractBackend there -- entangling the two would make every
    # loop test depend on getting a fake body kind right.

    def mesh(self, request):
        self.requests.append(request)
        step = (self.script[self.calls] if self.calls < len(self.script)
                else self.script[-1] if self.script else {})
        self.calls += 1

        n = int(step.get("n", 6))
        skew = float(step.get("skew", 0.0))
        v, t = _grid_mesh(n, skew=skew)
        if step.get("invert"):
            t = t.copy()
            t[0] = t[0][[0, 2, 1]]
        if step.get("empty"):
            v, t = np.zeros((0, 3)), np.zeros((0, 3), np.int64)

        mesh = SurfaceMesh(vertices=v, triangles=t, method="fake",
                           meta={"target_size": 1.0})
        half = len(t) // 2
        prov = MeshProvenance(
            tri_pid=["F0"] * half + ["F1"] * (len(t) - half),
            requested_size={f"F{i}": 1.0 for i in range(self.n_faces)},
            tri_field_size=[float(step.get("field", 1.0 / max(n, 1)))] * len(t))
        return MeshResult(mesh=mesh, provenance=prov,
                          stats=MeshStats(n_triangles=len(t), backend=self.name),
                          request_fingerprint=request.fingerprint())


LOOSE = Criteria(min_shape=0.30, min_angle_deg=20.0, max_angle_deg=130.0,
                 min_size_in_band=0.0, max_gradation=3.0,
                 max_chordal_relative=1.0, max_normal_deg=90.0)
STRICT = Criteria(min_shape=0.99, min_angle_deg=59.0, min_size_in_band=0.0,
                  max_gradation=3.0)


# --------------------------------------------------------------------------
# Termination
# --------------------------------------------------------------------------

def test_accepts_immediately_when_ready():
    be = FakeBackend([{"n": 6}])
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False).run("m.step")
    assert rec.outcome is Outcome.ACCEPTED, rec.summary()
    assert len(rec.iterations) == 1
    assert rec.best.ready and rec.best_index == 0


def test_stops_on_max_iterations():
    be = FakeBackend([{"n": 6}])
    rec = AdaptiveMesher(be, criteria=STRICT, fidelity=False,
                         max_iterations=3).run("m.step")
    assert rec.outcome in (Outcome.MAX_ITERATIONS, Outcome.NO_ACTION,
                           Outcome.PLATEAU, Outcome.OSCILLATION), rec.outcome
    assert len(rec.iterations) <= 3


def test_oscillation_is_detected_not_looped_forever():
    """A parameter set that comes back around must stop the run."""
    rec = RunRecord()
    seen = set()
    fps = ["a", "b", "a"]
    outcome = None
    for f in fps:
        if f in seen:
            outcome = Outcome.OSCILLATION
            break
        seen.add(f)
    assert outcome is Outcome.OSCILLATION

    be = FakeBackend([{"n": 6}])
    m = AdaptiveMesher(be, criteria=STRICT, fidelity=False, max_iterations=20)
    r = m.run("m.step")
    assert len(r.iterations) < 20, "must not run to the cap on a static backend"


def test_plateau_stops_the_run():
    """A lever that stopped working burns budget for nothing."""
    rec = RunRecord()
    for i, s in enumerate((0.50, 0.80, 0.801, 0.8012)):
        rec.add(Iteration(index=i, is_valid=True, score=s))
    assert rec.plateaued(window=2, eps=1e-2)
    rec2 = RunRecord()
    for i, s in enumerate((0.2, 0.4, 0.6, 0.9)):
        rec2.add(Iteration(index=i, is_valid=True, score=s))
    assert not rec2.plateaued(window=2, eps=1e-2)


def test_backend_load_failure_is_recorded_not_raised():
    be = FakeBackend([{"n": 6}], fail_load=True)
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False).run("m.step")
    assert rec.outcome is Outcome.BACKEND_ERROR
    assert rec.best is None
    assert any("load failed" in n for n in rec.notes)


def test_over_budget_is_refused_before_meshing():
    """Refusing after the fact costs the whole iteration and teaches nothing."""
    be = FakeBackend([{"n": 6}])
    m = AdaptiveMesher(be, criteria=LOOSE, fidelity=False, budget=10,
                       max_iterations=2)
    rec = m.run("m.step")
    assert be.calls == 0, "the backend must not be invoked when over budget"
    assert any("over the" in (it.note or "") for it in rec.iterations)


# --------------------------------------------------------------------------
# Best-valid retention
# --------------------------------------------------------------------------

def test_returns_the_best_valid_iteration_not_the_last():
    """The final iteration is usually the most aggressive edit and can be worse."""
    rec = RunRecord()
    rec.add(Iteration(index=0, is_valid=True, score=0.60))
    rec.add(Iteration(index=1, is_valid=True, score=0.95))
    rec.add(Iteration(index=2, is_valid=True, score=0.30))
    assert rec.best_index == 1


def test_invalid_iterations_are_never_best():
    rec = RunRecord()
    rec.add(Iteration(index=0, is_valid=True, score=0.40))
    rec.add(Iteration(index=1, is_valid=False, score=0.0))
    assert rec.best_index == 0
    empty = RunRecord()
    empty.add(Iteration(index=0, is_valid=False, score=0.0))
    assert empty.best is None


def test_ready_outranks_a_higher_scoring_unready_iteration():
    rec = RunRecord()
    rec.add(Iteration(index=0, is_valid=True, ready=False, score=0.99))
    rec.add(Iteration(index=1, is_valid=True, ready=True, score=0.80))
    assert rec.best_index == 1


# --------------------------------------------------------------------------
# Policy: hard failures vs objectives are DIFFERENT code paths
# --------------------------------------------------------------------------

class _Obj:
    def __init__(self, name, value, target, satisfied=False, normalized=0.5,
                 detail=None):
        self.name, self.value, self.target = name, value, target
        self.satisfied, self.normalized = satisfied, normalized
        self.detail = detail or {}


class _Fail:
    def __init__(self, code, count=1, detail=None):
        self.code, self.count = code, count
        self.message, self.detail = code, detail or {}


class _Verdict:
    def __init__(self, valid, failures=(), objectives=()):
        self.is_valid = valid
        self.failures = list(failures)
        self.objectives = list(objectives)

    def unsatisfied(self):
        return sorted((o for o in self.objectives if not o.satisfied),
                      key=lambda o: o.normalized)


def _propose(verdict, scores=(), spec=None, strategy=None, budget=1_000_000,
             history=()):
    spec = spec or SizeFieldSpec.default(100.0)
    strategy = strategy or MeshingStrategy()
    return P.propose(verdict, list(scores), spec, strategy, model_scale=100.0,
                     face_areas={"F0": 100.0, "F1": 100.0},
                     current_sizes={"F0": 5.0, "F1": 5.0}, budget=budget,
                     history=list(history))


def test_hard_failure_never_proposes_a_sizing_action():
    """THE central rule. Refining against an inverted element burns the budget."""
    v = _Verdict(False, [_Fail("validity.inverted", 3)],
                 [_Obj("min_shape", 0.1, 0.3)])
    acts = _propose(v)
    assert acts, "a hard failure must still produce a remedy"
    assert all(a.is_strategy for a in acts), [a.kind for a in acts]
    assert not any(a.kind is P.ActionKind.REFINE_FACES for a in acts)


def test_degenerate_shape_proposes_healing_then_raises_tolerance():
    """Enable healing, then heal HARDER, and only then ask a human.

    Escalating after one ineffective heal wastes the remedy: healing is a tolerance
    and the first value may simply be too small for the artifact.
    """
    v = _Verdict(False, [_Fail("validity.degenerate_shape", 138,
                               {"worst": 0.018})])
    first = _propose(v, strategy=MeshingStrategy(heal=False))
    assert first[0].kind is P.ActionKind.ENABLE_HEALING

    after = _propose(v, strategy=MeshingStrategy(heal=True))
    assert after[0].kind is P.ActionKind.INCREASE_HEAL_TOLERANCE

    ceiling = _propose(v, strategy=MeshingStrategy(
        heal=True, heal_tolerance_frac=P.MAX_HEAL_TOLERANCE_FRAC))
    assert ceiling[0].kind is P.ActionKind.ESCALATE
    assert "defeatur" in ceiling[0].rationale.lower()


def test_nonmanifold_proposes_imprint_then_tolerance():
    v = _Verdict(False, [_Fail("validity.nonmanifold", 30)])
    off = _propose(v, strategy=MeshingStrategy(imprint=False))
    assert off[0].kind is P.ActionKind.REIMPRINT

    on = _propose(v, strategy=MeshingStrategy(imprint=True))
    assert on[0].kind is P.ActionKind.TIGHTEN_TOLERANCE
    assert on[0].params["tolerance"] > 0


def test_orientation_failure_escalates_rather_than_guessing():
    v = _Verdict(False, [_Fail("validity.orientation", 12)])
    acts = _propose(v)
    assert acts[0].kind is P.ActionKind.ESCALATE


def test_unmapped_failure_still_yields_an_action():
    v = _Verdict(False, [_Fail("validity.something_new", 1)])
    acts = _propose(v)
    assert acts and acts[0].kind is P.ActionKind.ESCALATE


def test_objectives_produce_sizing_actions_worst_first():
    from backends.base import FaceScore
    v = _Verdict(True, [], [
        _Obj("min_shape", 0.2, 0.3, normalized=0.66),
        _Obj("normal_deviation", 40.0, 15.0, normalized=0.20),
    ])
    scores = [FaceScore("F0", 10, 0.2, 0.5, 12, normal_p99_deg=40.0),
              FaceScore("F1", 10, 0.9, 0.9, 50, normal_p99_deg=2.0)]
    acts = _propose(v, scores)
    assert acts, "unsatisfied objectives must produce candidates"
    assert acts[0].evidence.get("objective") == "normal_deviation", \
        "worst attainment must be addressed first"
    assert not any(a.is_strategy and a.is_terminal for a in acts)


def test_curvature_action_tightens_only_the_binding_bound():
    """Normal error is O(h/R), chordal O(h^2/R). Tightening both over-refines."""
    from backends.base import FaceScore
    spec = SizeFieldSpec.default(100.0)
    cur = spec.get(RuleKind.CURVATURE).params
    scores = [FaceScore("F0", 10, 0.9, 0.9, 50, normal_p99_deg=40.0,
                        chordal_relative=0.01)]

    v_n = _Verdict(True, [], [_Obj("normal_deviation", 40.0, 15.0)])
    a = next(x for x in _propose(v_n, scores, spec)
             if x.kind is P.ActionKind.TIGHTEN_CURVATURE)
    assert a.params["normal_tol_deg"] < cur["normal_tol_deg"]
    assert a.params["chordal_tol"] == cur["chordal_tol"], "chordal untouched"

    v_c = _Verdict(True, [], [_Obj("chordal", 0.2, 0.05)])
    b = next(x for x in _propose(v_c, scores, spec)
             if x.kind is P.ActionKind.TIGHTEN_CURVATURE)
    assert b.params["chordal_tol"] < cur["chordal_tol"]
    assert b.params["normal_tol_deg"] == cur["normal_tol_deg"]


def test_refinement_is_dropped_when_it_exceeds_the_budget():
    from backends.base import FaceScore
    v = _Verdict(True, [], [_Obj("min_shape", 0.1, 0.3)])
    scores = [FaceScore("F0", 10, 0.1, 0.3, 5)]
    rich = _propose(v, scores, budget=10_000_000)
    poor = _propose(v, scores, budget=1)
    assert any(a.kind is P.ActionKind.REFINE_FACES for a in rich)
    assert not any(a.kind is P.ActionKind.REFINE_FACES for a in poor)


def test_refinement_declares_its_estimated_cost():
    from backends.base import FaceScore
    v = _Verdict(True, [], [_Obj("min_shape", 0.1, 0.3)])
    a = next(x for x in _propose(v, [FaceScore("F0", 10, 0.1, 0.3, 5)])
             if x.kind is P.ActionKind.REFINE_FACES)
    assert a.evidence["estimated_elements"] > 0
    assert a.evidence["budget"] > 0


def test_size_in_band_direction_depends_on_the_evidence():
    over = _Verdict(True, [], [_Obj("size_in_band", 0.3, 0.9,
                                    detail={"frac_oversized": 0.6,
                                            "frac_undersized": 0.1})])
    under = _Verdict(True, [], [_Obj("size_in_band", 0.3, 0.9,
                                     detail={"frac_oversized": 0.1,
                                             "frac_undersized": 0.6})])
    assert any(a.kind is P.ActionKind.COARSEN_GLOBAL for a in _propose(over))
    assert any(a.kind is P.ActionKind.RAISE_FLOOR for a in _propose(under))


# --------------------------------------------------------------------------
# Applying actions
# --------------------------------------------------------------------------

def test_apply_action_does_not_mutate_its_inputs():
    """The run record keeps each iteration's exact spec; mutation would corrupt it."""
    spec = SizeFieldSpec.default(100.0)
    strat = MeshingStrategy()
    before_fp, before_heal = spec.fingerprint(), strat.heal
    s2, st2 = P.apply_action(P.Action(P.ActionKind.ENABLE_HEALING), spec, strat)
    assert spec.fingerprint() == before_fp and strat.heal == before_heal
    assert st2.heal is True and s2.fingerprint() == before_fp


def test_apply_action_covers_every_kind():
    spec = SizeFieldSpec.default(100.0)
    strat = MeshingStrategy()
    params = {
        P.ActionKind.TIGHTEN_TOLERANCE: {"tolerance": 1e-3},
        P.ActionKind.SWITCH_ALGORITHM: {"algorithm": "delaunay"},
        P.ActionKind.REFINE_FACES: {"h": 0.5},
        P.ActionKind.TIGHTEN_CURVATURE: {"chordal_tol": 0.01,
                                         "normal_tol_deg": 6.0},
        P.ActionKind.TIGHTEN_GRADATION: {"max_growth": 1.2},
        P.ActionKind.COARSEN_GLOBAL: {"h": 3.0},
        P.ActionKind.REFINE_GLOBAL: {"h": 1.5},
        P.ActionKind.INCREASE_HEAL_TOLERANCE: {"heal_tolerance_frac": 1e-3},
        P.ActionKind.RAISE_FLOOR: {"h_min": 0.2},
    }
    for kind in P.ActionKind:
        a = P.Action(kind, params.get(kind, {}), scope=("F0",))
        s, st = P.apply_action(a, spec, strat)
        assert s is not None and st is not None, kind


def test_switch_algorithm_walks_the_ladder_without_repeating():
    hist = [P.Action(P.ActionKind.SWITCH_ALGORITHM, {"algorithm": "delaunay"})]
    nxt = P._next_algorithm("frontal", hist)
    assert nxt == "meshadapt", nxt
    assert P._next_algorithm("frontal", hist + [
        P.Action(P.ActionKind.SWITCH_ALGORITHM,
                 {"algorithm": "meshadapt"})]) is None


# --------------------------------------------------------------------------
# Selector -- the agent seam
# --------------------------------------------------------------------------

def test_deterministic_selector_skips_already_applied_actions():
    sel = P.DeterministicSelector()
    a = P.Action(P.ActionKind.ENABLE_HEALING, priority=0)
    b = P.Action(P.ActionKind.INCREASE_OPTIMIZATION, priority=5)
    assert sel.select([a, b], []) is a
    assert sel.select([a, b], [a]) is b
    assert sel.select([a, b], [a, b]) is None


def test_deterministic_selector_respects_priority():
    sel = P.DeterministicSelector()
    lo = P.Action(P.ActionKind.ESCALATE, priority=99)
    hi = P.Action(P.ActionKind.ENABLE_HEALING, priority=0)
    assert sel.select([lo, hi], []) is hi


def test_llm_selector_is_an_unimplemented_seam():
    """Fixed interface now, so the loop never changes to accept one."""
    try:
        P.LLMSelector().select([], [])
    except NotImplementedError as e:
        assert "deterministic" in str(e).lower()
    else:
        raise AssertionError("LLMSelector must not silently do something")


def test_selector_interface_is_uniform():
    for cls in (P.DeterministicSelector, P.LLMSelector):
        assert hasattr(cls, "select") and hasattr(cls, "name")


# --------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------

def test_record_serializes_with_evidence_and_decisions():
    be = FakeBackend([{"n": 6}])
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False).run("m.step")
    import json
    d = json.loads(rec.to_json())
    assert d["outcome"] == "accepted" and d["succeeded"] is True
    assert d["n_iterations"] == 1
    it = d["iterations"][0]
    for key in ("request_fingerprint", "size_field", "strategy", "objectives",
                "estimated_elements"):
        assert key in it, key
    assert d["criteria"]["min_shape"] == LOOSE.min_shape


def test_record_summary_is_readable():
    rec = RunRecord(model="m.step", backend="fake", selector="deterministic")
    rec.add(Iteration(index=0, is_valid=True, score=0.5, n_triangles=100,
                      action={"kind": "refine_faces", "rationale": "because"}))
    rec.outcome = Outcome.MAX_ITERATIONS
    text = rec.summary()
    assert "refine_faces" in text and "max_iterations" in text


def test_outcome_success_flag():
    assert Outcome.ACCEPTED.succeeded
    for o in (Outcome.PLATEAU, Outcome.OSCILLATION, Outcome.ESCALATED,
              Outcome.NO_ACTION, Outcome.MAX_ITERATIONS,
              Outcome.BUDGET_EXHAUSTED, Outcome.BACKEND_ERROR):
        assert not o.succeeded


# --------------------------------------------------------------------------
# Fidelity objectives must actually reach the verdict
# --------------------------------------------------------------------------

def test_fidelity_objectives_are_aggregated_from_per_face_scores():
    """REGRESSION: assess() takes ONE surface, a CAD model has one per face.

    Without aggregation the chordal and normal objectives are never created, so the
    policy's fidelity branches are unreachable and the loop can never act on
    geometric fidelity -- silently, since everything else still scores.
    """
    from backends.base import FaceScore
    from pipeline.loop import _add_fidelity_objectives

    v = _Verdict(True, [], [_Obj("min_shape", 0.9, 0.3, satisfied=True)])
    scores = [FaceScore("F0", 10, 0.9, 0.9, 50, chordal_relative=0.01,
                        normal_p99_deg=3.0),
              FaceScore("F1", 10, 0.9, 0.9, 50, chordal_relative=0.40,
                        normal_p99_deg=44.0)]
    _add_fidelity_objectives(v, scores, LOOSE)
    names = {o.name for o in v.objectives}
    assert {"chordal", "normal_deviation"} <= names, names


def test_fidelity_uses_the_worst_face_not_the_average():
    """One face cutting across a fillet is a defect however good the others are."""
    from backends.base import FaceScore
    from pipeline.loop import _add_fidelity_objectives

    v = _Verdict(True, [], [])
    scores = [FaceScore(f"F{i}", 10, 0.9, 0.9, 50, chordal_relative=0.001,
                        normal_p99_deg=1.0) for i in range(20)]
    scores.append(FaceScore("BAD", 10, 0.9, 0.9, 50, chordal_relative=0.5,
                            normal_p99_deg=60.0))
    _add_fidelity_objectives(v, scores, Criteria())
    ch = next(o for o in v.objectives if o.name == "chordal")
    nd = next(o for o in v.objectives if o.name == "normal_deviation")
    assert ch.value == 0.5 and nd.value == 60.0
    assert not ch.satisfied and not nd.satisfied


def test_fidelity_absent_when_no_face_measured_it():
    from backends.base import FaceScore
    from pipeline.loop import _add_fidelity_objectives
    v = _Verdict(True, [], [])
    _add_fidelity_objectives(v, [FaceScore("F0", 10, 0.9, 0.9, 50)], Criteria())
    assert not any(o.name in ("chordal", "normal_deviation")
                   for o in v.objectives), "NaN must not become a fake objective"


def test_fidelity_objectives_are_not_duplicated():
    from backends.base import FaceScore
    from pipeline.loop import _add_fidelity_objectives
    v = _Verdict(True, [], [_Obj("chordal", 0.01, 0.05, satisfied=True)])
    _add_fidelity_objectives(v, [FaceScore("F0", 10, 0.9, 0.9, 50,
                                           chordal_relative=0.9)], Criteria())
    assert sum(1 for o in v.objectives if o.name == "chordal") == 1


# --------------------------------------------------------------------------
# Closed-loop convergence: a backend that RESPONDS to the request
# --------------------------------------------------------------------------

class ResponsiveBackend(FakeBackend):
    """Mesh quality improves as the requested base size shrinks.

    The scripted FakeBackend ignores the request, which tests termination but not
    CONTROL. This one closes the loop: the policy's edits change the mesh, so a run
    either converges or demonstrably fails to.
    """

    def __init__(self, *, threshold: float = 3.0, **kw):
        super().__init__(**kw)
        self.threshold = threshold
        self.bases: list[float] = []

    def mesh(self, request):
        sizes = request.size_field.resolve(self.face_info(), smooth=False)
        h = min(sizes.values()) if sizes else 5.0
        self.bases.append(h)
        self.requests.append(request)
        self.calls += 1
        # Finer request -> denser, better-shaped mesh.
        n = max(4, min(40, int(round(20.0 / max(h, 0.25)))))
        skew = 0.0 if h <= self.threshold else 0.55
        v, t = _grid_mesh(n, skew=skew)
        mesh = SurfaceMesh(vertices=v, triangles=t, method="responsive",
                           meta={"target_size": h})
        half = len(t) // 2
        prov = MeshProvenance(
            tri_pid=["F0"] * half + ["F1"] * (len(t) - half),
            requested_size={f"F{i}": h for i in range(self.n_faces)},
            tri_field_size=[1.0 / n] * len(t))
        return MeshResult(mesh=mesh, provenance=prov,
                          stats=MeshStats(n_triangles=len(t), backend=self.name),
                          request_fingerprint=request.fingerprint())


CONVERGE = Criteria(min_shape=0.60, min_angle_deg=30.0, max_angle_deg=100.0,
                    min_size_in_band=0.0, max_gradation=3.0,
                    max_chordal_relative=1.0, max_normal_deg=90.0)


def test_loop_converges_when_the_backend_responds():
    """The end-to-end control test: bad first mesh, accepted after adaptation."""
    be = ResponsiveBackend(threshold=3.0, scale=100.0)
    rec = AdaptiveMesher(be, criteria=CONVERGE, fidelity=False,
                         max_iterations=8).run("m.step")
    assert rec.iterations[0].score < 1.0, "the first mesh must not already pass"
    assert rec.outcome is Outcome.ACCEPTED, rec.summary()
    assert rec.best.ready
    assert len(rec.iterations) >= 2, "it has to have actually adapted"


def test_loop_records_what_it_changed_each_iteration():
    be = ResponsiveBackend(threshold=3.0, scale=100.0)
    rec = AdaptiveMesher(be, criteria=CONVERGE, fidelity=False,
                         max_iterations=8).run("m.step")
    acted = [it for it in rec.iterations if it.action]
    assert acted, "at least one iteration must record a decision"
    for it in acted:
        assert it.action["rationale"], "every action must explain itself"
    later = [it for it in rec.iterations[1:] if it.spec_diff]
    assert later, "spec diffs must show what changed between iterations"


def test_loop_never_repeats_a_parameter_set():
    be = ResponsiveBackend(threshold=0.0, scale=100.0)   # never satisfiable
    rec = AdaptiveMesher(be, criteria=STRICT, fidelity=False,
                         max_iterations=10).run("m.step")
    fps = [it.request_fingerprint for it in rec.iterations]
    assert len(fps) == len(set(fps)), fps
    assert rec.outcome is not Outcome.ACCEPTED


def test_unsatisfiable_run_still_returns_the_best_valid_mesh():
    """A run that cannot reach READY must not throw away a usable mesh."""
    be = ResponsiveBackend(threshold=0.0, scale=100.0)
    rec = AdaptiveMesher(be, criteria=STRICT, fidelity=False,
                         max_iterations=6).run("m.step")
    assert not rec.outcome.succeeded
    assert rec.best is not None, "valid-but-not-ready meshes were produced"
    assert rec.best.is_valid and not rec.best.ready


def test_localized_refinement_escalates_to_global():
    """LOCALIZED BEFORE GLOBAL needs the escalation to be meaningful.

    Refining only the worst faces cannot help when the problem is spread across the
    model. Without escalation the loop re-proposes a slightly different local
    refinement every iteration and plateaus while looking busy -- observed as base
    sizes 5.0, 5.0, 5.0 across three iterations.
    """
    from backends.base import FaceScore
    v = _Verdict(True, [], [_Obj("min_angle", 24.0, 30.0)])
    scores = [FaceScore("F0", 10, 0.3, 0.5, 24), FaceScore("F1", 10, 0.9, 0.9, 50)]

    first = _propose(v, scores, history=[])
    assert not any(a.kind is P.ActionKind.REFINE_GLOBAL for a in first), \
        "the first attempt must stay local"

    hist = [P.Action(P.ActionKind.REFINE_FACES, {"h": 3.0}, scope=("F0",))]
    after = _propose(v, scores, history=hist)
    g = next(a for a in after if a.kind is P.ActionKind.REFINE_GLOBAL)
    assert g.scope == (), "a global action carries no scope"
    assert g.evidence["localized_attempts"] == 1
    assert g.priority <= min(a.priority for a in after
                             if a.kind is P.ActionKind.REFINE_FACES)


def test_global_refinement_also_respects_the_budget():
    from backends.base import FaceScore
    v = _Verdict(True, [], [_Obj("min_angle", 24.0, 30.0)])
    scores = [FaceScore("F0", 10, 0.3, 0.5, 24)]
    hist = [P.Action(P.ActionKind.REFINE_FACES, {"h": 3.0}, scope=("F0",))]
    assert not any(a.kind is P.ActionKind.REFINE_GLOBAL
                   for a in _propose(v, scores, history=hist, budget=1))


# --------------------------------------------------------------------------
# Observability: was the action actually applied?
# --------------------------------------------------------------------------

def test_reload_receives_the_updated_strategy():
    """A geometry-changing action must reach the backend, not just the record."""
    from quality.criteria import Criteria

    class Degenerate(FakeBackend):
        def mesh(self, request):
            self.requests.append(request); self.calls += 1
            v = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.0005, 0],
                          [0, 1, 0], [1, 1, 0]], float)
            t = np.array([[0, 1, 2], [0, 3, 4]])
            mesh = SurfaceMesh(vertices=v, triangles=t, method="degen",
                               meta={"target_size": 1.0})
            prov = MeshProvenance(tri_pid=["F0", "F1"],
                                  requested_size={"F0": 1.0, "F1": 1.0},
                                  tri_field_size=[1.0, 1.0])
            return MeshResult(mesh=mesh, provenance=prov,
                              stats=MeshStats(n_triangles=2, backend=self.name),
                              request_fingerprint=request.fingerprint())

    be = Degenerate()
    rec = AdaptiveMesher(be, criteria=Criteria(), fidelity=False,
                         max_iterations=4).run("m.step")
    kinds = [it.action["kind"] for it in rec.iterations if it.action]
    assert kinds[0] == "enable_healing", kinds
    assert "increase_heal_tolerance" in kinds, kinds
    heals = [s.heal for s in be.strategies]
    assert heals[0] is False and all(heals[1:]), heals
    tols = [s.heal_tolerance_frac for s in be.strategies]
    assert tols == sorted(tols), "each reload must heal at least as hard"


def test_ineffective_strategy_action_is_flagged():
    """REGRESSION from sliver_block: enable_healing produced a byte-identical mesh.

    Same triangle count, same failure codes, same score means the change never
    reached the mesher. That is a plumbing problem, not a policy one, and the two
    are indistinguishable from the score alone.
    """
    from pipeline.loop import _indistinguishable
    a = Iteration(index=0, n_triangles=10386, score=0.0,
                  failures=[{"code": "validity.degenerate_shape"}])
    b = Iteration(index=1, n_triangles=10386, score=0.0,
                  failures=[{"code": "validity.degenerate_shape"}])
    assert _indistinguishable(b, [a])
    assert not _indistinguishable(a, []), "the first iteration has no predecessor"

    c = Iteration(index=1, n_triangles=9000, score=0.0,
                  failures=[{"code": "validity.degenerate_shape"}])
    assert not _indistinguishable(c, [a]), "a different count IS a difference"
    d = Iteration(index=1, n_triangles=10386, score=0.0,
                  failures=[{"code": "validity.inverted"}])
    assert not _indistinguishable(d, [a]), "different failures ARE a difference"


def test_reload_report_is_recorded_on_the_following_iteration():
    """So the record can show whether healing changed the face count."""
    from quality.criteria import Criteria

    class Degenerate(FakeBackend):
        def mesh(self, request):
            self.requests.append(request); self.calls += 1
            v = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.0005, 0],
                          [0, 1, 0], [1, 1, 0]], float)
            t = np.array([[0, 1, 2], [0, 3, 4]])
            mesh = SurfaceMesh(vertices=v, triangles=t, method="degen",
                               meta={"target_size": 1.0})
            prov = MeshProvenance(tri_pid=["F0", "F1"],
                                  requested_size={"F0": 1.0, "F1": 1.0},
                                  tri_field_size=[1.0, 1.0])
            return MeshResult(mesh=mesh, provenance=prov,
                              stats=MeshStats(n_triangles=2, backend=self.name),
                              request_fingerprint=request.fingerprint())

    be = Degenerate()
    rec = AdaptiveMesher(be, criteria=Criteria(), fidelity=False,
                         max_iterations=4).run("m.step")
    assert rec.iterations[0].load_report is None, "no reload preceded iteration 0"
    assert rec.iterations[1].load_report is not None
    assert rec.iterations[1].load_report["imprinted"] is True
    assert rec.iterations[1].ineffective, "identical meshes must be flagged"
    assert any("no measurable effect" in n for n in rec.notes)


# --------------------------------------------------------------------------
# Healing is a TOLERANCE, not a switch
# --------------------------------------------------------------------------

def test_heal_tolerance_default_is_scale_relative():
    """REGRESSION: gmsh's healShapes() defaults to an ABSOLUTE 1e-8.

    On sliver_block (76 mm scale, 0.005 mm rail) that is 500,000x smaller than the
    artifact it is asked to remove, so enabling healing produced a byte-identical
    mesh and the loop reported 'previous action had NO EFFECT'. Every other
    threshold in this project is a fraction of model scale for this reason.
    """
    st = MeshingStrategy()
    assert 0 < st.heal_tolerance_frac <= 1e-3, st.heal_tolerance_frac
    scale, rail = 76.81, 0.005
    assert st.heal_tolerance_frac * scale > rail, \
        "the default must be able to reach a real sub-tolerance artifact"


def test_heal_ladder_escalates_tolerance_before_escalating_to_a_human():
    """Toggling a boolean cannot express 'heal harder'."""
    v = _Verdict(False, [_Fail("validity.degenerate_shape", 138,
                               {"worst": 0.018})])
    strat = MeshingStrategy()
    spec = SizeFieldSpec.default(76.81)
    seen = []
    for _ in range(6):
        a = _propose(v, strategy=strat)[0]
        seen.append((strat.heal, strat.heal_tolerance_frac, a.kind))
        if a.kind is P.ActionKind.ESCALATE:
            break
        _, strat = P.apply_action(a, spec, strat)

    kinds = [k for _, _, k in seen]
    assert kinds[0] is P.ActionKind.ENABLE_HEALING
    assert P.ActionKind.INCREASE_HEAL_TOLERANCE in kinds, kinds
    assert kinds[-1] is P.ActionKind.ESCALATE, kinds
    tols = [t for _, t, _ in seen]
    assert tols == sorted(tols), "tolerance must increase monotonically"


def test_heal_ladder_stops_before_eating_real_features():
    """Above ~1% of model scale healing removes genuine fillets, not artifacts."""
    v = _Verdict(False, [_Fail("validity.degenerate_shape", 5)])
    at_ceiling = MeshingStrategy(heal=True,
                                 heal_tolerance_frac=P.MAX_HEAL_TOLERANCE_FRAC)
    a = _propose(v, strategy=at_ceiling)[0]
    assert a.kind is P.ActionKind.ESCALATE
    assert "feature contract" in a.rationale

    below = MeshingStrategy(heal=True,
                            heal_tolerance_frac=P.MAX_HEAL_TOLERANCE_FRAC / 10)
    b = _propose(v, strategy=below)[0]
    assert b.kind is P.ActionKind.INCREASE_HEAL_TOLERANCE
    assert b.params["heal_tolerance_frac"] <= P.MAX_HEAL_TOLERANCE_FRAC


def test_increase_heal_tolerance_also_turns_healing_on():
    """The action must be self-sufficient, not depend on ordering."""
    _, st = P.apply_action(
        P.Action(P.ActionKind.INCREASE_HEAL_TOLERANCE,
                 {"heal_tolerance_frac": 1e-3}),
        SizeFieldSpec.default(100.0), MeshingStrategy(heal=False))
    assert st.heal is True and st.heal_tolerance_frac == 1e-3


# --------------------------------------------------------------------------
# Uninformative outcomes -- found on the ABC corpus
# --------------------------------------------------------------------------

def test_empty_mesh_becomes_a_hard_failure_not_silence():
    """REGRESSION: 14 of 46 ABC runs ended 'no_action_available' with NO
    unsatisfied objectives and NO failures -- a combination that should be
    impossible, since valid + all objectives met means ready.

    Cause: the loop returned before assess() when the mesh was empty, leaving the
    iteration with nothing for the policy to act on. quality.criteria already had a
    mesh.empty hard failure; it was simply never reached, so the run reported a
    property of the POLICY instead of the problem.
    """
    be = FakeBackend([{"empty": True}])
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False,
                         max_iterations=2).run("m.step")
    it = rec.iterations[0]
    assert it.failures, "an empty mesh must record a failure"
    assert "mesh.empty" in {f["code"] for f in it.failures}
    assert not it.is_valid and not it.ready


def test_over_budget_is_reported_as_budget_exhausted():
    """Distinct from NO_ACTION: the policy had a move, the budget forbade it."""
    be = FakeBackend([{"n": 6}])
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False, budget=10,
                         max_iterations=3).run("m.step")
    assert rec.outcome is Outcome.BUDGET_EXHAUSTED, rec.outcome
    assert rec.iterations[0].over_budget
    assert be.calls == 0, "must refuse before meshing"


def test_stale_evidence_never_drives_a_proposal():
    """An iteration that does not reach assessment must leave no evidence behind.

    Otherwise the policy proposes actions against the PREVIOUS iteration's mesh,
    which no longer exists.
    """
    be = FakeBackend([{"n": 6}, {"empty": True}])
    m = AdaptiveMesher(be, criteria=STRICT, fidelity=False, max_iterations=2)
    m.run("m.step")
    # After an empty iteration the cached evidence must describe THAT iteration.
    cached = getattr(m, "_last", None)
    if cached is not None:
        verdict = cached[0]
        assert not verdict.is_valid


# --------------------------------------------------------------------------
# Mesher exceptions are recoverable, not dead ends
# --------------------------------------------------------------------------

class RaisingBackend(FakeBackend):
    """Raises a scripted gmsh-style exception until the strategy changes."""

    def __init__(self, message, *, recover_on=None, **kw):
        super().__init__([{"n": 6}], **kw)
        self.message = message
        self.recover_on = recover_on or {}
        self.attempts = []

    def mesh(self, request):
        st = request.strategy
        self.attempts.append((st.algorithm, st.heal))
        # An EMPTY recover_on means "never recover". all([]) is True, so the
        # obvious spelling silently made this backend never raise at all.
        ok = bool(self.recover_on) and all(
            getattr(st, k) == v for k, v in self.recover_on.items())
        if not ok:
            raise Exception(self.message)
        return super().mesh(request)


def test_mesher_exception_becomes_an_actionable_failure():
    """REGRESSION: 9 of 46 ABC parts died on the first mesh with no recovery.

    The loop caught the exception and returned before assessment, leaving no
    failures and no objectives, so the policy proposed nothing and the run ended
    'no action available' -- describing the policy, not the part.
    """
    be = RaisingBackend("Exception: Impossible to mesh periodic surface 16")
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False,
                         max_iterations=2).run("m.step")
    it = rec.iterations[0]
    assert it.failures, "a mesher exception must record a failure"
    assert it.failures[0]["code"] == "mesher.failed"
    assert rec.outcome is not Outcome.NO_ACTION, rec.outcome


def test_periodic_surface_failure_goes_straight_to_meshadapt():
    """Stage 0 already prescribes MeshAdapt for degenerate parametrizations.

    Walking the ladder through Delaunay first wastes an iteration: it fails the
    same way, for the same reason.
    """
    be = RaisingBackend("Exception: Impossible to mesh periodic surface 82",
                        recover_on={"algorithm": "meshadapt"})
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False,
                         max_iterations=4).run("m.step")
    assert rec.outcome is Outcome.ACCEPTED, rec.summary()
    assert [a for a, _ in be.attempts][1] == "meshadapt", be.attempts
    assert "delaunay" not in [a for a, _ in be.attempts]


def test_unclosed_loop_failure_tries_healing_first():
    """A 1D loop that will not close is a boundary/tolerance defect, not a 2D
    algorithm choice."""
    be = RaisingBackend("Exception: The 1D mesh seems not to be forming a "
                        "closed loop", recover_on={"heal": True})
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False,
                         max_iterations=4).run("m.step")
    assert rec.outcome is Outcome.ACCEPTED, rec.summary()
    assert be.attempts[0][1] is False and be.attempts[1][1] is True
    algos = {a for a, _ in be.attempts}
    assert len(algos) == 1, \
        f"switching algorithm would not help a boundary defect: {be.attempts}"


def test_both_recovery_ladders_terminate_at_escalate():
    from quality.criteria import Failure

    class V:
        def __init__(self, f):
            self.is_valid, self.failures, self.objectives = False, f, []

        def unsatisfied(self):
            return []

    spec = SizeFieldSpec.default(86.6)
    for exc in ("Exception: Impossible to mesh periodic surface 16",
                "Exception: The 1D mesh seems not to be forming a closed loop"):
        strat, hist, kinds = MeshingStrategy(), [], []
        for _ in range(6):
            acts = P.propose(
                V([Failure("mesher.failed", "raised", 1, {"exception": exc})]),
                [], spec, strat, model_scale=86.6, face_areas={},
                current_sizes={}, budget=250_000, history=hist)
            a = acts[0]
            kinds.append(a.kind)
            if a.kind is P.ActionKind.ESCALATE:
                break
            hist.append(a)
            _, strat = P.apply_action(a, spec, strat)
        assert kinds[-1] is P.ActionKind.ESCALATE, (exc, kinds)
        assert len(kinds) <= 5, kinds


def test_unrecognized_mesher_error_still_gets_an_attempt():
    from quality.criteria import Failure

    class V:
        def __init__(self, f):
            self.is_valid, self.failures, self.objectives = False, f, []

        def unsatisfied(self):
            return []

    acts = P.propose(
        V([Failure("mesher.failed", "raised", 1, {"exception": "something new"})]),
        [], SizeFieldSpec.default(100.0), MeshingStrategy(), model_scale=100.0,
        face_areas={}, current_sizes={}, budget=250_000, history=[])
    assert acts[0].kind is P.ActionKind.SWITCH_ALGORITHM


def test_thickness_failure_after_a_strategy_change_is_not_fatal():
    """REGRESSION: the recovery defeated itself.

    measure_thickness() runs its own generate(2), so on a part the mesher cannot
    handle it raises exactly the error the policy just switched algorithm to avoid.
    Inside the reload try-block that aborted the run as backend_error -- 6 of 46
    ABC parts ended that way, never testing the algorithm that had been chosen.

    Thickness feeds ONE sizing rule. It is an input, not a precondition.
    """
    class ThicknessRaises(RaisingBackend):
        def measure_thickness(self):
            raise Exception("Impossible to mesh periodic surface 16")

    be = ThicknessRaises("Exception: Impossible to mesh periodic surface 16",
                         recover_on={"algorithm": "meshadapt"})
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False,
                         max_iterations=4).run("m.step")
    assert rec.outcome is not Outcome.BACKEND_ERROR, rec.summary()
    assert rec.outcome is Outcome.ACCEPTED, rec.summary()
    assert any("thickness unavailable" in n for n in rec.notes)


def test_reload_failure_is_still_fatal():
    """The guard must not swallow a genuine reload failure."""
    class LoadRaises(RaisingBackend):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._n = 0

        def load(self, path, strategy):
            self._n += 1
            if self._n > 1:
                raise RuntimeError("cannot re-import")
            return super().load(path, strategy)

    be = LoadRaises("Exception: Impossible to mesh periodic surface 16")
    rec = AdaptiveMesher(be, criteria=LOOSE, fidelity=False,
                         max_iterations=4).run("m.step")
    assert rec.outcome is Outcome.BACKEND_ERROR
    assert any("reload after" in n for n in rec.notes)


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
