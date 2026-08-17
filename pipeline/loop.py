"""
The adaptive surface-meshing loop. PURE of gmsh -- the backend is injected.

    mesh -> assess -> attribute -> propose -> select -> apply -> mesh ...

Terminates on the deterministic criteria in quality/criteria.py, never on a model's
opinion. Scope is surface meshing only; volume meshing, boundary conditions and
solver integration are out.

Why the backend is injected
---------------------------
The loop takes anything satisfying backends.base.MeshingBackend. That keeps the
control logic testable against a fake backend with scripted behaviour -- degrading
meshes, stubborn failures, oscillation traps -- none of which can be provoked
reliably from real CAD. A loop that has only ever been run on cases that converge is
untested where it matters.

Termination
-----------
  ACCEPTED           verdict.ready: valid AND every objective satisfied
  ESCALATED          the policy asked for a human decision (e.g. defeaturing)
  NO_ACTION          no untried candidate remains
  OSCILLATION        a parameter set already meshed came back around
  PLATEAU            the score stopped moving
  MAX_ITERATIONS     budget of attempts exhausted
  BUDGET_EXHAUSTED   every remaining action exceeds the element budget

Whatever the outcome, the loop returns the BEST VALID iteration, which is often not
the last one -- the final iteration is usually the most aggressive edit.
"""

from __future__ import annotations

import time

from backends.base import MeshRequest, attribute_by_face
from pipeline import contracts as C
from pipeline import policy as P
from pipeline.session import Iteration, Outcome, RunRecord
from pipeline.sizefield import SizeFieldSpec, estimate_element_count
from quality.criteria import Criteria, Failure, Objective, Verdict
from quality.criteria import evaluate as assess


class AdaptiveMesher:
    """Runs the loop against an injected backend."""

    def __init__(self, backend, *, criteria: Criteria | None = None,
                 selector: P.Selector | None = None,
                 budget: int = 250_000, max_iterations: int = 8,
                 fidelity: bool = True, fidelity_samples: int = 60,
                 approved_removals=(), auto_approve_area_frac: float = 1e-6,
                 auto_approve_circularity: float = 1e-2):
        self.backend = backend
        self.criteria = criteria or Criteria()
        self.selector = selector or P.DeterministicSelector()
        self.budget = budget
        self.max_iterations = max_iterations
        self.fidelity = fidelity
        self.fidelity_samples = fidelity_samples
        self.approved_removals = set(approved_removals)
        self.auto_approve_area_frac = auto_approve_area_frac
        self.auto_approve_circularity = auto_approve_circularity
        self.best_result = None
        """MeshResult of the best VALID iteration.

        RunRecord keeps counts and scores, not geometry, so without retaining this
        the pipeline validates a mesh and then discards it -- which is how the
        stated deliverable ("a validated surface mesh") went unproduced. Retained
        rather than re-meshed at the end because the best iteration is often not the
        last, and re-running its parameters would repeat the whole cost."""

    # -----------------------------------------------------------------------
    def run(self, path: str, *, spec: SizeFieldSpec | None = None,
            strategy=None) -> RunRecord:
        self.best_result = None
        self._last_result = None
        from backends.base import MeshingStrategy

        strategy = strategy or MeshingStrategy()
        record = RunRecord(model=path, backend=getattr(self.backend, "name", "?"),
                           selector=self.selector.name,
                           criteria=self.criteria.to_dict(), budget=self.budget)

        try:
            load_report = self.backend.load(path, strategy)
            record.notes.append(f"load: {load_report}")
        except Exception as e:
            record.outcome = Outcome.BACKEND_ERROR
            record.notes.append(f"load failed: {type(e).__name__}: {e}")
            return record

        scale = self.backend.model_scale()
        if spec is None:
            spec = SizeFieldSpec.default(scale)

        # Wall thickness feeds the thickness sizing rule. Without it the rule is
        # inert and thin walls are meshed at the base size.
        try:
            th = self.backend.measure_thickness()
            record.notes.append(f"thickness measured on {len(th)} face(s)")
        except Exception as e:
            record.notes.append(f"thickness unavailable: {type(e).__name__}: {e}")

        self._rebuild_contracts(scale)
        if self._topology is None:
            record.notes.append(
                "contracts unavailable: the backend does not describe its "
                "geometry, so only quality checks apply")
        else:
            record.notes.append(
                f"contract: {self._topology.body_kind.value}, "
                f"{len(self._feature.required_pids)} required face(s), "
                f"{len(self._feature.removals)} removal(s)")

        applied: list[P.Action] = []
        seen_requests: set[str] = set()
        prev_spec: SizeFieldSpec | None = None
        pending_load: dict | None = None

        for i in range(self.max_iterations):
            request = MeshRequest(spec, strategy)
            fp = request.fingerprint()
            if fp in seen_requests:
                record.outcome = Outcome.OSCILLATION
                record.notes.append(
                    f"iteration {i} reproduced an already-meshed parameter set")
                break
            seen_requests.add(fp)

            it = self._iterate(i, request, spec, strategy, prev_spec, record)
            if it is None:
                record.outcome = Outcome.BACKEND_ERROR
                break
            it.load_report = pending_load
            pending_load = None
            it.ineffective = _indistinguishable(it, record.iterations)
            prev_best = record.best_index
            if it.ineffective:
                record.notes.append(
                    f"iteration {i} is indistinguishable from the previous one; "
                    "the strategy action had no measurable effect")
            record.add(it)
            if record.best_index == it.index and record.best_index != prev_best:
                self.best_result = self._last_result

            if it.ready:
                record.outcome = Outcome.ACCEPTED
                break
            if getattr(it, "over_budget", False):
                record.outcome = Outcome.BUDGET_EXHAUSTED
                break
            if record.plateaued():
                record.outcome = Outcome.PLATEAU
                record.notes.append("score stopped improving; no lever left")
                break

            action = self.selector.select(self._candidates(it, spec, strategy,
                                                           scale, applied),
                                          applied)
            if action is None:
                record.outcome = Outcome.NO_ACTION
                break
            it.action = action.to_dict()
            if action.is_terminal:
                record.outcome = Outcome.ESCALATED
                record.notes.append(action.rationale)
                break

            applied.append(action)
            prev_spec = spec
            spec, strategy = P.apply_action(action, spec, strategy)

            # A strategy change can alter the geometry itself (healing, imprinting,
            # tolerance), so the model must be reloaded and entity ids re-resolved
            # before the next mesh.
            if action.is_strategy and action.kind is not \
                    P.ActionKind.INCREASE_OPTIMIZATION:
                try:
                    pending_load = self.backend.load(path, strategy)
                    # Geometry changed, so the contracts must be re-derived. A
                    # healed model has fewer faces, and carrying the old required
                    # set forward would report every healed-away face as lost.
                    self._rebuild_contracts(self.backend.model_scale())
                except Exception as e:
                    record.notes.append(
                        f"reload after {action.kind.value} failed: {e}")
                    record.outcome = Outcome.BACKEND_ERROR
                    break
                # Thickness is an INPUT to one sizing rule, not a precondition for
                # meshing. It runs its own generate(2), so on a part the mesher
                # cannot handle it raises the very error the recovery is addressing
                # -- and inside the try above that aborted the run as
                # backend_error, defeating the retry. 6 of 46 ABC parts ended that
                # way, never testing the algorithm the policy had chosen.
                try:
                    self.backend.measure_thickness()
                except Exception as e:
                    record.notes.append(
                        f"thickness unavailable after {action.kind.value}: "
                        f"{type(e).__name__}: {e}")
        else:
            record.outcome = Outcome.MAX_ITERATIONS

        return record

    # -----------------------------------------------------------------------
    def _iterate(self, index, request, spec, strategy, prev_spec,
                 record) -> Iteration | None:
        t0 = time.time()
        # Clear last-iteration evidence up front. If this iteration does not reach
        # assessment, the policy must see nothing rather than the PREVIOUS
        # iteration's verdict, which would propose actions against a mesh that no
        # longer exists.
        self._last = None
        info = self.backend.face_info()
        sizes = spec.resolve(info, smooth=False)
        areas = {pid: float(v.get("area") or 0.0) for pid, v in info.items()}
        est = estimate_element_count(sizes, areas)

        it = Iteration(index=index, request_fingerprint=request.fingerprint(),
                       size_field=spec.to_dict(),
                       spec_diff=(prev_spec.diff(spec) if prev_spec else []),
                       strategy=strategy.to_dict(), estimated_elements=est)

        if est > self.budget:
            # Refuse BEFORE meshing. Discovering a two-million-element mesh after
            # the fact costs the whole iteration and teaches nothing new.
            it.note = (f"size field implies {est:,} elements, over the "
                       f"{self.budget:,} budget; not meshed")
            it.over_budget = True
            it.seconds = time.time() - t0
            record.notes.append(it.note)
            return it

        self._last_result = None
        try:
            result = self.backend.mesh(request)
            self._last_result = result
        except Exception as e:
            # A mesher exception is a HARD FAILURE the policy can act on, not a
            # dead end. Returning silently here left the iteration with no failures
            # and no objectives, so the run ended "no action available" without a
            # single recovery attempt -- 9 of 46 ABC parts, all on two recoverable
            # gmsh errors.
            msg = f"{type(e).__name__}: {e}"
            it.note = f"backend.mesh failed: {msg}"
            fail = Failure(code="mesher.failed",
                           message=f"the mesher raised: {msg}", count=1,
                           detail={"exception": msg})
            it.failures = [{"code": fail.code, "count": 1,
                            "message": fail.message}]
            it.seconds = time.time() - t0
            self._last = (Verdict(is_valid=False, failures=[fail],
                                  method="gmsh"), [], sizes, areas)
            return it

        mesh, prov = result.mesh, result.provenance
        it.n_triangles, it.n_vertices = mesh.n_triangles, mesh.n_vertices
        if mesh.n_triangles == 0:
            # Assess it anyway. Returning early leaves the iteration with no
            # failures AND no objectives, so the policy sees nothing to act on and
            # the run ends as "no action available" -- which describes the policy,
            # not the problem. quality.criteria already has a mesh.empty hard
            # failure; it just was never reached. This accounted for most of a
            # 14-of-46 block of uninformative outcomes on the ABC corpus.
            it.note = "backend produced no triangles"
            empty_verdict = assess(mesh, criteria=self.criteria)
            it.failures = [{"code": f.code, "count": f.count,
                            "message": f.message} for f in empty_verdict.failures]
            it.seconds = time.time() - t0
            self._last = (empty_verdict, [], sizes, areas)
            return it

        surface_for = (self.backend.surface_for
                       if self.fidelity and hasattr(self.backend, "surface_for")
                       else None)
        verdict = assess(mesh, criteria=self.criteria,
                         target_size=prov.size_field_callable(),
                         surface=None,
                         interface_triangles=prov.interface_triangles())
        scores = attribute_by_face(mesh, prov, surface_for=surface_for,
                                   max_fidelity_samples=self.fidelity_samples)
        # assess() takes ONE surface, but a CAD model has one per face, so the
        # fidelity objectives cannot be produced there. Without this the policy's
        # chordal and normal branches are unreachable -- it would never act on
        # geometric fidelity at all. Aggregate them from the per-face scores, which
        # attribution has already computed.
        _add_fidelity_objectives(verdict, scores, self.criteria)

        it.is_valid = verdict.is_valid
        it.ready = verdict.ready
        it.score = verdict.score
        it.failures = [{"code": f.code, "count": f.count, "message": f.message}
                       for f in verdict.failures]
        it.objectives = [{"name": o.name, "value": o.value, "target": o.target,
                          "satisfied": o.satisfied,
                          "attainment": o.normalized} for o in verdict.objectives]
        it.worst_faces = [s.to_dict() for s in scores[:5]]
        it.seconds = time.time() - t0

        # Contracts compare the mesh against the CAD, which no quality metric can
        # do. Violations are HARD failures: a mesh missing a required face is a mesh
        # of the wrong part, not a lower-quality one.
        if self._topology is None:
            self._last = (verdict, scores, sizes, areas)
            return it
        topo = _topology_of(mesh, prov)
        creport = C.verify(
            self._topology, self._feature,
            meshed_pids=prov.pids(),
            boundary_edge_length=topo["boundary_length"],
            n_nonmanifold_skin=topo["n_nonmanifold_edges"],
            n_inconsistent_normals=topo["n_inconsistent_normals"])
        it.contract = creport.to_dict()
        if not creport.satisfied:
            verdict.failures.extend(C.as_failures(creport))
            verdict.is_valid = False
            it.is_valid = False
            it.ready = False
            it.score = verdict.score
            it.failures = [{"code": f.code, "count": f.count,
                            "message": f.message} for f in verdict.failures]

        self._last = (verdict, scores, sizes, areas)
        return it

    def _rebuild_contracts(self, scale: float) -> None:
        """Derive both contracts from the CAD. Optional: a backend that cannot
        describe its geometry gets quality checks only, not contract checks."""
        if not hasattr(self.backend, "body_kind_inputs"):
            self._topology = self._feature = None
            return
        inputs = self.backend.body_kind_inputs()
        self._topology = C.TopologyContract.derive(
            n_volumes=inputs["n_volumes"], n_faces=inputs["n_faces"],
            free_edge_length=inputs["free_edge_length"],
            interface_pids=inputs["interface_pids"])
        removals = (self.backend.removed_faces(scale)
                    if hasattr(self.backend, "removed_faces") else ())
        removals = tuple(
            C.Removal(r.pid, r.area, r.area_frac, r.perimeter, r.reason,
                      approved=r.pid in self.approved_removals,
                      approved_by="operator" if r.pid in self.approved_removals
                      else "")
            for r in removals)
        self._feature = C.FeatureContract.derive(
            required_pids=[r.pid for r in self.backend.registry.live(dim=2)],
            removals=removals,
            auto_approve_area_frac=self.auto_approve_area_frac,
            auto_approve_circularity=self.auto_approve_circularity)

    def _candidates(self, it: Iteration, spec, strategy, scale, applied):
        """Ask the policy for ranked actions from the last iteration's evidence."""
        cached = getattr(self, "_last", None)
        if cached is None:
            return []
        verdict, scores, sizes, areas = cached
        return P.propose(verdict, scores, spec, strategy, model_scale=scale,
                         face_areas=areas, current_sizes=sizes,
                         budget=self.budget, history=applied)


def _topology_of(mesh, prov) -> dict:
    from quality.core import topology_checks
    return topology_checks(mesh, prov.interface_triangles())


def _indistinguishable(it, previous) -> bool:
    """True when this iteration's mesh is not measurably different from the last.

    Same element count and the same failure codes means whatever was changed did not
    reach the mesher. Distinguishing that from "the change was applied and did not
    help" is the difference between a policy problem and a plumbing problem, and it
    is not visible from the score alone.
    """
    if not previous:
        return False
    prev = previous[-1]
    return (it.n_triangles == prev.n_triangles
            and it.n_triangles > 0
            and {f["code"] for f in it.failures} == {f["code"]
                                                     for f in prev.failures}
            and abs(it.score - prev.score) < 1e-9)


def _add_fidelity_objectives(verdict, scores, criteria) -> None:
    """Fold per-face fidelity into the verdict as whole-mesh objectives.

    Uses the WORST face rather than an average. Fidelity is a tolerance: one face
    cutting a 5 mm chord across a fillet is a defect whether or not the other forty
    faces are perfect, and averaging would let good faces mask it.
    """
    import math

    def worst(attr):
        vals = [getattr(s, attr) for s in scores
                if not math.isnan(getattr(s, attr))]
        return max(vals) if vals else None

    have = {o.name for o in verdict.objectives}
    w = criteria.weights

    ch = worst("chordal_relative")
    if ch is not None and "chordal" not in have:
        verdict.objectives.append(Objective(
            name="chordal", value=float(ch),
            target=float(criteria.max_chordal_relative),
            satisfied=ch <= criteria.max_chordal_relative, direction="min",
            weight=float(w.get("chordal", 1.0)),
            detail={"source": "worst face"}))

    nd = worst("normal_p99_deg")
    if nd is not None and "normal_deviation" not in have:
        verdict.objectives.append(Objective(
            name="normal_deviation", value=float(nd),
            target=float(criteria.max_normal_deg),
            satisfied=nd <= criteria.max_normal_deg, direction="min",
            weight=float(w.get("normal_deviation", 1.0)),
            detail={"source": "worst face"}))
