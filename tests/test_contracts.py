"""
Topology and feature contract tests. PURE -- no gmsh.

Contracts answer the one question quality metrics cannot: does the mesh agree with
the CAD it came from? Every check here compares against the INPUT, not against the
mesh itself.

The removal-approval tests carry the most weight. On sliver_block, healing deleted
two 0.005 mm rail faces, element count fell 10,386 -> 1,858, and the run reported
"accepted" with the deletion recorded only inside a load report. That was the right
outcome for that part -- and indistinguishable from healing quietly removing a real
fillet.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pipeline import contracts as C
from pipeline.contracts import BodyKind, FeatureContract, Removal, TopologyContract


def _feat(required=(), removals=(), auto=1e-6, circ=1e-2):
    return FeatureContract.derive(required, removals, auto, circ)


def _verify(topo, feat, meshed, boundary=0.0, nonmanifold=0, normals=0):
    return C.verify(topo, feat, meshed_pids=meshed,
                    boundary_edge_length=boundary,
                    n_nonmanifold_skin=nonmanifold,
                    n_inconsistent_normals=normals)


# --------------------------------------------------------------------------
# Derivation -- the expectation comes from the CAD
# --------------------------------------------------------------------------

def test_body_kind_is_derived_not_configured():
    """'require watertight' as a flag is wrong in both directions."""
    solid = TopologyContract.derive(n_volumes=1, n_faces=11, free_edge_length=0.0)
    assert solid.body_kind is BodyKind.SOLID and solid.require_closed

    sheet = TopologyContract.derive(n_volumes=0, n_faces=1, free_edge_length=4.0)
    assert sheet.body_kind is BodyKind.SHEET and not sheet.require_closed
    assert sheet.expected_free_edge_length == 4.0

    asm = TopologyContract.derive(n_volumes=2, n_faces=11, free_edge_length=0.0,
                                  interface_pids=("I0",))
    assert asm.body_kind is BodyKind.ASSEMBLY and asm.require_closed

    nothing = TopologyContract.derive(n_volumes=0, n_faces=0, free_edge_length=0.0)
    assert nothing.body_kind is BodyKind.UNKNOWN


def test_a_closed_body_expects_no_free_edge_length():
    solid = TopologyContract.derive(n_volumes=1, n_faces=6, free_edge_length=99.0)
    assert solid.expected_free_edge_length == 0.0, \
        "a solid's free-edge length is meaningless and must not be carried"


# --------------------------------------------------------------------------
# Topology verification
# --------------------------------------------------------------------------

def test_closed_solid_must_produce_a_closed_mesh():
    topo = TopologyContract.derive(1, 6, 0.0)
    ok = _verify(topo, _feat(("A", "B")), {"A", "B"}, boundary=0.0)
    assert ok.satisfied, ok.summary()

    bad = _verify(topo, _feat(("A", "B")), {"A", "B"}, boundary=12.5)
    assert not bad.satisfied
    assert "contract.unexpected_boundary" in {v.code for v in bad.violations}


def test_sheet_body_is_validated_against_its_own_boundary_not_watertightness():
    """A shell legitimately has free edges; requiring closure is the wrong test."""
    topo = TopologyContract.derive(0, 1, free_edge_length=4.0)
    ok = _verify(topo, _feat(("A",)), {"A"}, boundary=3.95)   # chords cut corners
    assert ok.satisfied, ok.summary()

    closed = _verify(topo, _feat(("A",)), {"A"}, boundary=0.0)
    assert not closed.satisfied
    assert "contract.boundary_mismatch" in {v.code for v in closed.violations}


def test_sheet_with_an_unexpected_hole_is_caught():
    """A hole in the middle of a sheet adds boundary the CAD does not have.

    This is the check that 'do not require watertightness' would miss entirely.
    """
    topo = TopologyContract.derive(0, 1, free_edge_length=4.0)
    holed = _verify(topo, _feat(("A",)), {"A"}, boundary=4.0 + 1.6)
    assert not holed.satisfied
    v = next(x for x in holed.violations
             if x.code == "contract.boundary_mismatch")
    assert v.detail["relative_error"] > 0.05


def test_boundary_tolerance_absorbs_discretization_but_not_a_defect():
    topo = TopologyContract.derive(0, 1, free_edge_length=100.0)
    assert _verify(topo, _feat(("A",)), {"A"}, boundary=97.0).satisfied
    assert not _verify(topo, _feat(("A",)), {"A"}, boundary=80.0).satisfied


def test_nonmanifold_skin_and_orientation_are_contract_violations():
    topo = TopologyContract.derive(1, 6, 0.0)
    r = _verify(topo, _feat(("A",)), {"A"}, nonmanifold=3, normals=7)
    codes = {v.code for v in r.violations}
    assert "contract.nonmanifold_skin" in codes
    assert "contract.inconsistent_orientation" in codes


# --------------------------------------------------------------------------
# Feature verification
# --------------------------------------------------------------------------

def test_a_required_face_producing_no_elements_is_lost_geometry():
    topo = TopologyContract.derive(1, 3, 0.0)
    r = _verify(topo, _feat(("A", "B", "C")), {"A", "B"})
    v = next(x for x in r.violations if x.code == "contract.missing_face")
    assert v.pids == ("C",)
    assert "lost geometry" in v.message


def test_assembly_interface_must_be_meshed():
    """An unmeshed interface means the bodies share no nodes and carry no load."""
    topo = TopologyContract.derive(2, 11, 0.0, interface_pids=("IFACE",))
    r = _verify(topo, _feat(("A", "IFACE")), {"A"})
    codes = {v.code for v in r.violations}
    assert "contract.interface_not_conformal" in codes


# --------------------------------------------------------------------------
# Removal approval -- the sliver_block lesson
# --------------------------------------------------------------------------

def test_sliver_removals_auto_approve_on_SHAPE_not_area():
    """REGRESSION: area is the wrong test, and this cost a real escalation.

    sliver_block's rails are 50 x 0.005: aspect 10000:1, but area 0.25 mm^2, which
    is 4.2e-5 of scale^2 and nowhere near "tiny". Area-based auto-approval refused
    them and demanded a human decision about an obvious artifact -- the same mistake
    stage0's face.sliver detector made before area was replaced with circularity.
    """
    scale = 76.81
    rail = Removal("R1", area=0.25, area_frac=0.25 / scale**2,
                   perimeter=2 * (50 + 0.005))
    assert rail.area_frac > 1e-6, "by AREA this is not tiny"
    assert rail.circularity < 1e-3, rail.circularity
    assert rail.aspect_ratio > 1000

    feat = _feat(("A",), (rail,))
    assert feat.removals[0].approved
    assert "sliver" in feat.removals[0].approved_by


def test_genuinely_tiny_removals_still_auto_approve_on_area():
    """The secondary test: a small round patch is an artifact too."""
    scale = 76.81
    tiny = Removal("R2", area=1e-6, area_frac=1e-6 / scale**2, perimeter=4e-3)
    feat = _feat(("A",), (tiny,))
    assert feat.removals[0].approved
    assert "area" in feat.removals[0].approved_by


def test_a_real_thin_feature_still_needs_a_human():
    """The check must not become a rubber stamp for anything elongated.

    A 30 x 1.5 fillet strip is 20:1 -- thin, but a real feature. Removing it
    changes the part.
    """
    scale = 76.81
    fillet = Removal("F9", area=45.0, area_frac=45.0 / scale**2,
                     perimeter=2 * (30 + 1.5))
    assert 10 < fillet.aspect_ratio < 60, fillet.aspect_ratio
    assert not _feat(("A",), (fillet,)).removals[0].approved


def test_circularity_threshold_maps_to_a_stated_aspect_ratio():
    from pipeline.contracts import aspect_from_circularity
    r = aspect_from_circularity(1e-2)
    assert 200 < r < 400, f"1e-2 should be roughly 300:1, got {r:.0f}"
    # Tighter than stage0's 5e-2 detection cut on purpose: auto-approving a REMOVAL
    # is a stronger claim than flagging a face for attention.
    assert aspect_from_circularity(5e-2) < r


def test_removal_without_perimeter_falls_back_to_area():
    """Older records carry no perimeter; they must not silently auto-approve."""
    scale = 76.81
    r = Removal("R", area=0.25, area_frac=0.25 / scale**2, perimeter=0.0)
    assert not _feat(("A",), (r,)).removals[0].approved


def test_a_substantial_removal_is_a_violation_until_approved():
    """THE POINT. Healing removing a real fillet must not look like success."""
    scale = 76.81
    fillet = Removal("F7", area=45.0, area_frac=45.0 / scale**2,
                     perimeter=2 * (7 + 6.5))          # a chunky 45 mm^2 face
    feat = _feat(("A",), (fillet,))
    assert feat.unapproved, "a chunky 45 mm^2 face is not an artifact"

    topo = TopologyContract.derive(1, 6, 0.0)
    r = _verify(topo, feat, {"A"})
    v = next(x for x in r.violations if x.code == "contract.unapproved_removal")
    assert "different geometry than the input" in v.message
    assert v.pids == ("F7",)


def test_explicit_approval_clears_the_violation():
    scale = 76.81
    fillet = Removal("F7", area=45.0, area_frac=45.0 / scale**2,
                     perimeter=2 * (7 + 6.5))
    feat = _feat(("A",), (fillet,)).approve("F7", by="kington")
    assert not feat.unapproved
    assert feat.removals[0].approved_by == "kington"
    assert _verify(TopologyContract.derive(1, 6, 0.0), feat, {"A"}).satisfied


def test_approval_returns_a_new_contract():
    """Immutability keeps each iteration's recorded contract meaningful."""
    fillet = Removal("F7", area=45.0, area_frac=0.01, perimeter=2 * (7 + 6.5))
    feat = _feat(("A",), (fillet,))
    feat2 = feat.approve("F7")
    assert feat.unapproved and not feat2.unapproved
    assert feat is not feat2


def test_approving_an_unknown_pid_changes_nothing():
    fillet = Removal("F7", area=45.0, area_frac=0.01, perimeter=2 * (7 + 6.5))
    feat = _feat(("A",), (fillet,)).approve("NOT_THERE")
    assert len(feat.unapproved) == 1


# --------------------------------------------------------------------------
# Failures, serialization
# --------------------------------------------------------------------------

def test_violations_become_hard_failures_never_objectives():
    """A mesh missing a required face is the wrong mesh, not a worse one."""
    topo = TopologyContract.derive(1, 3, 0.0)
    r = _verify(topo, _feat(("A", "B")), {"A"})
    failures = C.as_failures(r)
    assert failures and all(f.code.startswith("contract.") for f in failures)
    assert all(hasattr(f, "count") and f.count >= 1 for f in failures)


def test_report_serializes_with_the_contracts_it_checked():
    topo = TopologyContract.derive(2, 11, 0.0, interface_pids=("I",))
    feat = _feat(("A", "I"), (Removal("R", 0.25, 4e-5, perimeter=100.0),))
    r = _verify(topo, feat, {"A"})
    import json
    d = json.loads(json.dumps(r.to_dict()))
    assert d["topology"]["body_kind"] == "assembly"
    assert d["feature"]["removals"][0]["pid"] == "R"
    assert not d["satisfied"] and d["violations"]


def test_satisfied_report_reads_cleanly():
    topo = TopologyContract.derive(1, 2, 0.0)
    r = _verify(topo, _feat(("A", "B")), {"A", "B"})
    assert r.satisfied
    assert "satisfied" in r.summary() and "solid" in r.summary()


# --------------------------------------------------------------------------
# Loop integration
# --------------------------------------------------------------------------

class ContractBackend:
    """Backend that describes its geometry, so the loop runs contract checks."""

    name = "contract_fake"

    def __init__(self, *, n_volumes=0, free_edge_length=4.0, removals=(),
                 required=("F0", "F1")):
        self._nv = n_volumes
        self._fel = free_edge_length
        self._removals = tuple(removals)
        self._required = tuple(required)
        self.loads = 0

    def load(self, path, strategy):
        self.loads += 1
        return {"imported_faces": len(self._required)}

    def model_scale(self):
        return 100.0

    def measure_thickness(self):
        return {}

    def face_info(self):
        return {p: {"curvature_max": 0.0, "thickness": None, "area": 100.0}
                for p in self._required}

    def face_adjacency(self):
        return {p: set(self._required) - {p} for p in self._required}

    def surface_for(self, pid):
        return None

    def body_kind_inputs(self):
        return {"n_volumes": self._nv, "n_faces": len(self._required),
                "free_edge_length": self._fel, "interface_pids": ()}

    def removed_faces(self, model_scale=None):
        return self._removals

    class _Reg:
        def __init__(self, pids):
            self.pids = pids

        def live(self, dim=2):
            class R:
                def __init__(self, pid):
                    self.pid = pid
            return [R(p) for p in self.pids]

    @property
    def registry(self):
        return ContractBackend._Reg(self._required)

    def mesh(self, request):
        from backends.base import MeshProvenance, MeshResult, MeshStats
        from quality.core import SurfaceMesh
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        t = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = SurfaceMesh(vertices=v, triangles=t, method="contract_fake",
                           meta={"target_size": 1.0})
        prov = MeshProvenance(tri_pid=["F0", "F1"],
                              requested_size={p: 1.0 for p in self._required},
                              tri_field_size=[1.1547, 1.1547])
        return MeshResult(mesh=mesh, provenance=prov,
                          stats=MeshStats(n_triangles=2, backend=self.name),
                          request_fingerprint=request.fingerprint())


def _run(be, **kw):
    from pipeline.loop import AdaptiveMesher
    from quality.criteria import Criteria
    crit = Criteria(min_shape=0.3, min_angle_deg=20.0, max_angle_deg=130.0,
                    min_size_in_band=0.0, max_gradation=3.0)
    return AdaptiveMesher(be, criteria=crit, fidelity=False,
                          max_iterations=3, **kw).run("m.step")


def test_loop_accepts_a_sheet_whose_boundary_matches():
    rec = _run(ContractBackend(n_volumes=0, free_edge_length=4.0))
    assert rec.outcome.succeeded, rec.summary()
    assert rec.iterations[0].contract["satisfied"]
    assert rec.iterations[0].contract["topology"]["body_kind"] == "sheet"


def test_loop_rejects_an_open_mesh_for_a_solid():
    rec = _run(ContractBackend(n_volumes=1, free_edge_length=0.0))
    assert not rec.outcome.succeeded
    it = rec.iterations[0]
    assert not it.is_valid, "a contract violation must invalidate the mesh"
    assert "contract.unexpected_boundary" in {f["code"] for f in it.failures}


def test_loop_blocks_on_an_unapproved_removal():
    """The sliver_block gap: healing succeeded and nothing asked permission."""
    big = Removal("GONE", area=500.0, area_frac=500.0 / 100.0**2,
                  perimeter=2 * (25 + 20))
    rec = _run(ContractBackend(n_volumes=0, free_edge_length=4.0,
                               removals=(big,)))
    assert not rec.outcome.succeeded
    codes = {f["code"] for f in rec.iterations[0].failures}
    assert "contract.unapproved_removal" in codes


def test_loop_accepts_once_the_removal_is_approved():
    big = Removal("GONE", area=500.0, area_frac=500.0 / 100.0**2,
                  perimeter=2 * (25 + 20))
    rec = _run(ContractBackend(n_volumes=0, free_edge_length=4.0,
                               removals=(big,)),
               approved_removals={"GONE"})
    assert rec.outcome.succeeded, rec.summary()
    assert rec.iterations[0].contract["satisfied"]


def test_loop_skips_contracts_for_a_backend_that_cannot_describe_geometry():
    """Optional, not silently wrong: quality checks still apply."""
    import tests.test_loop as TL
    rec = _run(TL.FakeBackend([{"n": 6}]))
    assert rec.iterations[0].contract is None
    assert any("contracts unavailable" in n for n in rec.notes)


def test_violation_message_names_the_faces():
    """An escalation that says a decision is needed but not WHAT to decide on
    forces a JSON dig for the one fact required to act."""
    scale = 76.81
    fillet = Removal("F7", area=45.0, area_frac=45.0 / scale**2,
                     perimeter=2 * (7 + 6.5))
    r = _verify(TopologyContract.derive(1, 6, 0.0), _feat(("A",), (fillet,)),
                {"A"})
    v = next(x for x in r.violations if x.code == "contract.unapproved_removal")
    assert "F7" in v.message, v.message
    assert v.pids == ("F7",)


def test_iteration_summary_prints_the_offending_pids():
    from pipeline.session import Iteration
    it = Iteration(index=0, contract={
        "satisfied": False,
        "violations": [{"code": "contract.unapproved_removal",
                        "pids": ["abc123", "def456"], "message": "", "detail": {}}]})
    text = it.summary()
    assert "abc123" in text and "def456" in text, text
    assert "contract.unapproved_removal" in text


def test_cli_arguments_reach_the_loop():
    """REGRESSION: a patch updated the CALL SITE but not the SIGNATURE.

    str.replace does not error when its pattern misses, so run_one() gained an
    argument at the call site that its signature would not accept. Everything
    compiled and every unit test passed; the failure only appeared when the CLI was
    actually invoked, on all six models at once.
    """
    import ast
    import pathlib as _p

    src = _p.Path(__file__).resolve().parents[1] / "tools" / "run_pipeline.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_one")
    accepts = ({a.arg for a in fn.args.kwonlyargs}
               | {a.arg for a in fn.args.args})
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "run_one")
    passes = {k.arg for k in call.keywords} | {"path"}
    assert not (passes - accepts), f"call site passes {passes - accepts}"


def test_cli_options_are_defined_for_every_forwarded_argument():
    """Each args.X the call site forwards must exist as an argparse option."""
    import ast
    import pathlib as _p

    src = _p.Path(__file__).resolve().parents[1] / "tools" / "run_pipeline.py"
    tree = ast.parse(src.read_text())
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "run_one")
    needed = {n.attr for k in call.keywords for n in ast.walk(k.value)
              if isinstance(n, ast.Attribute)
              and getattr(n.value, "id", "") == "args"}
    declared = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "add_argument"):
            for a in n.args:
                if isinstance(a, ast.Constant) and a.value.startswith("--"):
                    declared.add(a.value[2:].replace("-", "_"))
    assert needed <= declared, f"undeclared CLI options: {needed - declared}"


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
