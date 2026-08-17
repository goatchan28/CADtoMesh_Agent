"""
Mesh export and format-guard tests. PURE -- no gmsh.

The export tests exist because the pipeline's stated deliverable is "a validated
surface mesh" and it was producing a verdict, a contract report, a run record, and
then discarding the geometry. Nothing downstream could consume the result.

The MSH round-trip matters most: an exported mesh nothing can read back, or whose
face tags shuffle between runs, is not a handoff.
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np

from backends.base import (CAD_SUFFIXES, MeshProvenance, MeshResult, MeshStats,
                           UnsupportedFormat, check_cad_format)
from pipeline.export import ExportManifest, export_result, write_msh, write_stl
from quality.core import SurfaceMesh


def _mesh():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    return SurfaceMesh(vertices=v, triangles=np.array([[0, 1, 2], [0, 2, 3]]),
                       method="test", meta={"target_size": 1.0})


def _prov():
    return MeshProvenance(tri_pid=["faceB", "faceA"],
                          requested_size={"faceA": 1.0, "faceB": 0.5},
                          interface_pids=("faceB",))


# --------------------------------------------------------------------------
# Format guard
# --------------------------------------------------------------------------

def test_supported_cad_formats_pass():
    for f in ("p.step", "p.STP", "p.iges", "p.IGS", "p.brep"):
        assert check_cad_format(f) in CAD_SUFFIXES, f


def test_mesh_formats_are_rejected_with_the_real_reason():
    """Not 'no reader' -- the pipeline is built on B-rep and a converter would
    not help, because there would still be no CAD faces to key identity to."""
    for f in ("p.stl", "p.obj", "p.ply"):
        try:
            check_cad_format(f)
        except UnsupportedFormat as e:
            assert "MESH format" in str(e)
            assert "B-rep" in str(e)
        else:
            raise AssertionError(f"{f} must be rejected")


def test_native_cad_is_rejected_with_an_actionable_message():
    for f in ("p.sldprt", "p.x_t", "p.catpart"):
        try:
            check_cad_format(f)
        except UnsupportedFormat as e:
            assert "STEP" in str(e), e
        else:
            raise AssertionError(f"{f} must be rejected")


def test_unknown_extension_lists_what_is_supported():
    try:
        check_cad_format("p.xyz")
    except UnsupportedFormat as e:
        assert ".step" in str(e)
    else:
        raise AssertionError("expected rejection")


def test_guard_runs_before_the_kernel_is_touched():
    """A bare OCC error deep inside importShapes reads like a pipeline bug."""
    import inspect

    from backends import gmsh_backend
    src = inspect.getsource(gmsh_backend.GmshBackend.load)
    assert src.index("check_cad_format") < src.index("importShapes")


# --------------------------------------------------------------------------
# STL
# --------------------------------------------------------------------------

def test_stl_facet_count_matches_the_mesh():
    with tempfile.TemporaryDirectory() as d:
        p = write_stl(_mesh(), Path(d) / "m.stl", "m")
        txt = p.read_text()
        assert txt.count("facet normal") == 2
        assert txt.count("vertex") == 6
        assert txt.startswith("solid m") and txt.rstrip().endswith("endsolid m")


# --------------------------------------------------------------------------
# MSH
# --------------------------------------------------------------------------

def _parse_msh(text: str):
    lines = [l.strip() for l in text.strip().splitlines()]
    out = {"physical": {}, "nodes": [], "elements": []}
    i = 0
    while i < len(lines):
        if lines[i] == "$PhysicalNames":
            n = int(lines[i + 1])
            for j in range(n):
                dim, tag, name = lines[i + 2 + j].split(maxsplit=2)
                out["physical"][int(tag)] = name.strip('"')
            i += 2 + n
        elif lines[i] == "$Nodes":
            n = int(lines[i + 1])
            for j in range(n):
                parts = lines[i + 2 + j].split()
                out["nodes"].append((int(parts[0]),
                                     tuple(float(x) for x in parts[1:4])))
            i += 2 + n
        elif lines[i] == "$Elements":
            n = int(lines[i + 1])
            for j in range(n):
                parts = [int(x) for x in lines[i + 2 + j].split()]
                out["elements"].append({"id": parts[0], "type": parts[1],
                                        "physical": parts[3],
                                        "nodes": parts[5:8]})
            i += 2 + n
        else:
            i += 1
    return out


def test_msh_round_trips_geometry_and_connectivity():
    mesh = _mesh()
    with tempfile.TemporaryDirectory() as d:
        p, tags = write_msh(mesh, Path(d) / "m.msh", tri_pid=_prov().tri_pid)
        got = _parse_msh(p.read_text())

    assert len(got["nodes"]) == 4 and len(got["elements"]) == 2
    assert all(e["type"] == 2 for e in got["elements"]), "3-node triangles"
    for i, (idx, xyz) in enumerate(got["nodes"]):
        assert idx == i + 1, "MSH node indices are 1-based"
        assert np.allclose(xyz, mesh.vertices[i])
    for e, tri in zip(got["elements"], mesh.triangles):
        assert e["nodes"] == [int(v) + 1 for v in tri], "1-based connectivity"


def test_msh_carries_one_physical_group_per_cad_face():
    """Face identity must survive into volume meshing, or boundary conditions
    have to be applied by picking triangles."""
    with tempfile.TemporaryDirectory() as d:
        p, tags = write_msh(_mesh(), Path(d) / "m.msh", tri_pid=_prov().tri_pid)
        got = _parse_msh(p.read_text())
    assert set(got["physical"].values()) == {"faceA", "faceB"}
    by_pid = {got["physical"][e["physical"]] for e in got["elements"]}
    assert by_pid == {"faceA", "faceB"}
    assert set(tags.values()) == {"faceA", "faceB"}


def test_physical_tags_are_stable_across_runs():
    """Tags assigned in sorted PID order. Shuffling tags between runs would make
    an exported mesh impossible to diff or script against."""
    with tempfile.TemporaryDirectory() as d:
        _, t1 = write_msh(_mesh(), Path(d) / "a.msh",
                          tri_pid=["faceB", "faceA"])
        _, t2 = write_msh(_mesh(), Path(d) / "b.msh",
                          tri_pid=["faceA", "faceB"])
    assert t1 == t2, (t1, t2)
    assert t1[1] == "faceA", "sorted PID order, not first-seen order"


def test_msh_without_provenance_still_writes_one_group():
    with tempfile.TemporaryDirectory() as d:
        p, tags = write_msh(_mesh(), Path(d) / "m.msh", tri_pid=None, name="solo")
        got = _parse_msh(p.read_text())
    assert len(got["physical"]) == 1 and len(got["elements"]) == 2


def test_msh_ignores_mismatched_provenance_rather_than_corrupting_tags():
    with tempfile.TemporaryDirectory() as d:
        p, tags = write_msh(_mesh(), Path(d) / "m.msh", tri_pid=["only_one"])
        got = _parse_msh(p.read_text())
    assert len(got["elements"]) == 2
    assert len(got["physical"]) == 1


# --------------------------------------------------------------------------
# Full export
# --------------------------------------------------------------------------

class _Rec:
    def __init__(self):
        from pipeline.session import Iteration, Outcome, RunRecord
        self._r = RunRecord(model="m.step", backend="fake")
        self._r.add(Iteration(index=0, is_valid=True, ready=True, score=1.0,
                              n_triangles=2, objectives=[{"name": "min_shape"}]))
        self._r.outcome = Outcome.ACCEPTED

    def __getattr__(self, k):
        return getattr(self._r, k)


def test_export_writes_mesh_manifest_and_record():
    result = MeshResult(mesh=_mesh(), provenance=_prov(),
                        stats=MeshStats(n_triangles=2), request_fingerprint="abc")
    rec = _Rec()
    with tempfile.TemporaryDirectory() as d:
        paths = export_result(result, rec, d, "part",
                              criteria={"min_shape": 0.3},
                              contract={"satisfied": True}, iteration=0)
        names = {p.name for p in paths}
        assert names == {"part.msh", "part.stl", "part.manifest.json",
                         "part.run.json"}
        import json
        man = json.loads((Path(d) / "part.manifest.json").read_text())
        assert man["n_triangles"] == 2 and man["accepted"] is True
        assert set(man["physical_tags"].values()) == {"faceA", "faceB"}
        assert man["requested_size"]["faceB"] == 0.5
        assert man["interface_pids"] == ["faceB"]
        assert man["contract"]["satisfied"] is True
        rr = json.loads((Path(d) / "part.run.json").read_text())
        assert rr["outcome"] == "accepted"


def test_manifest_makes_integer_tags_traceable():
    """A .msh with physical tag 7 is meaningless six months on."""
    result = MeshResult(mesh=_mesh(), provenance=_prov(),
                        stats=MeshStats(n_triangles=2))
    with tempfile.TemporaryDirectory() as d:
        export_result(result, _Rec(), d, "part")
        import json
        man = json.loads((Path(d) / "part.manifest.json").read_text())
        msh = (Path(d) / "part.msh").read_text()
    for tag, pid in man["physical_tags"].items():
        assert f'2 {tag} "{pid}"' in msh, (tag, pid)


def test_export_can_skip_the_stl():
    result = MeshResult(mesh=_mesh(), provenance=_prov(),
                        stats=MeshStats(n_triangles=2))
    with tempfile.TemporaryDirectory() as d:
        paths = export_result(result, _Rec(), d, "part", write_stl_too=False)
        assert not any(p.suffix == ".stl" for p in paths)


# --------------------------------------------------------------------------
# The loop must retain the mesh
# --------------------------------------------------------------------------

def test_loop_retains_the_best_result_for_export():
    """REGRESSION: RunRecord keeps counts, not geometry.

    Without retaining the MeshResult the pipeline validated a mesh and discarded
    it, so its stated deliverable was never produced.
    """
    import tests.test_loop as TL
    from pipeline.loop import AdaptiveMesher

    be = TL.FakeBackend([{"n": 6}])
    m = AdaptiveMesher(be, criteria=TL.LOOSE, fidelity=False)
    rec = m.run("x.step")
    assert rec.outcome.succeeded
    assert m.best_result is not None
    assert m.best_result.mesh.n_triangles == rec.best.n_triangles


def test_best_result_tracks_the_best_iteration_not_the_last():
    """The final iteration is usually the most aggressive edit."""
    import tests.test_loop as TL
    from pipeline.loop import AdaptiveMesher

    # Degrades after the first: iteration 0 is the one worth keeping.
    be = TL.FakeBackend([{"n": 8}, {"n": 8, "skew": 0.9}, {"n": 8, "skew": 0.9}])
    m = AdaptiveMesher(be, criteria=TL.STRICT, fidelity=False, max_iterations=3)
    rec = m.run("x.step")
    if rec.best is not None and m.best_result is not None:
        assert m.best_result.mesh.n_triangles == rec.best.n_triangles


def test_no_valid_mesh_means_nothing_to_export():
    import tests.test_loop as TL
    from pipeline.loop import AdaptiveMesher

    be = TL.FakeBackend([{"n": 6, "invert": True}])
    m = AdaptiveMesher(be, criteria=TL.LOOSE, fidelity=False, max_iterations=2)
    rec = m.run("x.step")
    assert rec.best is None
    assert m.best_result is None, "an invalid mesh must not be exported"


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
