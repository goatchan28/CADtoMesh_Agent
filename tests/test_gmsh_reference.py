"""
Tests for the gmsh black-box reference benchmark.

Real gmsh is not required. A stub module records the geometry calls and returns a
known synthetic mesh, which exercises the parts this project actually owns:

  * node tag -> array index remapping (an off-by-one here silently scrambles the
    mesh into garbage triangles that still pass every shape metric)
  * triangle-only filtering (algorithm 11 returns quads)
  * boundary node counting, which is what makes the 1D asymmetry visible
  * geometry seeded from the analytic surface rather than hardcoded coordinates

Gmsh's meshing algorithms are neither inspected nor reproduced; only its public
API is called.
"""

import math
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from research.surface import CylinderSurface, PlaneSurface, SphereSurface

R = 10.0


# --------------------------------------------------------------------------
# Stub gmsh
# --------------------------------------------------------------------------

class _StubGmsh(types.ModuleType):
    """Minimal stand-in. Records calls; returns a fixed two-triangle mesh.

    Node tags deliberately start at 101, not 0 or 1, so any code that confuses a
    gmsh node TAG with an array INDEX fails loudly.
    """

    def __init__(self):
        super().__init__("gmsh")
        self.calls: list[tuple] = []
        self.options: dict = {}
        self.points: list[tuple] = []
        self.revolve_angle = None
        self.generated_dim = None
        self._next = 1
        self.element_type = 2          # 3-node triangle
        self.model = self._Model(self)
        self.option = self._Option(self)

    # module-level
    def initialize(self, *a):
        self.calls.append(("initialize",))

    def finalize(self):
        self.calls.append(("finalize",))

    class _Option:
        def __init__(self, o):
            self.o = o

        def setNumber(self, k, v):
            self.o.options[k] = v

    class _Model:
        def __init__(self, o):
            self.o = o
            self.occ = _StubGmsh._Occ(o)
            self.mesh = _StubGmsh._Mesh(o)

        def add(self, name):
            self.o.calls.append(("model.add", name))

        def getBoundary(self, dimtags, combined=True, oriented=False):
            return [(1, 7), (1, 8)]

    class _Occ:
        def __init__(self, o):
            self.o = o

        def _tag(self):
            self.o._next += 1
            return self.o._next

        def addPoint(self, x, y, z, *a):
            self.o.points.append((float(x), float(y), float(z)))
            return self._tag()

        def addLine(self, a, b):
            return self._tag()

        def addCircleArc(self, a, c, b):
            self.o.calls.append(("addCircleArc",))
            return self._tag()

        def addCurveLoop(self, lines):
            return self._tag()

        def addPlaneSurface(self, loops):
            t = self._tag()
            self.o.calls.append(("addPlaneSurface", t))
            return t

        def revolve(self, dimtags, ox, oy, oz, ax, ay, az, angle):
            self.o.revolve_angle = float(angle)
            return [(2, self._tag())]

        def synchronize(self):
            self.o.calls.append(("synchronize",))

    class _Mesh:
        def __init__(self, o):
            self.o = o

        def generate(self, dim):
            self.o.generated_dim = dim

        def getNodes(self, dim=None, tag=None, includeBoundary=False,
                    returnParametricCoord=False):
            if dim == 1:
                return ([101, 102], [0.0] * 6, [])
            tags = [101, 102, 103, 104]
            coords = [0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      1.0, 1.0, 0.0,
                      0.0, 1.0, 0.0]
            return (tags, coords, [])

        def getElements(self, dim, tag):
            if self.o.element_type == 2:
                return ([2], [[1, 2]], [[101, 102, 103, 101, 103, 104]])
            return ([3], [[1]], [[101, 102, 103, 104]])   # a quad


def _with_stub(fn):
    stub = _StubGmsh()
    saved = sys.modules.get("gmsh")
    sys.modules["gmsh"] = stub
    try:
        for m in [k for k in list(sys.modules) if k.startswith("tools.gmsh_reference")]:
            del sys.modules[m]
        from tools import gmsh_reference as G
        return fn(G, stub)
    finally:
        if saved is not None:
            sys.modules["gmsh"] = saved
        else:
            sys.modules.pop("gmsh", None)


SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]
CYL = [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)]
BAND = [(0, 0.9), (1.2, 0.9), (1.2, 2.2), (0, 2.2)]


# --------------------------------------------------------------------------

def test_node_tags_are_remapped_not_used_as_indices():
    """Tags start at 101; using them as indices would blow up or scramble."""
    def body(G, stub):
        r = G.mesh_case("plane", 0.5, PlaneSurface(), SQUARE)
        assert r.mesh.n_triangles == 2, r.mesh.n_triangles
        assert r.mesh.n_vertices == 4, r.mesh.n_vertices
        assert r.mesh.triangles.max() == 3, r.mesh.triangles
        assert r.mesh.triangles.min() == 0
        # The two triangles must tile the unit square: total area 1.0.
        from quality import core as Q
        a, b, c = r.mesh.corners()
        assert abs(float(Q.triangle_areas(a, b, c).sum()) - 1.0) < 1e-12
    _with_stub(body)


def test_non_triangle_elements_are_skipped():
    """Algorithm 11 yields quads; the benchmark must report empty, not garbage."""
    def body(G, stub):
        stub.element_type = 3
        r = G.mesh_case("plane", 0.5, PlaneSurface(), SQUARE, algorithm=11)
        assert r.mesh.n_triangles == 0
    _with_stub(body)


def test_gmsh_runs_its_own_1d_pass():
    """generate(2) must be called -- gmsh choosing its own boundary is the point."""
    def body(G, stub):
        G.mesh_case("plane", 0.5, PlaneSurface(), SQUARE)
        assert stub.generated_dim == 2, stub.generated_dim
        assert ("synchronize",) in stub.calls
    _with_stub(body)


def test_boundary_node_count_is_reported():
    """Needed to make the 1D asymmetry visible in the report."""
    def body(G, stub):
        r = G.mesh_case("plane", 0.5, PlaneSurface(), SQUARE)
        assert r.n_boundary_nodes == 2, r.n_boundary_nodes
    _with_stub(body)


def test_uniform_sizing_is_requested_by_default():
    """Curvature adaptation off by default, so the comparison is uniform-vs-uniform."""
    def body(G, stub):
        G.mesh_case("plane", 0.37, PlaneSurface(), SQUARE)
        assert stub.options["Mesh.MeshSizeMin"] == 0.37
        assert stub.options["Mesh.MeshSizeMax"] == 0.37
        assert stub.options["Mesh.MeshSizeFromCurvature"] == 0
        assert stub.options["Mesh.ElementOrder"] == 1
        G.mesh_case("plane", 0.37, PlaneSurface(), SQUARE, curvature_adapt=12)
        assert stub.options["Mesh.MeshSizeFromCurvature"] == 12
    _with_stub(body)


def test_algorithm_is_passed_through_unmodified():
    def body(G, stub):
        for algo in (1, 5, 6):
            r = G.mesh_case("plane", 0.5, PlaneSurface(), SQUARE, algorithm=algo)
            assert stub.options["Mesh.Algorithm"] == algo
            assert r.mesh.method == f"gmsh_algo{algo}"
            assert r.mesh.meta["own_1d_discretization"] is True
    _with_stub(body)


def test_cylinder_geometry_is_seeded_from_the_surface():
    """THE REGRESSION. Hardcoding (R,0,0) described a cylinder rotated 90 degrees.

    CylinderSurface builds e1 = unit(axis x seed), so for a z-axis cylinder u=0 is
    along +y. The benchmark must mesh the SAME surface the other three did, or its
    area and chordal numbers are not comparable.
    """
    def body(G, stub):
        surf = CylinderSurface(radius=R)
        G.mesh_case("cylinder", 1.0, surf, CYL)
        expect_lo = tuple(np.round(surf.point((0.0, 0.0)), 9))
        expect_hi = tuple(np.round(surf.point((0.0, 5.0)), 9))
        got = [tuple(np.round(p, 9)) for p in stub.points]
        assert expect_lo in got, f"{expect_lo} not among {got}"
        assert expect_hi in got, f"{expect_hi} not among {got}"
        # And NOT the hardcoded guess that was wrong.
        assert (R, 0.0, 0.0) not in got or expect_lo == (R, 0.0, 0.0)
        assert abs(stub.revolve_angle - 1.0) < 1e-12, stub.revolve_angle
    _with_stub(body)


def test_sphere_arc_endpoints_lie_on_the_sphere():
    def body(G, stub):
        sp = SphereSurface(radius=R)
        G.mesh_case("sphere_band", 1.5, sp, BAND)
        assert ("addCircleArc",) in stub.calls, "must build a true arc, not a chord"
        radii = [np.linalg.norm(np.asarray(p) - sp.center) for p in stub.points]
        on_sphere = [r for r in radii if abs(r - R) < 1e-9]
        assert len(on_sphere) >= 2, radii
        assert min(radii) < 1e-12, "the arc centre must be the sphere centre"
        assert abs(stub.revolve_angle - 1.2) < 1e-12, stub.revolve_angle
    _with_stub(body)


def test_plane_polygon_uses_all_vertices():
    def body(G, stub):
        L = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        G.mesh_case("lshape", 0.3, PlaneSurface(), L)
        assert len(stub.points) == 6, stub.points
        assert any(c[0] == "addPlaneSurface" for c in stub.calls), stub.calls
    _with_stub(body)


def test_missing_gmsh_raises_a_named_error():
    saved = sys.modules.get("gmsh")
    sys.modules["gmsh"] = None          # import gmsh -> ImportError
    try:
        for m in [k for k in list(sys.modules) if k.startswith("tools.gmsh_reference")]:
            del sys.modules[m]
        from tools import gmsh_reference as G
        try:
            G.mesh_case("plane", 0.5, PlaneSurface(), SQUARE)
        except G.GmshUnavailable:
            pass
        except Exception as e:
            raise AssertionError(f"expected GmshUnavailable, got {type(e).__name__}")
        else:
            raise AssertionError("expected GmshUnavailable")
    finally:
        if saved is not None:
            sys.modules["gmsh"] = saved
        else:
            sys.modules.pop("gmsh", None)


def test_unknown_case_is_rejected():
    def body(G, stub):
        try:
            G.mesh_case("nope", 0.5, PlaneSurface(), SQUARE)
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for an unknown case")
    _with_stub(body)


def test_every_compare_case_has_reference_geometry():
    """A case in the fair table with no reference geometry silently drops out."""
    def body(G, stub):
        from tools.compare_meshers import CASES
        missing = [c for c in CASES if c not in G.BUILDERS]
        assert not missing, f"no gmsh geometry for {missing}"
    _with_stub(body)


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
