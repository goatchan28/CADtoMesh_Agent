"""
Advancing-front tests. No gmsh -- analytic surfaces with known answers.

Two regressions are pinned here, both of which produced meshes that looked
plausible while being badly wrong:

  * Front.commit orientation. Using the triangle's CCW edges instead of their
    reverses meant the front never shrank; a unit square emitted 200000 triangles
    over 14 nodes.
  * direct-3D normal convention. Using surface.normal() instead of
    unit(dX/du x dX/dv) sent the front OUT of the patch on any surface where the
    two disagree, wrapping the sphere for 240-630% area error.
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stage1 import af_direct3d as AF3
from stage1 import af_parametric as AFP
from stage1 import quality as Q
from stage1.af_core import (AFConfig, Front, SpatialHash, apex_height,
                            ideal_apex_2d, left_normal_2d, order_candidates,
                            shape_quality_2d)
from stage1.boundary import boundary_from_uv_polygon, resample_uv_polygon
from stage1.surface import CylinderSurface, PlaneSurface, SphereSurface

PLANE = PlaneSurface()
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]
LSHAPE = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
CYL_PATCH = [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)]
SPH_BAND = [(0, 0.9), (1.2, 0.9), (1.2, 2.2), (0, 2.2)]
SPH_POLE = [(0, 0.05), (1.2, 0.05), (1.2, 1.2), (0, 1.2)]


def _bnd(surface, poly, target):
    return boundary_from_uv_polygon(
        [resample_uv_polygon(poly, surface, target)], surface, target)


# --------------------------------------------------------------------------
# Shared core
# --------------------------------------------------------------------------

def test_spatial_hash_radius_query():
    h = SpatialHash(cell=1.0, dim=2)
    pts = {i: np.array([float(i), 0.0]) for i in range(10)}
    for i, p in pts.items():
        h.add(i, p)
    near = set(h.near(np.array([5.0, 0.0]), 1.5))
    assert {4, 5, 6} <= near, near
    assert 0 not in near and 9 not in near


def test_left_normal_and_apex_side():
    d = np.array([1.0, 0.0])
    assert np.allclose(left_normal_2d(d), [0.0, 1.0]), "left of +x is +y"
    apex = ideal_apex_2d(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 1.0)
    assert apex[1] > 0, "apex must open to the LEFT (into the material)"
    assert abs(apex[0] - 0.5) < 1e-12


def test_apex_height_is_equilateral_when_target_matches_base():
    """Both methods use this one function, so element counts stay comparable."""
    assert abs(apex_height(1.0, 1.0) - math.sqrt(3) / 2) < 1e-12
    # Base much longer than target: fall back to equilateral, never degenerate.
    assert apex_height(10.0, 1.0) >= 0.35 * 10.0
    assert apex_height(1.0, 5.0) > 1.0, "large target -> tall apex"


def test_shape_quality_2d_matches_3d_formula():
    """Candidate scoring must use the same scale the final mesh is judged on."""
    equi2 = (np.array([0.0, 0.0]), np.array([1.0, 0.0]),
             np.array([0.5, math.sqrt(3) / 2]))
    assert abs(shape_quality_2d(*equi2) - 1.0) < 1e-12
    a = np.array([[0.0, 0.0, 0.0]]); b = np.array([[1.0, 0.0, 0.0]])
    c = np.array([[0.5, math.sqrt(3) / 2, 0.0]])
    assert abs(shape_quality_2d(*equi2) - float(Q.shape_quality(a, b, c)[0])) < 1e-12


def test_front_commit_uses_reversed_edges():
    """THE REGRESSION. Committing must add (k,j) and (i,k), not (j,k) and (k,i).

    Front edges carry unmeshed material on their left; triangle (i,j,k) is CCW so
    the TRIANGLE is left of (j,k) and (k,i). The remaining material is on the other
    side, hence the reversal. Getting it wrong means new edges duplicate existing
    ones instead of cancelling, the front never shrinks, and the mesher runs away.
    """
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    f = Front(pts, cell=1.0, dim=2)
    f.add_edge(0, 1)
    f.add_edge(1, 2)
    f.add_edge(2, 0)
    assert len(f.edges) == 3
    f.commit(0, 1, 2)
    # (2,1) reverses to (1,2) which is active -> cancels.
    # (0,2) reverses to (2,0) which is active -> cancels.
    assert len(f.edges) == 0, f"front should close completely, got {f.edges}"
    assert f.triangles == [(0, 1, 2)]


def test_front_commit_adds_genuinely_new_edges():
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    f = Front(pts, cell=1.0, dim=2)
    f.add_edge(0, 1)
    f.commit(0, 1, 2)
    assert f.edges == {(2, 1), (0, 2)}, f.edges


def test_order_candidates_prefers_new_node_over_distant_reuse():
    """THE OTHER ORDERING REGRESSION.

    Ranking existing nodes by distance and trying them before the ideal apex meant
    a far boundary node always won, so a unit square at target 0.25 came out as 14
    triangles with a full-diagonal edge instead of ~37 well-shaped ones.
    """
    cfg = AFConfig()
    pi, pj = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    ideal = np.array([0.5, 0.866])
    far = {7: np.array([0.5, 1.9])}          # inside reuse radius, far from ideal

    def proj(n):
        return far[n]

    out = order_candidates(pi, pj, ideal, list(far), proj, 1.0, cfg)
    assert out[0][0] is None, "the ideal NEW node must come first"

    # But a node essentially AT the ideal point must force reuse.
    at = {9: ideal + np.array([0.05, 0.0])}
    out2 = order_candidates(pi, pj, ideal, list(at), lambda n: at[n], 1.0, cfg)
    assert out2[0][0] == 9, "a node at the ideal position must be reused"


def test_order_candidates_rejects_over_long_legs():
    cfg = AFConfig()
    pi, pj = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    ideal = np.array([0.5, 0.866])
    miles = {3: np.array([0.5, 40.0])}
    out = order_candidates(pi, pj, ideal, list(miles), lambda n: miles[n], 1.0, cfg)
    assert all(n is None for n, _ in out), "a 40-unit leg must not be a candidate"


# --------------------------------------------------------------------------
# Both methods, shared expectations
# --------------------------------------------------------------------------

def _run_all(surface, poly, target):
    fb = _bnd(surface, poly, target)
    return fb, {
        "parametric_af": AFP.mesh_face(fb, surface),
        "direct3d_af": AF3.mesh_face(fb, surface),
    }


def test_both_methods_use_the_same_boundary():
    """The fairness precondition, asserted rather than assumed."""
    fb, res = _run_all(PLANE, SQUARE, 0.2)
    for name, r in res.items():
        assert r.mesh.meta["boundary_fingerprint"] == fb.fingerprint(), name
        assert r.mesh.meta["target_size"] == fb.target_size, name


def test_both_methods_close_the_front_on_simple_domains():
    for surface, poly, t in ((PLANE, SQUARE, 0.2), (PLANE, LSHAPE, 0.3),
                             (CylinderSurface(radius=10.0), CYL_PATCH, 1.0)):
        _, res = _run_all(surface, poly, t)
        for name, r in res.items():
            assert r.stats.completed, f"{name} stalled: {r.stats.summary()}"
            assert r.stats.front_remaining == 0, name


def test_both_methods_conserve_area_exactly_on_a_plane():
    for poly, exact in ((SQUARE, 1.0), (LSHAPE, 3.0)):
        _, res = _run_all(PLANE, poly, 0.25)
        for name, r in res.items():
            a, b, c = r.mesh.corners()
            got = float(Q.triangle_areas(a, b, c).sum())
            assert abs(got - exact) < 1e-9, f"{name}: {got} != {exact}"


def test_both_methods_produce_no_inverted_triangles():
    for surface, poly, t in ((PLANE, SQUARE, 0.2), (PLANE, LSHAPE, 0.3),
                             (CylinderSurface(radius=10.0), CYL_PATCH, 1.0),
                             (SphereSurface(radius=10.0), SPH_BAND, 1.5)):
        _, res = _run_all(surface, poly, t)
        for name, r in res.items():
            assert Q.evaluate(r.mesh).n_inverted == 0, name


def test_af_element_count_is_near_ideal():
    """AF should land closer to the ideal count than refinement-based CDT.

    ideal = area / (sqrt(3)/4 * h^2). Measured on the unit square at h=0.15:
    AF 106 and 102 against an ideal of 103; CDT 140.
    """
    _, res = _run_all(PLANE, SQUARE, 0.15)
    ideal = 1.0 / (math.sqrt(3) / 4 * 0.15 ** 2)
    for name, r in res.items():
        ratio = r.mesh.n_triangles / ideal
        assert 0.8 < ratio < 1.35, f"{name}: {r.mesh.n_triangles} vs ideal {ideal:.0f}"


def test_af_mean_shape_beats_cdt_on_a_plane():
    """AF's characteristic strength: better AVERAGE interior quality."""
    from stage1.cdt_parametric import triangulate
    fb = _bnd(PLANE, SQUARE, 0.15)
    cdt = Q.evaluate(triangulate(fb, PLANE).mesh)
    for name, r in (("parametric_af", AFP.mesh_face(fb, PLANE)),
                    ("direct3d_af", AF3.mesh_face(fb, PLANE))):
        rep = Q.evaluate(r.mesh)
        assert rep.shape_mean > cdt.shape_mean, (name, rep.shape_mean, cdt.shape_mean)


# --------------------------------------------------------------------------
# Direct 3D specifics
# --------------------------------------------------------------------------

def test_uv_normal_disagrees_with_surface_normal_on_the_sphere():
    """Documents the trap. unit(du x dv) is INWARD on this sphere parametrization
    while normal() is outward, so the two are opposite."""
    sp = SphereSurface(radius=10.0)
    uv = (0.0, math.pi / 2)
    du, dv = [np.asarray(x, float) for x in sp.derivatives(uv)]
    cr = np.cross(du, dv)
    cr /= np.linalg.norm(cr)
    n = np.asarray(sp.normal(uv), float)
    n /= np.linalg.norm(n)
    assert float(n @ cr) < -0.9, "sphere normals must be opposite (that is the trap)"
    # Plane and cylinder happen to agree, which is why they never exposed the bug.
    for s, p in ((PLANE, (0.3, 0.4)), (CylinderSurface(radius=10.0), (0.5, 2.0))):
        du, dv = [np.asarray(x, float) for x in s.derivatives(p)]
        cr = np.cross(du, dv)
        cr /= np.linalg.norm(cr)
        nn = np.asarray(s.normal(p), float)
        nn /= np.linalg.norm(nn)
        assert float(nn @ cr) > 0.9


def test_direct3d_does_not_escape_the_sphere_patch():
    """THE REGRESSION. Using the outward normal instead of the uv-consistent one
    marched the front off the patch and around the sphere: 538 triangles against
    an ideal of 149, with 240% area error."""
    R = 10.0
    sp = SphereSurface(radius=R)
    for poly, v0, v1 in ((SPH_BAND, 0.9, 2.2), (SPH_POLE, 0.05, 1.2)):
        fb = _bnd(sp, poly, 1.5)
        r = AF3.mesh_face(fb, sp)
        exact = R * R * 1.2 * (math.cos(v0) - math.cos(v1))
        a, b, c = r.mesh.corners()
        got = float(Q.triangle_areas(a, b, c).sum())
        assert abs(got - exact) / exact < 0.02, f"area {got} vs {exact}"
        ideal = exact / (math.sqrt(3) / 4 * 1.5 ** 2)
        assert r.mesh.n_triangles < 1.5 * ideal, (r.mesh.n_triangles, ideal)


def test_direct3d_orientation_assertion_fires_when_flipped():
    """A flipped convention must fail loudly rather than mesh the wrong region."""
    class Flipped:
        def __init__(self, inner):
            self._s = inner
        def __getattr__(self, k):
            return getattr(self._s, k)
        def derivatives(self, uv):
            du, dv = self._s.derivatives(uv)
            return dv, du            # swapping reverses du x dv

    sp = SphereSurface(radius=10.0)
    fb = _bnd(sp, SPH_BAND, 1.5)
    try:
        AF3.mesh_face(fb, Flipped(sp))
    except ValueError as e:
        assert "orientation" in str(e).lower(), e
    else:
        raise AssertionError("expected the orientation guard to fire")


def test_direct3d_beats_parametric_near_a_sphere_pole():
    """The reason this method exists.

    At the pole det(I) collapses, so the parametric methods degrade. Measured at
    target 1.5 on R=10:
        parametric_cdt  shape_min 0.535, min angle 19.3, closed
        parametric_af   shape_min 0.136, min angle  4.5, STALLED
        direct3d_af     shape_min 0.602, min angle 21.9, closed
    """
    sp = SphereSurface(radius=10.0)
    fb = _bnd(sp, SPH_POLE, 1.5)
    p = AFP.mesh_face(fb, sp)
    d = AF3.mesh_face(fb, sp)
    rp, rd = Q.evaluate(p.mesh), Q.evaluate(d.mesh)
    assert rd.min_angle > rp.min_angle, (rp.min_angle, rd.min_angle)
    assert rd.shape_min > 2 * rp.shape_min, (rp.shape_min, rd.shape_min)
    assert d.stats.completed, "direct3d should close where parametric AF stalls"


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
