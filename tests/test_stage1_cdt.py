"""
Parametric CDT tests. No gmsh -- all against analytic surfaces with known answers.

The invariants worth pinning are the ones whose violation is silent:
  * total area == exact domain area  (catches lost or duplicated triangles)
  * every constrained edge present   (catches failed recovery)
  * zero inverted triangles          (catches orientation bugs)
  * holes stay empty                 (catches winding/classification bugs)
  * max METRIC edge <= tolerance     (catches metric-blind sizing)
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stage1 import quality as Q
from stage1.boundary import (Loop, boundary_from_uv_polygon, orient_loops,
                             resample_uv_polygon, FaceBoundary)
from stage1.cdt_parametric import (Triangulation, recover_constraints,
                                   super_triangle, triangulate, winding_number)
from stage1.metric import MetricField
from stage1.surface import CylinderSurface, PlaneSurface, SphereSurface

PLANE = PlaneSurface()


def _square(target, lo=0.0, hi=1.0):
    poly = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    return boundary_from_uv_polygon(
        [resample_uv_polygon(poly, PLANE, target)], PLANE, target)


def _all_edges(mesh):
    t = mesh.triangles
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=0)
    return {tuple(sorted(p)) for p in e.tolist()}


# --------------------------------------------------------------------------
# Bowyer-Watson core
# --------------------------------------------------------------------------

def test_delaunay_property_on_random_points():
    """No vertex may sit strictly inside any triangle's circumcircle."""
    rng = np.random.default_rng(3)
    pts = [tuple(p) for p in rng.random((40, 2))]
    st = super_triangle(pts)
    tri = Triangulation(pts + st)
    sup = {len(pts), len(pts) + 1, len(pts) + 2}
    tri.add_triangle(*sorted(sup))
    for i in range(len(pts)):
        tri.insert_existing(i)
    assert tri.is_delaunay(ignore=sup), "empty-circumcircle property violated"


def test_insert_existing_actually_triangulates():
    """REGRESSION: insert_point's coordinate dedupe silently skipped nodes.

    Boundary nodes are registered in pts up front so their indices match
    FaceBoundary's numbering. Routing them through insert_point returned the
    existing index and never triangulated them, leaving only the super triangle
    and failing every single constraint.
    """
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    st = super_triangle(pts)
    tri = Triangulation(pts + st)
    tri.add_triangle(4, 5, 6)
    assert len(tri.tris) == 1
    for i in range(4):
        tri.insert_existing(i)
    assert len(tri.tris) > 1, "insert_existing did not create triangles"
    # All four real points must now appear in some triangle.
    used = {v for t in tri.tris.values() for v in t}
    assert {0, 1, 2, 3} <= used, used


def test_triangulation_enforces_ccw():
    tri = Triangulation([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    tid = tri.add_triangle(0, 2, 1)          # given clockwise
    a, b, c = tri.tris[tid]
    from stage1.predicates import orient2d
    assert orient2d(tri.pts[a], tri.pts[b], tri.pts[c]) > 0


# --------------------------------------------------------------------------
# Constraint recovery
# --------------------------------------------------------------------------

def test_recovers_constraint_absent_from_delaunay():
    """A thin quad's Delaunay diagonal is the SHORT one; force the long one."""
    pts = [(0.0, 0.0), (4.0, 0.0), (4.0, 0.4), (0.0, 0.4)]
    st = super_triangle(pts)
    tri = Triangulation(pts + st)
    tri.add_triangle(4, 5, 6)
    for i in range(4):
        tri.insert_existing(i)
    # Delaunay picks the short diagonal (1,3); demand the long one (0,2).
    failed = recover_constraints(tri, [(0, 2)])
    assert failed == [], failed
    assert tri.has_edge(0, 2), "constraint not recovered"


def test_all_boundary_constraints_present_in_output():
    fb = _square(0.25)
    res = triangulate(fb, PLANE)
    assert res.failed_constraints == [], res.failed_constraints


# --------------------------------------------------------------------------
# Domain classification
# --------------------------------------------------------------------------

def test_winding_number_conventions():
    ccw = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    assert winding_number((1.0, 1.0), ccw) == 1
    assert winding_number((3.0, 1.0), ccw) == 0
    cw = ccw[::-1]
    assert winding_number((1.0, 1.0), cw) == -1


def test_square_area_and_quality():
    fb = _square(0.25)
    res = triangulate(fb, PLANE)
    m = res.mesh
    a, b, c = m.corners()
    assert abs(float(Q.triangle_areas(a, b, c).sum()) - 1.0) < 1e-9
    rep = Q.evaluate(m)
    assert rep.n_inverted == 0
    assert rep.n_nonmanifold_edges == 0
    assert rep.n_inconsistent_normals == 0
    assert rep.shape_min > 0.3, rep.shape_min
    assert rep.min_angle > 20.0, rep.min_angle


def test_square_with_square_hole_leaves_hole_empty():
    """The classification test that matters: a hole must not be filled.

    Filling it produces a perfectly valid mesh of the WRONG domain, which no
    shape-quality metric would flag.
    """
    target = 0.25
    outer = resample_uv_polygon([(0, 0), (3, 0), (3, 3), (0, 3)], PLANE, target)
    inner = resample_uv_polygon([(1, 1), (2, 1), (2, 2), (1, 2)], PLANE, target)
    loops = orient_loops([
        Loop(uv=outer, xyz=np.array([PLANE.point(p) for p in outer])),
        Loop(uv=inner, xyz=np.array([PLANE.point(p) for p in inner])),
    ])
    fb = FaceBoundary(face_tag=0, loops=loops, target_size=target)
    res = triangulate(fb, PLANE)
    m = res.mesh
    a, b, c = m.corners()
    area = float(Q.triangle_areas(a, b, c).sum())
    assert abs(area - (9.0 - 1.0)) < 1e-7, f"area {area}, expected 8.0"
    # No triangle centroid may fall in the hole.
    cen = (a + b + c) / 3.0
    inside_hole = ((cen[:, 0] > 1.02) & (cen[:, 0] < 1.98) &
                   (cen[:, 1] > 1.02) & (cen[:, 1] < 1.98))
    assert not inside_hole.any(), f"{int(inside_hole.sum())} triangles in the hole"
    assert Q.evaluate(m).n_inverted == 0


def test_concave_L_domain():
    """A concave domain: the convex-hull filler must be discarded."""
    target = 0.5
    poly = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    fb = boundary_from_uv_polygon(
        [resample_uv_polygon(poly, PLANE, target)], PLANE, target)
    res = triangulate(fb, PLANE)
    a, b, c = res.mesh.corners()
    area = float(Q.triangle_areas(a, b, c).sum())
    assert abs(area - 3.0) < 1e-7, f"area {area}, expected 3.0 (L-shape)"
    cen = (a + b + c) / 3.0
    # The notch (x>1.02, y>1.02) is outside the L and must be empty.
    assert not ((cen[:, 0] > 1.02) & (cen[:, 1] > 1.02)).any()


# --------------------------------------------------------------------------
# Metric-driven sizing
# --------------------------------------------------------------------------

def test_refinement_respects_target_size_on_plane():
    target = 0.2
    fb = _square(target)
    res = triangulate(fb, PLANE, size_tolerance=1.3)
    a, b, c = res.mesh.corners()
    assert float(Q.edge_lengths(a, b, c).max()) <= 1.3 * target + 1e-9
    assert res.n_refinement_inserts > 0, "nothing was refined"


def test_finer_target_gives_more_triangles():
    coarse = triangulate(_square(0.4), PLANE).mesh
    fine = triangulate(_square(0.15), PLANE).mesh
    assert fine.n_triangles > 3 * coarse.n_triangles, (
        coarse.n_triangles, fine.n_triangles)


def test_cylinder_sizing_is_physical_not_parametric():
    """On a cylinder of radius R, a du of 1 rad is R units of arc.

    A metric-blind mesher would size in radians and produce elements R times too
    large. The check is on REAL 3D edge lengths.
    """
    R, target = 10.0, 1.0
    cyl = CylinderSurface(radius=R)
    poly = [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)]
    fb = boundary_from_uv_polygon(
        [resample_uv_polygon(poly, cyl, target)], cyl, target)
    res = triangulate(fb, cyl, size_tolerance=1.3)
    a, b, c = res.mesh.corners()
    L = Q.edge_lengths(a, b, c)
    assert float(L.max()) <= 1.3 * target + 1e-6, float(L.max())
    # Patch area: arc(1 rad * R=10) x height 5 = 50.
    area = float(Q.triangle_areas(a, b, c).sum())
    assert abs(area - 50.0) < 0.5, area


def test_area_converges_but_not_from_below():
    """Area error must shrink under refinement -- WITHOUT assuming a direction.

    A flat triangulation with vertices ON a curved surface does NOT necessarily
    under-measure it. This is the Schwarz lantern: inscribed triangulations of a
    cylinder can have total area larger than the surface, and can be made to
    converge to the wrong value entirely, because area convergence requires the
    triangle NORMALS to converge to the surface normal, not just the vertices to
    lie on it.

    Measured here: 50.05 -> 50.01 -> 50.003, converging to 50 from ABOVE. An
    earlier version of this test asserted monotone increase toward the exact value
    and failed for exactly this reason.

    The practical lesson: area is a weak convergence proxy. Chordal deviation is
    the metric that actually measures geometric fidelity.
    """
    R = 10.0
    cyl = CylinderSurface(radius=R)
    poly = [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)]
    exact = 1.0 * R * 5.0
    errs = []
    for target in (2.0, 1.0, 0.5):
        fb = boundary_from_uv_polygon(
            [resample_uv_polygon(poly, cyl, target)], cyl, target)
        a, b, c = triangulate(fb, cyl).mesh.corners()
        errs.append(abs(float(Q.triangle_areas(a, b, c).sum()) - exact) / exact)
    assert errs[0] > errs[1] > errs[2], f"error must shrink monotonically: {errs}"
    assert errs[-1] < 0.005, errs


def test_metric_linearization_fixes_anisotropic_quality():
    """The single most important behaviour in this module.

    A cylinder of radius 10 has metric diag(100, 1). Euclidean Delaunay in raw
    (u,v) produces triangles equilateral in (u,v) and 10:1 stretched in R^3.
    Working in xi = Lt @ (uv - uv0) makes the domain the unrolled 10x5 rectangle,
    where ordinary Delaunay is correct.

    Measured: min angle 3.1 -> 28.4 degrees, shape_min 0.087 -> 0.654.
    """
    R, target = 10.0, 1.0
    cyl = CylinderSurface(radius=R)
    poly = [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)]
    fb = boundary_from_uv_polygon(
        [resample_uv_polygon(poly, cyl, target)], cyl, target)

    raw = Q.evaluate(triangulate(fb, cyl, linearize_metric=False).mesh)
    lin = Q.evaluate(triangulate(fb, cyl, linearize_metric=True).mesh)

    assert raw.min_angle < 6.0, f"raw baseline should be bad, got {raw.min_angle}"
    assert lin.min_angle > 20.0, f"linearized min angle {lin.min_angle}"
    assert lin.shape_min > 5 * raw.shape_min, (raw.shape_min, lin.shape_min)
    assert lin.n_inverted == 0
    # And it should need FEWER elements, not more, for the same size target.
    n_raw = triangulate(fb, cyl, linearize_metric=False).mesh.n_triangles
    n_lin = triangulate(fb, cyl, linearize_metric=True).mesh.n_triangles
    assert n_lin < n_raw, (n_raw, n_lin)


def test_linearization_is_identity_on_a_plane():
    """A plane's metric is already the identity, so results must not change."""
    fb = _square(0.25)
    a = Q.evaluate(triangulate(fb, PLANE, linearize_metric=True).mesh)
    b = Q.evaluate(triangulate(fb, PLANE, linearize_metric=False).mesh)
    assert a.n_triangles == b.n_triangles
    assert abs(a.shape_min - b.shape_min) < 1e-9


def test_seam_face_is_refused_not_silently_wrong():
    """A parametric loop across a seam is ambiguous; refusing beats guessing."""
    fb = FaceBoundary(face_tag=7, loops=(), target_size=1.0, has_seam=True)
    try:
        triangulate(fb, PLANE)
    except ValueError as e:
        assert "seam" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for a seam face")


def test_boundary_fingerprint_is_carried_into_the_mesh():
    """Fairness: the mesh must record which boundary produced it."""
    fb = _square(0.3)
    m = triangulate(fb, PLANE).mesh
    assert m.meta["boundary_fingerprint"] == fb.fingerprint()
    assert m.method == "parametric_cdt"


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
