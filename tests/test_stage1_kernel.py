"""
Tests for the Stage 1 geometric kernel: robust predicates and metric geometry.
No gmsh needed.

These two modules decide whether the meshers work at all. A wrong predicate sign
does not give a slightly-wrong mesh -- it gives an infinite loop or an inverted
triangle. A wrong metric gives triangles that look fine in (u,v) and are stretched
garbage in R^3.
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stage1.predicates import (circumcenter, incircle, orient2d,
                              point_in_triangle, segments_properly_intersect)
from stage1.metric import (MetricField, anisotropy, cholesky_frame, is_degenerate,
                           metric_distance, metric_tensor,
                           parametric_step_for_length)
from stage1.surface import CylinderSurface, PlaneSurface, SphereSurface


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------

def test_orient2d_basic():
    assert orient2d((0, 0), (1, 0), (0, 1)) == 1, "CCW"
    assert orient2d((0, 0), (0, 1), (1, 0)) == -1, "CW"
    assert orient2d((0, 0), (1, 1), (2, 2)) == 0, "collinear"


def test_orient2d_exact_near_degenerate():
    """Float arithmetic gets this wrong; the exact fallback must not.

    (0,0), (1e-15, 1e-15) scaled collinear points where the float determinant
    underflows to a wrong sign.
    """
    a, b = (0.0, 0.0), (1.0, 1.0)
    # Exactly on the line y = x, so every one of these is collinear.
    for t in (0.1, 0.3, 1e-8, 1 - 1e-16):
        assert orient2d(a, b, (t, t)) == 0, t
    # Just barely off the line. 0.5 + 1e-17 is NOT usable here -- it rounds to
    # exactly 0.5 in float64 (eps at 0.5 is ~1.1e-16), so the point would be
    # genuinely collinear. nextafter gives the smallest representable step, which
    # is what actually drives the float filter to its noise floor and forces the
    # exact fallback to decide.
    up = np.nextafter(0.5, 1.0)
    dn = np.nextafter(0.5, 0.0)
    assert up != 0.5 and dn != 0.5, "perturbation must be representable"
    assert orient2d(a, b, (0.5, up)) == 1
    assert orient2d(a, b, (0.5, dn)) == -1


def test_incircle_basic():
    # Unit square corners are cocircular about (0.5, 0.5).
    a, b, c = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)
    assert orient2d(a, b, c) == 1, "fixture must be CCW"
    assert incircle(a, b, c, (0.0, 1.0)) == 0, "cocircular"
    assert incircle(a, b, c, (0.5, 0.5)) == 1, "centre is inside"
    assert incircle(a, b, c, (5.0, 5.0)) == -1, "far point outside"


def test_incircle_cocircular_exact():
    """Points exactly on a circle of radius 1 must read 0, not noise."""
    a, b, c = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    assert orient2d(a, b, c) == 1
    assert incircle(a, b, c, (0.0, -1.0)) == 0
    assert incircle(a, b, c, (0.0, -0.999999999)) == 1
    assert incircle(a, b, c, (0.0, -1.000000001)) == -1


def test_segment_intersection_ignores_shared_endpoints():
    """Front edges share vertices constantly; that must not read as a crossing."""
    assert not segments_properly_intersect((0, 0), (1, 0), (1, 0), (1, 1))
    assert not segments_properly_intersect((0, 0), (1, 0), (0, 0), (0, 1))
    assert segments_properly_intersect((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_properly_intersect((0, 0), (1, 0), (0, 1), (1, 1))


def test_collinear_overlap_counts_as_intersection():
    assert segments_properly_intersect((0, 0), (2, 0), (1, 0), (3, 0))
    assert not segments_properly_intersect((0, 0), (1, 0), (2, 0), (3, 0))


def test_circumcenter_of_equilateral():
    h = math.sqrt(3.0) / 2.0
    cx, cy = circumcenter((0.0, 0.0), (1.0, 0.0), (0.5, h))
    assert abs(cx - 0.5) < 1e-12
    assert abs(cy - h / 3.0) < 1e-12, "centroid == circumcenter for equilateral"


def test_point_in_triangle():
    a, b, c = (0, 0), (1, 0), (0, 1)
    assert point_in_triangle((0.25, 0.25), a, b, c)
    assert point_in_triangle((0.0, 0.0), a, b, c), "vertex is inclusive"
    assert not point_in_triangle((1.0, 1.0), a, b, c)


# --------------------------------------------------------------------------
# Metric geometry
# --------------------------------------------------------------------------

def test_plane_metric_is_identity():
    """A plane's parametrization is isometric, so the metric is I."""
    s = PlaneSurface()
    M = metric_tensor(*[np.asarray(d, float) for d in s.derivatives((0.3, 0.7))])
    assert np.allclose(M, np.eye(2)), M
    assert abs(anisotropy(M) - 1.0) < 1e-12
    assert not is_degenerate(M)


def test_cylinder_metric_is_anisotropic_by_radius():
    """|Xu| = R, |Xv| = 1, orthogonal. So M = diag(R^2, 1)."""
    R = 12.0
    s = CylinderSurface(radius=R)
    M = metric_tensor(*[np.asarray(d, float) for d in s.derivatives((0.4, 3.0))])
    assert np.allclose(M, np.diag([R * R, 1.0])), M
    assert abs(anisotropy(M) - R) < 1e-9, "stretch ratio must equal the radius"


def test_cholesky_frame_linearizes_the_metric():
    """|d|_M must equal the plain Euclidean length of Lt @ d."""
    rng = np.random.default_rng(0)
    for M in (np.diag([144.0, 1.0]), np.array([[4.0, 1.0], [1.0, 9.0]]),
              np.array([[1e6, 0.0], [0.0, 1e-2]])):
        Lt, Lt_inv = cholesky_frame(M)
        for _ in range(20):
            d = rng.normal(size=2)
            assert abs(np.linalg.norm(Lt @ d) - metric_distance(M, d)) < 1e-9
            xi = Lt @ d
            assert np.allclose(Lt_inv @ xi, d, atol=1e-9), "inverse must round-trip"


def test_cholesky_frame_survives_degenerate_metric():
    """A sphere pole collapses det I. The frame must regularize, not crash."""
    s = SphereSurface(radius=5.0)
    M = metric_tensor(*[np.asarray(d, float) for d in s.derivatives((0.5, 1e-14))])
    assert is_degenerate(M), "pole must register as degenerate"
    assert anisotropy(M) > 1e6
    Lt, Lt_inv = cholesky_frame(M)
    assert np.all(np.isfinite(Lt)) and np.all(np.isfinite(Lt_inv))


def test_sphere_metric_matches_closed_form():
    """For a unit sphere, M = diag(sin^2 v, 1); area scale = sin v."""
    s = SphereSurface(radius=1.0)
    for v in (0.3, 1.0, math.pi / 2, 2.5):
        M = metric_tensor(*[np.asarray(d, float) for d in s.derivatives((0.7, v))])
        assert np.allclose(M, np.diag([math.sin(v) ** 2, 1.0]), atol=1e-12), v
        assert abs(math.sqrt(np.linalg.det(M)) - abs(math.sin(v))) < 1e-12


def test_parametric_step_spans_requested_real_length():
    """The step-sizing helper is what turns a target element size into a du."""
    R = 12.0
    M = np.diag([R * R, 1.0])
    for target in (0.5, 2.0, 7.5):
        d = parametric_step_for_length(M, np.array([1.0, 0.0]), target)
        assert abs(metric_distance(M, d) - target) < 1e-12
        # Along u the step must be target/R, since |Xu| = R.
        assert abs(d[0] - target / R) < 1e-12


def test_metric_field_distance_matches_chord_on_a_plane():
    """On a plane the metric distance is exact, so it must equal the chord."""
    mf = MetricField(PlaneSurface())
    a, b = (0.2, 0.4), (1.7, 2.9)
    assert abs(mf.distance(a, b) - mf.chord_distance(a, b)) < 1e-12


def test_metric_field_distance_beats_naive_on_a_cylinder():
    """A cylinder of radius R: a du of 0.5 rad is an ARC of 0.5R.

    The metric length must be the arc length, and it must exceed the straight
    chord -- which is the whole reason a parametric mesher needs the metric.
    """
    R = 10.0
    mf = MetricField(CylinderSurface(radius=R))
    a, b = (0.0, 0.0), (0.5, 0.0)
    arc = mf.distance(a, b)
    chord = mf.chord_distance(a, b)
    assert abs(arc - 0.5 * R) < 1e-9, arc
    assert arc > chord, "arc must exceed chord"
    assert abs(chord - 2 * R * math.sin(0.25)) < 1e-9


def test_metric_field_caches():
    mf = MetricField(PlaneSurface())
    for _ in range(5):
        mf.at((0.5, 0.5))
    assert mf.n_evals == 1, f"expected 1 evaluation, got {mf.n_evals}"


def test_predicates_accept_numpy_scalars():
    """REGRESSION: coordinates arrive as np.float64, not float.

    tuple(some_numpy_row) yields np.float64 elements, so a sign implementation
    written as (x>0)-(x<0) raises TypeError on np.bool_ subtraction. Every mesher
    feeds numpy coordinates into these predicates, so this is the normal path.
    """
    a = tuple(np.array([0.0, 0.0]))
    b = tuple(np.array([1.0, 0.0]))
    c = tuple(np.array([0.0, 1.0]))
    assert isinstance(a[0], np.floating), "fixture must use numpy scalars"
    assert orient2d(a, b, c) == 1
    assert incircle(a, b, c, tuple(np.array([0.1, 0.1]))) == 1
    assert not segments_properly_intersect(a, b, b, c)
    assert point_in_triangle(tuple(np.array([0.2, 0.2])), a, b, c)
    # Exact fallback path must also survive numpy input.
    up = np.nextafter(0.5, 1.0)
    assert orient2d(a, tuple(np.array([1.0, 1.0])),
                    tuple(np.array([0.5, up]))) == 1


def test_cholesky_frame_is_orientation_preserving():
    """det(Lt) > 0 is required, not incidental.

    numpy.linalg.eigh's eigenvector matrix can have determinant -1. If Lt inherits
    that, the change of coordinates is a REFLECTION: every triangle's winding
    reverses, and a mesher using this frame as its working coordinate system emits
    an entirely inverted mesh with no error anywhere.
    """
    mats = [np.diag([144.0, 1.0]), np.diag([1.0, 144.0]),
            np.array([[4.0, 1.0], [1.0, 9.0]]),
            np.array([[9.0, -2.0], [-2.0, 1.0]]),
            np.array([[1e6, 0.0], [0.0, 1e-2]])]
    rng = np.random.default_rng(7)
    for _ in range(30):
        A = rng.normal(size=(2, 2))
        mats.append(A @ A.T + 1e-3 * np.eye(2))
    for M in mats:
        Lt, Lt_inv = cholesky_frame(M)
        assert np.linalg.det(Lt) > 0, f"reflection for M={M.tolist()}"
        # Must still linearize the metric after the sign fix.
        d = np.array([0.3, -0.8])
        assert abs(np.linalg.norm(Lt @ d) - metric_distance(M, d)) < 1e-9
        assert np.allclose(Lt_inv @ (Lt @ d), d, atol=1e-9)


def test_cylinder_frame_unrolls_the_domain():
    """A cylinder's metric is constant, so the frame is an exact unrolling.

    R=10 means xi = (10u, v): a u-extent of 1 rad becomes 10 units of arc. This is
    why metric-normalized Delaunay produces correctly shaped triangles there.
    """
    R = 10.0
    s = CylinderSurface(radius=R)
    M = metric_tensor(*[np.asarray(d, float) for d in s.derivatives((0.5, 2.0))])
    Lt, _ = cholesky_frame(M)
    span = Lt @ np.array([1.0, 0.0])
    assert abs(np.linalg.norm(span) - R) < 1e-9, span
    span_v = Lt @ np.array([0.0, 5.0])
    assert abs(np.linalg.norm(span_v) - 5.0) < 1e-9, span_v


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
