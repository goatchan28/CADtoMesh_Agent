"""
Quality-metric tests. No gmsh -- analytic surfaces with closed-form answers.

Chordal deviation is the one metric here that can be checked EXACTLY. A chord
subtending angle theta on radius R sags by R(1-cos(theta/2)), so a cylinder and a
sphere give ground truth to compare against. Every shape metric can only be
checked against reference triangles.
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quality import core as Q
from research.boundary import boundary_from_uv_polygon, resample_uv_polygon
from research.cdt_parametric import triangulate
from research.surface import CylinderSurface, PlaneSurface, SphereSurface

PLANE = PlaneSurface()


def _mesh(surface, poly, target):
    fb = boundary_from_uv_polygon(
        [resample_uv_polygon(poly, surface, target)], surface, target)
    return triangulate(fb, surface).mesh


CYL_PATCH = [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)]


# --------------------------------------------------------------------------
# Closed-form reference
# --------------------------------------------------------------------------

def test_sagitta_closed_form():
    # A chord equal to the diameter is a semicircle: sagitta == R.
    assert abs(Q.sagitta(5.0, 10.0) - 5.0) < 1e-12
    # Small-chord approximation c^2/(8R).
    for R, c in ((10.0, 1.0), (10.0, 0.5), (100.0, 2.0)):
        assert abs(Q.sagitta(R, c) - c * c / (8 * R)) < 1e-4 * c * c / (8 * R) * 10
    # Exact identity: sagitta == R(1 - cos(theta/2)) with theta = 2*asin(c/2R).
    R, c = 10.0, 3.0
    theta = 2 * math.asin(c / (2 * R))
    assert abs(Q.sagitta(R, c) - R * (1 - math.cos(theta / 2))) < 1e-12
    # Degenerate guard: chord wider than the circle must not produce a NaN.
    assert Q.sagitta(1.0, 5.0) == 1.0


# --------------------------------------------------------------------------
# Chordal deviation
# --------------------------------------------------------------------------

def test_chordal_is_exactly_zero_on_a_plane():
    """The mesh lies IN the surface, so every sample projects onto itself.

    This is the test that sampling and projection agree; any nonzero value here
    would be a bug in one of them, not geometry.
    """
    m = _mesh(PLANE, [(0, 0), (1, 0), (1, 1), (0, 1)], 0.25)
    ch = Q.chordal_deviation(m, PLANE, target_size=0.25)
    assert ch.max_deviation == 0.0, ch.max_deviation
    assert ch.mean_deviation == 0.0
    assert ch.max_relative == 0.0


def test_chordal_matches_sagitta_on_a_cylinder():
    """Worst facet must sag about as much as its longest chord, converging to it.

    The ratio is BELOW 1 at coarse sizes because the longest edge is only partly
    circumferential -- an axial edge has zero curvature. As elements shrink, edges
    align with the principal directions and the ratio approaches 1.
    Measured: 0.64 -> 0.94 -> 0.99.
    """
    R = 10.0
    cyl = CylinderSurface(radius=R)
    ratios = []
    for t in (2.0, 1.0, 0.5):
        m = _mesh(cyl, CYL_PATCH, t)
        ch = Q.chordal_deviation(m, cyl, target_size=t)
        maxL = float(Q.edge_lengths(*m.corners()).max())
        ratios.append(ch.max_deviation / Q.sagitta(R, maxL))
        assert ch.max_deviation < Q.sagitta(R, maxL) * 1.05, (t, ratios[-1])
    # The ratio rises toward 1 but need not be monotone: which edge happens to be
    # longest, and how closely it aligns with the circumferential direction,
    # varies between refinement levels. Measured: 0.64 -> 1.00 -> 0.98.
    assert ratios[0] < 0.8, f"coarse level should undershoot: {ratios}"
    assert all(r > 0.9 for r in ratios[1:]), ratios


def test_chordal_shrinks_quadratically_with_element_size():
    """Deviation ~ L^2/(8R): halving the target should roughly quarter it.

    This quadratic law is the basis of curvature-driven mesh sizing, so it is
    worth pinning rather than assuming.
    """
    R = 10.0
    cyl = CylinderSurface(radius=R)
    devs = []
    for t in (1.0, 0.5, 0.25):
        m = _mesh(cyl, CYL_PATCH, t)
        devs.append(Q.chordal_deviation(m, cyl, target_size=t).max_deviation)
    for lo, hi in zip(devs[1:], devs[:-1]):
        assert 2.5 < hi / lo < 5.5, f"ratio {hi/lo:.2f} not near 4x: {devs}"


def test_chordal_detects_curvature_that_shape_quality_cannot():
    """The whole point of the metric.

    A coarse cylinder mesh can have EXCELLENT shape quality while being
    geometrically far from the surface. Shape metrics are blind to this because
    they never look at the surface at all.
    """
    R = 10.0
    cyl = CylinderSurface(radius=R)
    coarse = _mesh(cyl, CYL_PATCH, 2.0)
    rep = Q.evaluate(coarse)
    ch = Q.chordal_deviation(coarse, cyl, target_size=2.0)
    assert rep.shape_mean > 0.8, "shape quality should look good"
    assert rep.min_angle > 25.0, "angles should look good"
    assert ch.max_relative > 0.02, "yet the facets are measurably off the surface"


def test_chordal_and_shape_are_independent_metrics():
    """A sphere pole degrades SHAPE but not FIDELITY, and that is the point.

    A sphere has CONSTANT curvature, so chordal deviation depends only on edge
    length -- not on where on the sphere the element sits. The pole wrecks the
    parametrization (det I -> 0), which wrecks triangle shape, while leaving
    geometric fidelity essentially untouched.

    Measured at target 1.5, R=10:
        band  shape_min 0.630, min angle 26.8, chordal 0.0566 (maxL 1.906)
        pole  shape_min 0.535, min angle 19.3, chordal 0.0506 (maxL 1.914)

    So the two numbers move independently, which is exactly why shape quality
    alone is not a sufficient description of a surface mesh. An earlier version of
    this test asserted chordal error would be WORSE at the pole; it is not, because
    curvature is what drives chordal error and curvature is uniform here.
    """
    R, target = 10.0, 1.5
    sp = SphereSurface(radius=R)
    band = _mesh(sp, [(0, 0.9), (1.2, 0.9), (1.2, 2.2), (0, 2.2)], target)
    pole = _mesh(sp, [(0, 0.05), (1.2, 0.05), (1.2, 1.2), (0, 1.2)], target)

    rb, rp = Q.evaluate(band), Q.evaluate(pole)
    cb = Q.chordal_deviation(band, sp, target_size=target)
    cp = Q.chordal_deviation(pole, sp, target_size=target)

    # Shape degrades at the pole.
    assert rp.shape_min < rb.shape_min, (rb.shape_min, rp.shape_min)
    assert rp.min_angle < rb.min_angle, (rb.min_angle, rp.min_angle)

    # Fidelity does not: constant curvature means chordal error tracks edge length.
    Lb = float(Q.edge_lengths(*band.corners()).max())
    Lp = float(Q.edge_lengths(*pole.corners()).max())
    assert abs(Lb - Lp) / Lb < 0.15, (Lb, Lp)
    assert abs(cb.max_deviation - cp.max_deviation) / cb.max_deviation < 0.35, (
        cb.max_deviation, cp.max_deviation)
    # Both close to the closed-form sagitta of their own longest chord.
    for ch, L in ((cb, Lb), (cp, Lp)):
        assert 0.8 < ch.max_deviation / Q.sagitta(R, L) < 1.6, (
            ch.max_deviation, Q.sagitta(R, L))


def test_chordal_relative_is_scale_free():
    """max_relative normalizes by target_size, because 0.1 mm of error means
    nothing without knowing whether elements are 1 mm or 100 mm."""
    cyl = CylinderSurface(radius=10.0)
    m = _mesh(cyl, CYL_PATCH, 1.0)
    ch = Q.chordal_deviation(m, cyl, target_size=1.0)
    ch2 = Q.chordal_deviation(m, cyl, target_size=2.0)
    assert abs(ch.max_deviation - ch2.max_deviation) < 1e-15, "absolute is the same"
    assert abs(ch2.max_relative - ch.max_relative / 2.0) < 1e-12


def test_chordal_subsampling_and_empty_mesh():
    cyl = CylinderSurface(radius=10.0)
    m = _mesh(cyl, CYL_PATCH, 0.5)
    full = Q.chordal_deviation(m, cyl, target_size=0.5)
    sub = Q.chordal_deviation(m, cyl, target_size=0.5, max_triangles=50)
    assert sub.n_triangles_sampled == 50
    assert full.n_triangles_sampled == m.n_triangles
    assert sub.max_deviation <= full.max_deviation + 1e-12
    empty = Q.SurfaceMesh(vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3), int))
    z = Q.chordal_deviation(empty, cyl)
    assert z.max_deviation == 0.0 and z.n_samples == 0


def test_chordal_samples_include_edge_midpoints():
    """Sampling must hit edge midpoints and the centroid explicitly.

    Chordal error peaks there. Random interior sampling can miss the peak and
    under-report, which would make the metric quietly optimistic.
    """
    bary = Q.DEFAULT_BARYCENTRIC
    assert (1 / 3, 1 / 3, 1 / 3) in bary, "centroid must be sampled"
    for mid in ((0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)):
        assert mid in bary, f"edge midpoint {mid} must be sampled"
    for w in bary:
        assert abs(sum(w) - 1.0) < 1e-12, f"{w} is not barycentric"
        assert all(x >= 0 for x in w)


# --------------------------------------------------------------------------
# Shape metrics against reference triangles
# --------------------------------------------------------------------------

def _tri(p0, p1, p2):
    a = np.array([p0], float)
    b = np.array([p1], float)
    c = np.array([p2], float)
    return a, b, c


EQUI = _tri((0, 0, 0), (1, 0, 0), (0.5, math.sqrt(3) / 2, 0))
RIGHT_ISO = _tri((0, 0, 0), (1, 0, 0), (0, 1, 0))
SLIVER = _tri((0, 0, 0), (1, 0, 0), (0.5, 0.001, 0))


def test_shape_quality_reference_values():
    assert abs(float(Q.shape_quality(*EQUI)[0]) - 1.0) < 1e-12
    assert abs(float(Q.shape_quality(*RIGHT_ISO)[0]) - math.sqrt(3) / 2) < 1e-12
    assert float(Q.shape_quality(*SLIVER)[0]) < 0.01


def test_radius_ratio_reference_values():
    assert abs(float(Q.radius_ratio(*EQUI)[0]) - 1.0) < 1e-12
    assert float(Q.radius_ratio(*SLIVER)[0]) < 0.01


def test_angles_reference_values():
    assert np.allclose(Q.angles_deg(*EQUI)[0], [60, 60, 60], atol=1e-9)
    ang = sorted(Q.angles_deg(*RIGHT_ISO)[0])
    assert np.allclose(ang, [45, 45, 90], atol=1e-9)


def test_aspect_ratio_reference_values():
    assert abs(float(Q.aspect_ratio(*EQUI)[0]) - 2.0 / math.sqrt(3)) < 1e-12
    assert float(Q.aspect_ratio(*SLIVER)[0]) > 100.0


def test_area_of_reference_triangles():
    assert abs(float(Q.triangle_areas(*EQUI)[0]) - math.sqrt(3) / 4) < 1e-12
    assert abs(float(Q.triangle_areas(*RIGHT_ISO)[0]) - 0.5) < 1e-12


def test_topology_checks_on_an_open_patch():
    """An open patch is NOT watertight, and must say so."""
    m = _mesh(PLANE, [(0, 0), (1, 0), (1, 1), (0, 1)], 0.3)
    topo = Q.topology_checks(m)
    assert topo["is_watertight"] is False
    assert topo["n_boundary_edges"] > 0
    assert topo["n_nonmanifold_edges"] == 0
    assert topo["n_inconsistent_normals"] == 0


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
