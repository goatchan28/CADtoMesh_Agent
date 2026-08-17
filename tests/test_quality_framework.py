"""
Tests for the completed quality framework: size adequation, gradation, CAD normal
deviation, and the validity-gate / objective-score split.

No gmsh. Analytic surfaces and hand-built meshes with known answers.

The normal-deviation tests matter most, because that metric has a closed-form
reference (facet_angle_bound) AND it is the one that governs area convergence --
the Schwarz lantern failure is a normal-convergence failure, not a distance one.
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import quality as QU
from quality import core as Q
from quality.criteria import Criteria, evaluate as assess
from research.boundary import boundary_from_uv_polygon, resample_uv_polygon
from research.cdt_parametric import triangulate
from research.surface import CylinderSurface, PlaneSurface, SphereSurface

PLANE = PlaneSurface()
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]
CYL_PATCH = [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)]


def _unit_square_mesh():
    return Q.SurfaceMesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),
        triangles=np.array([[0, 1, 2], [0, 2, 3]]), method="test",
        meta={"target_size": 1.0})


def _mesh(surface, poly, target):
    fb = boundary_from_uv_polygon(
        [resample_uv_polygon(poly, surface, target)], surface, target)
    return triangulate(fb, surface).mesh


# --------------------------------------------------------------------------
# Size adequation
# --------------------------------------------------------------------------

def test_element_size_is_rms_edge_length():
    """RMS, so that an equilateral triangle of edge h reports exactly h."""
    h = 2.0
    a = np.array([[0, 0, 0]], float)
    b = np.array([[h, 0, 0]], float)
    c = np.array([[h / 2, h * math.sqrt(3) / 2, 0]], float)
    assert abs(float(QU.element_size(a, b, c)[0]) - h) < 1e-12
    # And size_from_area agrees for an equilateral element.
    assert abs(float(QU.size_from_area(a, b, c)[0]) - h) < 1e-12


def test_size_adequation_detects_correct_and_wrong_size():
    m = _unit_square_mesh()
    # Edges are 1, 1, sqrt(2): RMS = sqrt((1+1+2)/3) = 1.1547
    exact = math.sqrt(4.0 / 3.0)
    r = QU.size_adequation(m, exact)
    assert abs(r.ratio_mean - 1.0) < 1e-9, r.ratio_mean
    assert r.frac_in_band == 1.0
    # Ask for elements 10x smaller: everything is oversized.
    r2 = QU.size_adequation(m, exact / 10.0)
    assert r2.frac_oversized == 1.0
    assert abs(r2.ratio_mean - 10.0) < 1e-9


def test_size_adequation_is_symmetric():
    """Over-refinement must register too.

    Our parametric CDT produced 1.3-1.6x the ideal element count while every shape
    metric looked excellent. A one-sided metric would have called that perfect.
    """
    m = _unit_square_mesh()
    exact = math.sqrt(4.0 / 3.0)
    over = QU.size_adequation(m, exact * 4.0)      # elements 4x too SMALL vs target
    assert over.frac_undersized == 1.0, over
    assert over.frac_oversized == 0.0


def test_size_adequation_accepts_a_size_field():
    """A callable target is what lets this score curvature-adaptive sizing."""
    m = _mesh(PLANE, SQUARE, 0.25)

    def field(centroids):
        # Ask for finer elements toward +x.
        return 0.4 - 0.2 * centroids[:, 0]

    r = QU.size_adequation(m, field)
    assert r.target_mode == "field"
    uniform = QU.size_adequation(m, 0.25)
    assert uniform.target_mode == "uniform"
    assert r.n_elements == uniform.n_elements == m.n_triangles


def test_shape_quality_is_blind_to_size():
    """The reason size adequation must exist as a separate metric."""
    small = _unit_square_mesh()
    big = Q.SurfaceMesh(vertices=small.vertices * 1000.0,
                        triangles=small.triangles, method="scaled")
    assert abs(Q.evaluate(small).shape_min - Q.evaluate(big).shape_min) < 1e-12
    s1 = QU.size_adequation(small, 1.0).ratio_mean
    s2 = QU.size_adequation(big, 1.0).ratio_mean
    assert abs(s2 / s1 - 1000.0) < 1e-6, "size metric must see the difference"


# --------------------------------------------------------------------------
# Gradation
# --------------------------------------------------------------------------

def test_gradation_of_a_uniform_mesh_is_one():
    m = _unit_square_mesh()
    g = QU.gradation(m)
    assert g.n_pairs == 1, "two triangles share exactly one edge"
    assert abs(g.max_ratio - 1.0) < 1e-12
    assert g.frac_above_limit == 0.0


def test_gradation_detects_an_abrupt_size_jump():
    """One big triangle beside one small one: no per-element metric sees this."""
    v = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0],
                  [4.2, 0, 0], [4.2, 0.2, 0]], float)
    # Big triangles from the square, plus a tiny one sharing an edge.
    m = Q.SurfaceMesh(vertices=v, triangles=np.array([[0, 1, 2], [0, 2, 3]]),
                      method="jump")
    g = QU.gradation(m)
    assert abs(g.max_ratio - 1.0) < 1e-9, "these two are the same size"

    # Now a genuine jump: subdivide one triangle finely and leave the other coarse.
    fine = _mesh(PLANE, [(0, 0), (1, 0), (1, 1), (0, 1)], 0.2)
    coarse = _mesh(PLANE, [(0, 0), (1, 0), (1, 1), (0, 1)], 0.9)
    gf, gc = QU.gradation(fine), QU.gradation(coarse)
    assert gf.max_ratio < 3.0 and gc.max_ratio < 3.0, (gf.max_ratio, gc.max_ratio)


def test_gradation_uses_edge_adjacency_not_vertex():
    """Edge neighbours are where the solver needs interpolant continuity."""
    m = _mesh(PLANE, SQUARE, 0.3)
    g = QU.gradation(m)
    # An open patch: interior edges only, so pairs < total edges.
    total_edges = 3 * m.n_triangles
    assert 0 < g.n_pairs < total_edges
    assert g.max_ratio >= 1.0, "ratio is larger-over-smaller, always >= 1"


# --------------------------------------------------------------------------
# CAD normal deviation
# --------------------------------------------------------------------------

def test_normal_deviation_is_zero_on_a_plane():
    m = _mesh(PLANE, SQUARE, 0.25)
    nd = QU.normal_deviation(m, PLANE)
    assert nd.max_deg < 1e-9, nd.max_deg
    assert nd.area_weighted_mean_deg < 1e-9


def test_facet_angle_bound_closed_form():
    """theta/2 where theta = 2 asin(c/2R)."""
    R, c = 10.0, 2.0
    assert abs(QU.facet_angle_bound(R, c) - math.degrees(math.asin(0.1))) < 1e-12
    assert QU.facet_angle_bound(1.0, 5.0) == 90.0          # degenerate guard
    # Small-chord APPROXIMATION: angle ~ c/(2R) radians. This is an approximation
    # check, so the tolerance must be relative -- asin(x) and x differ at O(x^3),
    # which is 1.2e-6 degrees here and would fail an absolute 1e-6 bound.
    approx = math.degrees(1.0 / 200.0)
    exact = QU.facet_angle_bound(100.0, 1.0)
    assert abs(exact - approx) / exact < 1e-5, (exact, approx)


def test_normal_deviation_matches_closed_form_on_a_cylinder():
    R = 10.0
    cyl = CylinderSurface(radius=R)
    for t in (2.0, 1.0, 0.5):
        m = _mesh(cyl, CYL_PATCH, t)
        nd = QU.normal_deviation(m, cyl)
        maxL = float(Q.edge_lengths(*m.corners()).max())
        bound = QU.facet_angle_bound(R, maxL)
        assert nd.max_deg <= bound * 1.35 + 0.5, (t, nd.max_deg, bound)


def test_normal_deviation_scales_linearly_not_quadratically():
    """The key practical difference from chordal deviation.

    Normal error is O(c/R); chordal error is O(c^2/R). So halving element size
    HALVES normal deviation but QUARTERS chordal deviation, making normal deviation
    the binding constraint at coarse sizes and the slower one to improve. An
    adaptive loop that only watches chordal error will stop too early.
    """
    R = 10.0
    cyl = CylinderSurface(radius=R)
    nds, chs = [], []
    for t in (2.0, 1.0, 0.5):
        m = _mesh(cyl, CYL_PATCH, t)
        nds.append(QU.normal_deviation(m, cyl).p99_deg)
        chs.append(Q.chordal_deviation(m, cyl, target_size=t).max_deviation)
    n_ratio = nds[0] / nds[2]
    c_ratio = chs[0] / chs[2]
    assert 2.5 < n_ratio < 6.0, f"normal should fall ~4x over 4x refinement: {nds}"
    assert c_ratio > n_ratio * 1.5, (
        f"chordal must fall faster than normal: chordal {c_ratio:.1f}x vs "
        f"normal {n_ratio:.1f}x")


def test_normal_deviation_is_unsigned():
    """Must not report ~180 degrees for a globally flipped-but-perfect mesh.

    This project has already hit a surface whose normal() opposed
    unit(dX/du x dX/dv). Orientation consistency is topology_checks' job; this
    metric answers only "is the facet PARALLEL to the surface".
    """
    m = _mesh(PLANE, SQUARE, 0.3)
    flipped = Q.SurfaceMesh(vertices=m.vertices,
                            triangles=m.triangles[:, [0, 2, 1]], method="flipped")
    assert QU.normal_deviation(flipped, PLANE).max_deg < 1e-9


# --------------------------------------------------------------------------
# Validity gate vs objective score
# --------------------------------------------------------------------------

def test_valid_mesh_scores_and_is_ready():
    v = assess(_unit_square_mesh(), target_size=math.sqrt(4.0 / 3.0))
    assert v.is_valid
    assert v.all_objectives_satisfied
    assert v.ready
    assert v.score > 0.99


def test_inverted_element_is_a_hard_failure_not_a_low_score():
    """The central distinction. An invalid mesh must score 0, not 0.8."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    m = Q.SurfaceMesh(vertices=v,
                      triangles=np.array([[0, 1, 2], [0, 3, 2]]),  # opposing winding
                      method="inverted")
    verdict = assess(m, target_size=1.0)
    assert not verdict.is_valid
    assert verdict.score == 0.0, "invalid must not be comparable to valid on a scale"
    assert not verdict.ready
    codes = {f.code for f in verdict.failures}
    assert "validity.orientation" in codes, codes


def test_degenerate_shape_floor_is_validity_not_objective():
    """A 0.5-degree sliver is numerically degenerate, not merely poor quality."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-6, 0]], float)
    m = Q.SurfaceMesh(vertices=v, triangles=np.array([[0, 1, 2]]), method="sliver")
    verdict = assess(m, target_size=1.0)
    assert not verdict.is_valid
    assert "validity.degenerate_shape" in {f.code for f in verdict.failures}
    assert verdict.score == 0.0


def test_poor_but_usable_mesh_is_valid_with_unsatisfied_objectives():
    """The other side: valid, scored, and the loop knows what to work on."""
    h = math.sqrt(4.0 / 3.0)
    m = _unit_square_mesh()
    cr = Criteria(min_angle_deg=60.0, min_shape=0.99)     # unreachable targets
    verdict = assess(m, criteria=cr, target_size=h)
    assert verdict.is_valid, "still a perfectly usable mesh"
    assert not verdict.ready
    assert 0.0 < verdict.score < 1.0
    names = [o.name for o in verdict.unsatisfied()]
    assert "min_angle" in names and "min_shape" in names
    # unsatisfied() is ordered worst-attainment first -- the loop's work list.
    vals = [o.normalized for o in verdict.unsatisfied()]
    assert vals == sorted(vals)


def test_duplicate_nodes_are_a_hard_failure_when_checked():
    """The silent crack: coincident but distinct nodes look perfect."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                  [0, 0, 0], [1, 1, 0], [0, 1, 0]], float)   # 0/3 and 2/4 coincide
    m = Q.SurfaceMesh(vertices=v, triangles=np.array([[0, 1, 2], [3, 4, 5]]),
                      method="cracked")
    assert assess(m, criteria=Criteria()).is_valid, "off by default"
    verdict = assess(m, criteria=Criteria(max_duplicate_node_tolerance=1e-9))
    assert not verdict.is_valid
    assert "validity.duplicate_nodes" in {f.code for f in verdict.failures}


def test_watertight_requirement_is_opt_in():
    """A single trimmed face is legitimately open; a closed solid is not."""
    m = _mesh(PLANE, SQUARE, 0.3)
    assert assess(m, target_size=0.3).is_valid
    v = assess(m, criteria=Criteria(require_watertight=True), target_size=0.3)
    assert not v.is_valid
    assert "validity.not_watertight" in {f.code for f in v.failures}


def test_fidelity_objectives_appear_only_with_a_surface():
    m = _mesh(CylinderSurface(radius=10.0), CYL_PATCH, 1.0)
    without = {o.name for o in assess(m, target_size=1.0).objectives}
    with_s = {o.name for o in assess(m, target_size=1.0,
                                     surface=CylinderSurface(radius=10.0)).objectives}
    assert "chordal" not in without and "normal_deviation" not in without
    assert {"chordal", "normal_deviation"} <= with_s


def test_criteria_are_recorded_in_the_verdict():
    """A verdict that cannot state its own thresholds is not reproducible."""
    cr = Criteria(min_angle_deg=25.0)
    v = assess(_unit_square_mesh(), criteria=cr, target_size=1.0)
    assert v.criteria["min_angle_deg"] == 25.0
    assert "max_gradation" in v.criteria


def test_same_evaluator_scores_meshes_from_different_sources():
    """Backend independence, the reason quality/ has no gmsh import."""
    from research import af_direct3d as AF3
    from research import af_parametric as AFP
    cyl = CylinderSurface(radius=10.0)
    fb = boundary_from_uv_polygon(
        [resample_uv_polygon(CYL_PATCH, cyl, 1.0)], cyl, 1.0)
    meshes = [triangulate(fb, cyl).mesh, AFP.mesh_face(fb, cyl).mesh,
              AF3.mesh_face(fb, cyl).mesh]
    verdicts = [assess(m, target_size=1.0, surface=cyl) for m in meshes]
    assert all(v.is_valid for v in verdicts)
    names = {tuple(o.name for o in v.objectives) for v in verdicts}
    assert len(names) == 1, "every backend must be scored on the identical objectives"


def test_degenerate_floor_catches_sub_degree_elements():
    """REGRESSION: sliver_block.step passed the gate at shape 0.018.

    The floor answers "will the solve fail", not "is this good". A sub-degree
    element gives a near-singular element matrix however good the rest is.
    """
    from quality.criteria import Criteria
    cr = Criteria()
    assert cr.min_shape_floor >= 0.05, cr.min_shape_floor

    # A triangle with a ~1.7 degree apex angle -> shape ~0.017, which the old
    # 0.01 floor admitted.
    v = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.0075, 0]], float)
    m = Q.SurfaceMesh(vertices=v, triangles=np.array([[0, 1, 2]]), method="thin")
    s = float(Q.shape_quality(*m.corners())[0])
    assert 0.01 < s < 0.05, f"fixture must sit in the reopened gap: {s}"
    verdict = assess(m, target_size=1.0)
    assert not verdict.is_valid
    assert "validity.degenerate_shape" in {f.code for f in verdict.failures}


def test_floor_is_far_below_the_shape_objective():
    """Two thresholds on one metric, answering different questions."""
    from quality.criteria import Criteria
    cr = Criteria()
    assert cr.min_shape_floor < cr.min_shape / 4


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
