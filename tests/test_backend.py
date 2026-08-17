"""
Size field and backend-contract tests. PURE -- no gmsh.

Covers the two pieces of the meshing backend that contain real logic rather than
API plumbing: resolving a declarative spec into per-face sizes, and attributing
whole-mesh quality back to individual CAD faces.

The gmsh adapter itself is verified separately by tools/verify_backend.py against a
real kernel, because nothing here can substitute for OCC behaviour.
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backends.base import (FaceScore, MeshProvenance, MeshRequest, MeshStats,
                           MeshingStrategy, attribute_by_face, submesh,
                           worst_faces)
from pipeline import sizefield as SF
from pipeline.sizefield import (RuleKind, SizeFieldSpec, estimate_element_count,
                                ramp_distance, size_from_curvature, spec_from_dict)
from quality.core import SurfaceMesh


# --------------------------------------------------------------------------
# Curvature sizing
# --------------------------------------------------------------------------

def test_size_from_curvature_matches_closed_forms():
    R, delta, theta = 10.0, 0.01, 5.0
    h = size_from_curvature(R, chordal_tol=delta, normal_tol_deg=1e9,
                            min_per_2pi=0)
    assert abs(h - math.sqrt(8 * R * delta)) < 1e-12
    h = size_from_curvature(R, chordal_tol=1e9, normal_tol_deg=theta,
                            min_per_2pi=0)
    assert abs(h - 2 * R * math.sin(math.radians(theta))) < 1e-12
    assert math.isinf(size_from_curvature(0.0, 0.01, 5.0)), "planar -> unbounded"


def test_normal_bound_binds_at_coarse_sizes():
    """The reason both bounds exist.

    Chordal error is O(h^2/R), normal error O(h/R). At a loose chordal tolerance
    the normal bound is the smaller one, so a size field driven by sagitta alone
    systematically under-refines curved faces.
    """
    R = 10.0
    h_chord = size_from_curvature(R, chordal_tol=0.05, normal_tol_deg=1e9,
                                  min_per_2pi=0)
    h_normal = size_from_curvature(R, chordal_tol=1e9, normal_tol_deg=5.0,
                                   min_per_2pi=0)
    assert h_normal < h_chord, (h_chord, h_normal)
    assert abs(size_from_curvature(R, 0.05, 5.0, 0) - h_normal) < 1e-12


def test_elements_per_2pi_can_dominate():
    R = 1000.0    # very gentle curvature: both tolerance bounds are loose
    h = size_from_curvature(R, chordal_tol=100.0, normal_tol_deg=45.0,
                            min_per_2pi=12.0)
    assert abs(h - 2 * math.pi * R / 12.0) < 1e-9, h


def test_huge_normal_tolerance_does_not_invert_the_bound():
    """REGRESSION: sin() past 90 degrees turns around and goes negative.

    A large value passed to mean "ignore this term" became the tightest constraint
    and produced a NEGATIVE requested element size (-19.7 on R=10), which would have
    been handed straight to the mesher.
    """
    R = 10.0
    for deg in (90.0, 180.0, 1e9):
        h = size_from_curvature(R, chordal_tol=math.inf, normal_tol_deg=deg,
                                min_per_2pi=0)
        assert h > 0, (deg, h)
        assert abs(h - 2 * R) < 1e-9, (deg, h)
    assert size_from_curvature(R, math.inf, 1e9, 0) > 0


# --------------------------------------------------------------------------
# Spec resolution
# --------------------------------------------------------------------------

def _faces(**kw):
    """PID -> sizing inputs."""
    return {pid: {"curvature_max": c, "thickness": t}
            for pid, (c, t) in kw.items()}


def test_base_size_applies_to_every_face():
    spec = SizeFieldSpec([SF.base(2.0)])
    sizes = spec.resolve(_faces(a=(0.0, None), b=(0.0, None)))
    assert sizes == {"a": 2.0, "b": 2.0}


def test_every_rule_can_only_shrink():
    """Resolution order-independence depends on this."""
    spec = SizeFieldSpec([SF.base(2.0),
                          SF.curvature(chordal_tol=0.01, normal_tol_deg=5.0),
                          SF.thickness(3), SF.feature(0.5, scope=("c",))])
    sizes = spec.resolve(_faces(a=(0.0, None), b=(0.5, None), c=(0.0, 6.0)))
    assert sizes["a"] == 2.0, "planar, no thickness -> base"
    assert sizes["b"] < 2.0, "curved -> shrunk"
    assert sizes["c"] == 0.5, "feature rule wins over thickness 6/3=2.0"


def test_curvature_rule_respects_scope():
    spec = SizeFieldSpec([SF.base(2.0),
                          SF.curvature(0.01, 5.0, scope=("b",))])
    sizes = spec.resolve(_faces(a=(0.5, None), b=(0.5, None)))
    assert sizes["a"] == 2.0, "out of scope must be untouched"
    assert sizes["b"] < 2.0


def test_thickness_rule_gives_n_elements_through():
    spec = SizeFieldSpec([SF.base(10.0), SF.thickness(3)])
    sizes = spec.resolve(_faces(a=(0.0, 1.5)))
    assert abs(sizes["a"] - 0.5) < 1e-12, "1.5mm wall / 3 = 0.5"


def test_clamp_bounds_both_ends():
    spec = SizeFieldSpec([SF.base(10.0),
                          SF.curvature(1e-6, 0.01),      # would demand ~0
                          SF.clamp(h_min=0.1, h_max=5.0)])
    sizes = spec.resolve(_faces(a=(1.0, None), b=(0.0, None)))
    assert sizes["a"] == 0.1, "floor stops curvature exploding element count"
    assert sizes["b"] == 5.0, "ceiling caps the base"


def test_gradation_smoothing_limits_neighbour_ratio():
    """A small fillet beside a large planar face asks for a huge jump."""
    sizes = {"small": 0.1, "mid": 10.0, "big": 10.0}
    adj = {"small": {"mid"}, "mid": {"small", "big"}, "big": {"mid"}}
    out = SF.smooth_sizes(sizes, adj, max_growth=2.0)
    assert out["small"] == 0.1, "the driver must not grow"
    assert out["mid"] <= 0.1 * 2.0 + 1e-12
    assert out["big"] <= out["mid"] * 2.0 + 1e-12
    for pid, nbs in adj.items():
        for nb in nbs:
            assert max(out[pid], out[nb]) / min(out[pid], out[nb]) <= 2.0 + 1e-9


def test_gradation_smoothing_terminates_and_only_shrinks():
    rng = np.random.default_rng(0)
    pids = [f"p{i}" for i in range(30)]
    sizes = {p: float(rng.uniform(0.05, 20.0)) for p in pids}
    adj = {p: {q for q in pids if q != p} for p in pids}      # fully connected
    out = SF.smooth_sizes(sizes, adj, max_growth=1.2)
    assert all(out[p] <= sizes[p] + 1e-12 for p in pids), "must be monotone"
    lo, hi = min(out.values()), max(out.values())
    assert hi / lo <= 1.2 + 1e-9


def test_resolve_applies_gradation_when_adjacency_given():
    spec = SizeFieldSpec([SF.base(10.0), SF.feature(0.1, scope=("a",)),
                          SF.gradation(2.0)])
    faces = _faces(a=(0.0, None), b=(0.0, None))
    adj = {"a": {"b"}, "b": {"a"}}
    loose = spec.resolve(faces)
    tight = spec.resolve(faces, adj)
    assert loose["b"] == 10.0, "no adjacency -> no smoothing"
    assert tight["b"] <= 0.2 + 1e-12


# --------------------------------------------------------------------------
# Spec identity and diffing -- what the loop depends on
# --------------------------------------------------------------------------

def test_fingerprint_is_stable_and_sensitive():
    a = SizeFieldSpec.default(100.0)
    b = SizeFieldSpec.default(100.0)
    assert a.fingerprint() == b.fingerprint()
    c = a.replace_kind(RuleKind.BASE, SF.base(1.0))
    assert c.fingerprint() != a.fingerprint(), "the oscillation guard needs this"


def test_replace_kind_swaps_not_appends():
    a = SizeFieldSpec.default(100.0)
    n0 = len(a.rules)
    b = a.replace_kind(RuleKind.BASE, SF.base(1.23))
    assert len(b.rules) == n0
    assert b.get(RuleKind.BASE).params["h"] == 1.23
    assert a.get(RuleKind.BASE).params["h"] != 1.23, "must not mutate the original"


def test_replace_kind_appends_when_scope_differs():
    a = SizeFieldSpec([SF.base(2.0)])
    b = a.replace_kind(RuleKind.CURVATURE, SF.curvature(0.01, 5.0, scope=("x",)))
    assert len(b.rules) == 2


def test_diff_describes_what_changed():
    a = SizeFieldSpec.default(100.0)
    b = a.replace_kind(RuleKind.BASE, SF.base(1.0))
    d = b.diff(a) if False else a.diff(b)
    assert any("base" in line and "~" in line for line in d), d
    c = a.with_rule(SF.feature(0.1, scope=("p1", "p2")))
    d2 = a.diff(c)
    assert any(line.startswith("+") and "2 entities" in line for line in d2), d2


def test_spec_roundtrips_through_json():
    a = SizeFieldSpec.default(76.3)
    b = spec_from_dict(a.to_dict())
    assert b.fingerprint() == a.fingerprint()
    assert len(b.rules) == len(a.rules)


def test_default_spec_scales_with_the_model():
    small = SizeFieldSpec.default(10.0)
    big = SizeFieldSpec.default(1000.0)
    assert (big.get(RuleKind.BASE).params["h"]
            == 100 * small.get(RuleKind.BASE).params["h"])


# --------------------------------------------------------------------------
# Request identity
# --------------------------------------------------------------------------

def test_request_fingerprint_covers_strategy_and_size():
    spec = SizeFieldSpec.default(100.0)
    r1 = MeshRequest(spec, MeshingStrategy(algorithm="frontal"))
    r2 = MeshRequest(spec, MeshingStrategy(algorithm="frontal"))
    r3 = MeshRequest(spec, MeshingStrategy(algorithm="delaunay"))
    r4 = MeshRequest(spec.replace_kind(RuleKind.BASE, SF.base(1.0)),
                     MeshingStrategy(algorithm="frontal"))
    assert r1.fingerprint() == r2.fingerprint()
    assert len({r1.fingerprint(), r3.fingerprint(), r4.fingerprint()}) == 3


def test_imprint_defaults_on():
    """Off by default would silently produce disconnected assemblies."""
    assert MeshingStrategy().imprint is True
    assert MeshingStrategy().heal is False, "healing is remedial, not default"


# --------------------------------------------------------------------------
# Per-face attribution
# --------------------------------------------------------------------------

def _two_face_mesh():
    """Two unit squares side by side, each split into 2 triangles."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [2, 0, 0], [2, 1, 0]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    mesh = SurfaceMesh(vertices=v, triangles=tris, method="test")
    prov = MeshProvenance(tri_pid=["faceA", "faceA", "faceB", "faceB"],
                          requested_size={"faceA": 1.0, "faceB": 1.0})
    return mesh, prov


def test_submesh_reindexes_correctly():
    mesh, prov = _two_face_mesh()
    sm = submesh(mesh, prov.indices_for("faceB"))
    assert sm.n_triangles == 2
    assert sm.n_vertices == 4, "only the vertices faceB actually uses"
    assert sm.triangles.max() == 3 and sm.triangles.min() == 0
    from quality import core as Q
    a, b, c = sm.corners()
    assert abs(float(Q.triangle_areas(a, b, c).sum()) - 1.0) < 1e-12


def test_attribution_scores_each_face_separately():
    mesh, prov = _two_face_mesh()
    scores = attribute_by_face(mesh, prov)
    assert {s.pid for s in scores} == {"faceA", "faceB"}
    assert all(s.n_triangles == 2 for s in scores)
    assert all(s.shape_min > 0.5 for s in scores)


def test_attribution_is_sorted_worst_first():
    """The policy reads the head of this list."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [2, 0, 0], [2, 0.001, 0]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5]])
    mesh = SurfaceMesh(vertices=v, triangles=tris, method="test")
    prov = MeshProvenance(tri_pid=["good", "good", "sliver"])
    scores = attribute_by_face(mesh, prov)
    assert scores[0].pid == "sliver", [s.pid for s in scores]
    assert scores[0].shape_min < 0.05


def test_worst_faces_direction_is_metric_aware():
    scores = [FaceScore("a", 10, 0.9, 0.9, 50, chordal_relative=0.01),
              FaceScore("b", 10, 0.2, 0.5, 12, chordal_relative=0.30),
              FaceScore("c", 10, 0.6, 0.7, 30, chordal_relative=0.05)]
    assert worst_faces(scores, "shape_min", 2) == ["b", "c"], "low shape is bad"
    assert worst_faces(scores, "chordal_relative", 2) == ["b", "c"], "high is bad"


def test_worst_faces_ignores_unmeasured_metrics():
    scores = [FaceScore("a", 5, 0.9, 0.9, 50), FaceScore("b", 5, 0.2, 0.5, 12)]
    assert worst_faces(scores, "chordal_relative") == [], "all NaN -> empty"
    assert worst_faces(scores, "shape_min", 1) == ["b"]


def test_provenance_size_callable_scores_against_the_request():
    """Size adequation must compare against what was ASKED for per face.

    A correctly curvature-refined face is small on purpose; scoring it against a
    global nominal would report it as an oversized-element failure.
    """
    mesh, prov = _two_face_mesh()
    prov.requested_size = {"faceA": 1.0, "faceB": 0.25}
    f = prov.size_field_callable()
    assert f is not None
    vals = f(None)
    assert list(vals) == [1.0, 1.0, 0.25, 0.25]
    assert MeshProvenance().size_field_callable() is None


# --------------------------------------------------------------------------
# Assembly interfaces -- topology on the exterior skin
# --------------------------------------------------------------------------

def _assembly_mesh():
    """Two squares side by side plus an internal wall strip on the shared edge."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [2, 0, 0], [2, 1, 0], [1, 0, 1], [1, 1, 1]], float)
    t = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2],
                  [1, 6, 7], [1, 7, 2]])
    mesh = SurfaceMesh(vertices=v, triangles=t, method="assembly")
    prov = MeshProvenance(
        tri_pid=["A", "A", "B", "B", "IFACE", "IFACE"],
        requested_size={"A": 1.0, "B": 1.0, "IFACE": 1.0},
        interface_pids=("IFACE",))
    return mesh, prov


def test_interface_triangles_are_identified_from_provenance():
    _, prov = _assembly_mesh()
    assert prov.interface_triangles() == {4, 5}
    assert MeshProvenance(tri_pid=["A"]).interface_triangles() == set()


def test_topology_excludes_interfaces_from_manifold_count():
    """REGRESSION, from real two_blocks.step.

    After fragment() the shared face is a real face INSIDE the material, so every
    edge bounding it touches three faces and three triangles meet there. The mesh
    is correct; a blanket manifoldness rule reported 30 non-manifold edges and
    failed the validity gate.
    """
    from quality.core import topology_checks
    mesh, prov = _assembly_mesh()
    naive = topology_checks(mesh)
    skin = topology_checks(mesh, prov.interface_triangles())
    assert naive["n_nonmanifold_edges"] > 0, "the blanket rule must flag it"
    assert skin["n_nonmanifold_edges"] == 0, "the skin is manifold"
    assert skin["n_interface_triangles"] == 2


def test_verdict_passes_with_interface_exclusion():
    from quality.criteria import evaluate as assess
    mesh, prov = _assembly_mesh()
    bad = assess(mesh, target_size=1.0)
    good = assess(mesh, target_size=1.0,
                  interface_triangles=prov.interface_triangles())
    assert not bad.is_valid
    assert "validity.nonmanifold" in {f.code for f in bad.failures}
    assert good.is_valid, [f.code for f in good.failures]


def test_thickness_rule_is_inert_without_data():
    """Why thin_plate failed: the rule exists but had no data source.

    100 x 1.5 side faces sized at the 7.07 base gave 12.7 degree minimum angles;
    1.5/3 = 0.5 was wanted.
    """
    spec = SizeFieldSpec([SF.base(7.07), SF.thickness(3)])
    inert = spec.resolve({"side": {"curvature_max": 0.0, "thickness": None}})
    fed = spec.resolve({"side": {"curvature_max": 0.0, "thickness": 1.5}})
    assert inert["side"] == 7.07, "no data -> the rule does nothing, silently"
    assert abs(fed["side"] - 0.5) < 1e-12


# --------------------------------------------------------------------------
# Gradation is a SPATIAL limit
# --------------------------------------------------------------------------

def _sliver_block_like():
    sizes = {"rail": 0.0768, "top": 3.841, "side": 3.841, "bottom": 3.841}
    areas = {"rail": 0.25, "top": 2500.0, "side": 1500.0, "bottom": 2500.0}
    adj = {"rail": {"top"}, "top": {"rail", "side"},
           "side": {"top", "bottom"}, "bottom": {"side"}}
    extents = {k: math.sqrt(v) for k, v in areas.items()}
    return sizes, areas, adj, extents


def test_topological_gradation_propagates_a_tiny_size_everywhere():
    """REGRESSION, from real sliver_block.step.

    Capping each neighbour at h * growth treats one hop as one factor -- but a hop
    can cross a 50 mm face. A 0.005 mm rail forced a 0.077 mm size that walked
    across the whole model and produced 1,668,436 triangles in 16 s, up from ~1900.
    """
    sizes, areas, adj, _ = _sliver_block_like()
    naive = SF.smooth_sizes(sizes, adj, 1.4)
    assert naive["bottom"] < 0.3, "the far face is dragged down: that is the bug"
    assert estimate_element_count(naive, areas) > 500_000


def test_extent_aware_gradation_keeps_large_faces_coarse():
    """A face of length L carrying size h can grow internally by growth**(L/h)."""
    sizes, areas, adj, extents = _sliver_block_like()
    aware = SF.smooth_sizes(sizes, adj, 1.4, extents=extents)
    assert aware["rail"] == 0.0768, "the driver must not grow"
    assert aware["top"] < sizes["top"], "the immediate neighbour still responds"
    assert aware["bottom"] == 3.841, "a face two hops away is unaffected"
    assert estimate_element_count(aware, areas) < 50_000


def test_extent_aware_smoothing_still_only_shrinks():
    sizes, areas, adj, extents = _sliver_block_like()
    aware = SF.smooth_sizes(sizes, adj, 1.4, extents=extents)
    assert all(aware[k] <= sizes[k] + 1e-12 for k in sizes)


def test_resolve_uses_face_area_as_extent():
    """resolve() must derive extents from face_info, not ignore them."""
    spec = SizeFieldSpec([SF.base(4.0), SF.feature(0.05, scope=("tiny",)),
                          SF.gradation(1.4)])
    faces = {"tiny": {"curvature_max": 0.0, "thickness": None, "area": 0.25},
             "huge": {"curvature_max": 0.0, "thickness": None, "area": 2500.0}}
    adj = {"tiny": {"huge"}, "huge": {"tiny"}}
    out = spec.resolve(faces, adj)
    assert out["tiny"] == 0.05
    assert out["huge"] > 0.05 * 1.4, "extent must let the big face stay coarser"


def test_element_estimate_matches_the_ideal_formula():
    """area / (sqrt(3)/4 h^2) -- the same ideal count used throughout."""
    n = estimate_element_count({"f": 0.5}, {"f": 100.0})
    assert abs(n - 100.0 / (math.sqrt(3) / 4 * 0.25)) < 1.0
    assert estimate_element_count({"f": 0.0}, {"f": 100.0}) == 0
    assert estimate_element_count({"f": float("inf")}, {"f": 100.0}) == 0


def test_element_estimate_flags_a_runaway_before_meshing():
    """The guard exists so a two-million-element request is a DECISION, not a
    discovery made after 16 seconds of meshing."""
    sizes, areas, adj, extents = _sliver_block_like()
    naive = SF.smooth_sizes(sizes, adj, 1.4)
    aware = SF.smooth_sizes(sizes, adj, 1.4, extents=extents)
    budget = 200_000
    assert estimate_element_count(naive, areas) > budget
    assert estimate_element_count(aware, areas) < budget


# --------------------------------------------------------------------------
# Spatial ramps -- what makes the gradation objective achievable
# --------------------------------------------------------------------------

def test_ramp_distance_matches_the_geometric_series():
    """n elements growing by g from h cover h*(g^n - 1)/(g - 1)."""
    h, g = 1.56, 1.4
    far = 3.82
    d = ramp_distance(h, far, g)
    n = math.log(far / h) / math.log(g)
    assert abs(d - h * (g ** n - 1) / (g - 1)) < 1e-9
    assert abs(d - (far - h) / (g - 1)) < 1e-12


def test_ramp_distance_degenerate_cases():
    assert ramp_distance(3.0, 1.0, 1.4) == 0.0, "already coarser than the target"
    assert ramp_distance(1.0, 2.0, 1.0) == 0.0, "no growth allowed"
    assert ramp_distance(0.0, 2.0, 1.4) == 0.0


def test_gentler_growth_needs_a_longer_ramp():
    """The physical statement of the gradation limit."""
    a = ramp_distance(0.1, 10.0, 1.2)
    b = ramp_distance(0.1, 10.0, 2.0)
    assert a > b, (a, b)
    assert abs(a / b - (2.0 - 1.0) / (1.2 - 1.0)) < 1e-9


def test_resolve_can_skip_smoothing_for_ramping_backends():
    """A backend that ramps spatially must not also be pre-smoothed.

    Pre-smoothing coarsens the fine faces to buy a gradation the ramp would have
    provided for free, which is the trade that regressed block_hole from
    shp_min 0.767 to 0.485.
    """
    spec = SizeFieldSpec([SF.base(4.0), SF.feature(0.05, scope=("tiny",)),
                          SF.gradation(1.4)])
    faces = {"tiny": {"curvature_max": 0.0, "thickness": None, "area": 0.25},
             "huge": {"curvature_max": 0.0, "thickness": None, "area": 2500.0}}
    adj = {"tiny": {"huge"}, "huge": {"tiny"}}
    smoothed = spec.resolve(faces, adj, smooth=True)
    raw = spec.resolve(faces, adj, smooth=False)
    assert raw["huge"] == 4.0, "unsmoothed keeps the coarse face coarse"
    assert smoothed["huge"] < raw["huge"]
    assert raw["tiny"] == smoothed["tiny"] == 0.05, "the driver is unchanged"


def test_ramp_covers_the_face_size_jump_within_the_model():
    """A ramp must be long enough to matter and short enough to fit."""
    scale = 76.32
    for h_near, h_far in ((1.56, 3.82), (0.077, 3.84), (0.5, 7.07)):
        d = ramp_distance(h_near, h_far, 1.4)
        assert d > h_near, (h_near, d)
        assert d < 0.5 * scale, (h_near, d)


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
