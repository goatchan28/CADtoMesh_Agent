"""
Surface mesh container and quality metrics. PURE -- no gmsh.

This module is the single arbiter of quality for all three meshers, and that is
the crux of a fair comparison. If gmsh's getElementQualities scored the two
parametric meshes while our own code scored the direct-3D one, differences in
normalization would be indistinguishable from differences in the algorithms. So
every mesh -- however produced -- is reduced to (vertices, triangles) and measured
by the same functions here.

gmsh's minSICN is still computed separately as a cross-check on the two gmsh
meshes. It is a different normalization, so expect correlation, not equality.

Metrics
-------
shape         4*sqrt(3)*A / (l1^2+l2^2+l3^2). 1.0 equilateral, ->0 degenerate.
              The primary FEM number: equivalent to a normalized inverse
              condition number for a linear triangle.
radius_ratio  2*r_in/r_circ. 1.0 equilateral. More sensitive to slivers.
min_angle     degrees. FEM interpolation error bounds depend on it.
max_angle     degrees. Large angles drive the gradient error, and matter more
              than small angles for FEM accuracy (Babuska-Aziz).
aspect        longest edge / shortest altitude. 1.15 equilateral, unbounded.

chordal       max distance from the flat facet to the true surface. This is the
              only metric here that measures GEOMETRIC FIDELITY rather than
              element shape -- see chordal_deviation() for why that distinction
              matters more than it looks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EQUILATERAL_ASPECT = 2.0 / np.sqrt(3.0)   # 1.1547


@dataclass
class SurfaceMesh:
    """Triangulated surface. The common output format of every mesher."""
    vertices: np.ndarray          # (N, 3) float
    triangles: np.ndarray         # (M, 3) int, CCW w.r.t. outward normal
    face_tag: np.ndarray | None = None   # (M,) int, CAD face each triangle came from
    method: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.triangles = np.asarray(self.triangles, dtype=np.int64).reshape(-1, 3)
        if self.face_tag is not None:
            self.face_tag = np.asarray(self.face_tag, dtype=np.int64).ravel()

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def corners(self):
        """(a, b, c) vertex-coordinate arrays, each (M, 3)."""
        t = self.triangles
        v = self.vertices
        return v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]


# ---------------------------------------------------------------------------
# Per-triangle geometry
# ---------------------------------------------------------------------------

def triangle_areas(a, b, c) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def triangle_normals(a, b, c) -> np.ndarray:
    n = np.cross(b - a, c - a)
    L = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, L, out=np.zeros_like(n), where=L > 0)


def edge_lengths(a, b, c) -> np.ndarray:
    """(M, 3) lengths of edges (a-b, b-c, c-a)."""
    return np.stack([
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(a - c, axis=1),
    ], axis=1)


def shape_quality(a, b, c) -> np.ndarray:
    """4*sqrt(3)*A / sum(l^2). 1.0 for equilateral, 0.0 for degenerate."""
    A = triangle_areas(a, b, c)
    L2 = np.sum(edge_lengths(a, b, c) ** 2, axis=1)
    return np.divide(4.0 * np.sqrt(3.0) * A, L2,
                     out=np.zeros_like(A), where=L2 > 0)


def radius_ratio(a, b, c) -> np.ndarray:
    """2*r_in/r_circ. 1.0 equilateral. r_in = A/s, r_circ = abc/(4A)."""
    L = edge_lengths(a, b, c)
    A = triangle_areas(a, b, c)
    s = 0.5 * np.sum(L, axis=1)
    prod = np.prod(L, axis=1)
    # 2 * (A/s) / (prod/(4A)) = 8 A^2 / (s * prod)
    denom = s * prod
    return np.divide(8.0 * A * A, denom, out=np.zeros_like(A), where=denom > 0)


def angles_deg(a, b, c) -> np.ndarray:
    """(M, 3) interior angles in degrees, at vertices a, b, c."""
    L = edge_lengths(a, b, c)          # ab, bc, ca
    ab, bc, ca = L[:, 0], L[:, 1], L[:, 2]

    def _ang(opp, s1, s2):
        denom = 2.0 * s1 * s2
        cosv = np.divide(s1 * s1 + s2 * s2 - opp * opp, denom,
                         out=np.zeros_like(opp), where=denom > 0)
        return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

    return np.stack([_ang(bc, ab, ca), _ang(ca, ab, bc), _ang(ab, bc, ca)], axis=1)


def aspect_ratio(a, b, c) -> np.ndarray:
    """Longest edge / shortest altitude. 1.1547 equilateral, grows without bound."""
    L = edge_lengths(a, b, c)
    A = triangle_areas(a, b, c)
    longest = L.max(axis=1)
    # altitude to edge i is 2A/L_i; the shortest corresponds to the LONGEST edge
    alt_min = np.divide(2.0 * A, longest, out=np.zeros_like(A), where=longest > 0)
    return np.divide(longest, alt_min, out=np.full_like(A, np.inf),
                     where=alt_min > 0)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    method: str
    n_vertices: int
    n_triangles: int
    total_area: float
    shape_min: float
    shape_p01: float
    shape_mean: float
    radius_ratio_min: float
    min_angle: float          # global minimum interior angle, degrees
    max_angle: float          # global maximum interior angle, degrees
    angle_p01: float          # 1st percentile of per-triangle minimum angle
    aspect_max: float
    n_inverted: int           # zero or negative area
    n_below_shape_02: int     # count with shape < 0.2 -- unusable for FEM
    worst_triangles: list = field(default_factory=list)
    # topology
    is_watertight: bool | None = None
    n_boundary_edges: int | None = None
    n_nonmanifold_edges: int | None = None
    n_inconsistent_normals: int | None = None
    n_interface_triangles: int = 0
    boundary_length: float = 0.0
    """Triangles excluded from the topology checks as internal assembly walls.
    Present because evaluate() splats topology_checks()'s dict straight into this
    report, so the two must stay in step."""
    # optional extras filled in by callers
    max_deviation: float | None = None
    mean_deviation: float | None = None
    seconds: float | None = None
    gmsh_min_sicn: float | None = None
    notes: list = field(default_factory=list)


def evaluate(mesh: SurfaceMesh, n_worst: int = 20) -> QualityReport:
    """Score a mesh. Same code path for every mesher, by design."""
    a, b, c = mesh.corners()
    if mesh.n_triangles == 0:
        return QualityReport(
            method=mesh.method, n_vertices=mesh.n_vertices, n_triangles=0,
            total_area=0.0, shape_min=0.0, shape_p01=0.0, shape_mean=0.0,
            radius_ratio_min=0.0, min_angle=0.0, max_angle=0.0, angle_p01=0.0,
            aspect_max=float("inf"), n_inverted=0, n_below_shape_02=0,
            notes=["empty mesh"])

    q = shape_quality(a, b, c)
    rr = radius_ratio(a, b, c)
    ang = angles_deg(a, b, c)
    asp = aspect_ratio(a, b, c)
    area = triangle_areas(a, b, c)

    order = np.argsort(q)[:n_worst]
    worst = [{"tri": int(i), "shape": float(q[i]),
              "min_angle": float(ang[i].min()), "max_angle": float(ang[i].max()),
              "face_tag": int(mesh.face_tag[i]) if mesh.face_tag is not None else None}
             for i in order]

    topo = topology_checks(mesh)
    return QualityReport(
        method=mesh.method,
        n_vertices=mesh.n_vertices,
        n_triangles=mesh.n_triangles,
        total_area=float(area.sum()),
        shape_min=float(q.min()),
        shape_p01=float(np.percentile(q, 1.0)),
        shape_mean=float(q.mean()),
        radius_ratio_min=float(rr.min()),
        min_angle=float(ang.min()),
        max_angle=float(ang.max()),
        angle_p01=float(np.percentile(ang.min(axis=1), 1.0)),
        aspect_max=float(asp.max()),
        n_inverted=int(np.sum(area <= 0.0)),
        n_below_shape_02=int(np.sum(q < 0.2)),
        worst_triangles=worst,
        **topo,
    )


def topology_checks(mesh: SurfaceMesh,
                    interface_triangles=None) -> dict:
    """Watertightness, manifoldness, normal consistency. Pure.

    A CAD-derived surface mesh should be closed: every edge shared by exactly two
    triangles, and the two traversals of each shared edge in OPPOSITE directions
    (consistent orientation).

    interface_triangles: indices of triangles lying on INTERNAL faces of an
    imprinted assembly. When given, all checks run on the EXTERIOR SKIN with those
    triangles excluded.

    Without this, a correctly imprinted assembly fails. After fragment(), the
    shared face between two solids is a real face inside the material; every edge
    bounding it touches three faces (the interface plus one from each body), so
    three triangles meet there. Measured on two_blocks.step: 30 "non-manifold"
    edges and not watertight, on a mesh that is exactly right.

    This is the same distinction Stage 0 had to make between a seam edge and a free
    edge -- count incidences against known structure rather than applying a blanket
    rule. Excluding interface triangles from the COUNT (while leaving them in the
    mesh, since volume meshing needs them) restores 2 per edge on the skin.
    """
    t = mesh.triangles
    if interface_triangles:
        keep = np.ones(len(t), dtype=bool)
        keep[np.asarray(sorted(interface_triangles), dtype=np.int64)] = False
        t = t[keep]
    if len(t) == 0:
        return {"is_watertight": False, "n_boundary_edges": 0,
                "n_nonmanifold_edges": 0, "n_inconsistent_normals": 0,
                "n_interface_triangles": len(interface_triangles or ()),
                "boundary_length": 0.0}

    directed = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=0)
    undirected = np.sort(directed, axis=1)
    _, inv, counts = np.unique(undirected, axis=0, return_inverse=True,
                               return_counts=True)
    per_edge = counts[inv]

    n_boundary = int(np.sum(counts == 1))
    n_nonmanifold = int(np.sum(counts > 2))

    # Physical length of the open boundary. The COUNT depends on element size; the
    # length does not, so only the length can be compared against a CAD expectation.
    boundary_length = 0.0
    if n_boundary and len(mesh.vertices):
        once = np.where(counts == 1)[0]
        pos = {int(e): i for i, e in enumerate(np.unique(inv))}
        sel = np.isin(inv, once)
        seg = undirected[sel]
        if len(seg):
            d = mesh.vertices[seg[:, 0]] - mesh.vertices[seg[:, 1]]
            boundary_length = float(np.linalg.norm(d, axis=1).sum())

    # Orientation: for each edge used exactly twice, the two directed uses must
    # differ. If they match, the two triangles disagree on which way is out.
    inconsistent = 0
    twice = np.where(counts == 2)[0]
    if len(twice):
        buckets: dict[int, list[int]] = {}
        for i, e in enumerate(inv):
            if per_edge[i] == 2:
                buckets.setdefault(int(e), []).append(i)
        for e, idxs in buckets.items():
            if len(idxs) != 2:
                continue
            d0, d1 = directed[idxs[0]], directed[idxs[1]]
            if d0[0] == d1[0] and d0[1] == d1[1]:
                inconsistent += 1

    return {
        "is_watertight": n_boundary == 0 and n_nonmanifold == 0,
        "n_boundary_edges": n_boundary,
        "n_nonmanifold_edges": n_nonmanifold,
        "n_inconsistent_normals": inconsistent,
        "n_interface_triangles": len(interface_triangles or ()),
        "boundary_length": boundary_length,
    }


# ---------------------------------------------------------------------------
# Chordal deviation -- geometric fidelity
# ---------------------------------------------------------------------------
#
# Every other metric in this module scores triangle SHAPE. None of them can tell
# whether the mesh represents the GEOMETRY. A beautifully equilateral mesh can cut
# a 5 mm chord straight across a fillet and be geometrically wrong; shape_quality
# reports 0.9 and is not lying, it is answering a different question.
#
# This is also the metric that the Schwarz lantern makes necessary. Total surface
# area is a tempting fidelity proxy and a bad one: inscribed triangulations of a
# curved surface can have MORE area than the surface, and can be refined so that
# area converges to the wrong value entirely, because area convergence needs the
# triangle NORMALS to converge, not just the vertices to lie on the surface.
# Chordal deviation has no such loophole -- it is a distance, and it goes to zero
# if and only if the mesh approaches the surface.
#
# Purity is preserved: this takes an object satisfying the Surface protocol
# (needs only .project), never gmsh. So it stays testable against analytic
# surfaces with closed-form answers.

# Barycentric sample points per triangle. Chordal error peaks at edge midpoints
# (where the chord sags furthest from its arc) and at the centroid (where the
# facet sags furthest from the patch), so both are sampled explicitly rather than
# relying on random sampling to find them.
DEFAULT_BARYCENTRIC = (
    (1 / 3, 1 / 3, 1 / 3),                                  # centroid
    (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5),      # edge midpoints
    (2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3),
)


@dataclass
class ChordalReport:
    max_deviation: float
    mean_deviation: float
    p99_deviation: float
    max_relative: float          # max_deviation / target_size, scale-free
    n_triangles_sampled: int
    n_samples: int
    worst_triangles: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"chordal: max {self.max_deviation:.4g} "
                f"(rel {self.max_relative:.4g}), mean {self.mean_deviation:.4g}, "
                f"p99 {self.p99_deviation:.4g}, "
                f"{self.n_samples} samples on {self.n_triangles_sampled} tris")


def chordal_deviation(mesh: SurfaceMesh, surface, *, target_size: float | None = None,
                      barycentric=DEFAULT_BARYCENTRIC, max_triangles: int | None = None,
                      seed: int = 0, n_worst: int = 20) -> ChordalReport:
    """Distance from each flat facet to the true surface.

    Args:
        surface: anything with .project(xyz) -> (closest_xyz, uv). The Surface
            protocol from research.surface; analytic implementations are exact.
        target_size: if given, max_relative = max_deviation / target_size. A
            chordal error of 0.1 mm means nothing without knowing whether the
            elements are 1 mm or 100 mm.
        max_triangles: subsample for speed. Each sample is one projection, and for
            a gmsh-backed surface that is an OCC call, so a 3000-triangle face at
            7 samples each is 21000 calls.

    Returns ChordalReport. On a mesh whose vertices lie on a PLANE the result is
    exactly zero, which is the test that the sampling and projection agree.
    """
    if mesh.n_triangles == 0:
        return ChordalReport(0.0, 0.0, 0.0, 0.0, 0, 0)

    a, b, c = mesh.corners()
    idx = np.arange(len(a))
    if max_triangles is not None and len(idx) > max_triangles:
        idx = np.random.default_rng(seed).choice(idx, max_triangles, replace=False)
        idx.sort()

    per_tri_max = np.zeros(len(idx))
    n_samples = 0
    for w0, w1, w2 in barycentric:
        pts = w0 * a[idx] + w1 * b[idx] + w2 * c[idx]
        proj = np.array([surface.project(p)[0] for p in pts], dtype=float)
        d = np.linalg.norm(proj - pts, axis=1)
        per_tri_max = np.maximum(per_tri_max, d)
        n_samples += len(pts)

    order = np.argsort(per_tri_max)[::-1][:n_worst]
    worst = [{"tri": int(idx[i]), "deviation": float(per_tri_max[i])}
             for i in order]

    mx = float(per_tri_max.max())
    return ChordalReport(
        max_deviation=mx,
        mean_deviation=float(per_tri_max.mean()),
        p99_deviation=float(np.percentile(per_tri_max, 99.0)),
        max_relative=(mx / target_size) if target_size else float("nan"),
        n_triangles_sampled=len(idx),
        n_samples=n_samples,
        worst_triangles=worst,
    )


def sagitta(radius: float, chord: float) -> float:
    """Exact sagitta of a chord on a circle of given radius: R - sqrt(R^2-(c/2)^2).

    The closed-form answer chordal_deviation must reproduce on a cylinder or
    sphere. Equals R(1-cos(theta/2)) for subtended angle theta = chord/R, and is
    approximately chord^2/(8R) for small chords -- which is the rule of thumb
    behind curvature-driven mesh sizing.
    """
    half = 0.5 * chord
    if half >= radius:
        return float(radius)
    return float(radius - math.sqrt(radius * radius - half * half))
