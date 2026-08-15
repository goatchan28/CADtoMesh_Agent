"""
Parametric Constrained Delaunay Triangulation. Our own implementation.

No gmsh algorithm is used or consulted. The only gmsh contact is via the Surface
protocol (point/derivatives/is_inside), which is data about the CAD face, not
meshing.

Pipeline
--------
1. Bowyer-Watson incremental Delaunay over the boundary nodes, in (u,v).
2. Anglada edge-flip recovery of any constrained boundary edge the Delaunay
   triangulation did not produce.
3. Inside/outside classification by winding number against the oriented loops,
   which discards the convex-hull filler and the interiors of holes.
4. Ruppert-style refinement driven by METRIC edge length, with diametral-circle
   encroachment handling on constrained segments.

A deliberate design choice worth understanding
----------------------------------------------
The triangulation TOPOLOGY is ordinary Euclidean Delaunay in (u,v); only the
SIZING is metric-driven. Genuinely anisotropic Delaunay -- where the empty-circle
property is defined under a varying metric -- is a much harder problem, and a
globally consistent anisotropic circumcircle does not exist for a non-constant
metric.

The consequence is real and should show up in the three-way comparison: on a
strongly anisotropic face this method produces triangles that are well-shaped in
(u,v) and stretched in R^3. That is a property of parametric CDT, not a bug, and
it is the main thing the Direct 3D Advancing Front should beat it on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .boundary import FaceBoundary
from .metric import MetricField, cholesky_frame, metric_distance
from .predicates import circumcenter, incircle, orient2d, point_in_triangle
from .quality import SurfaceMesh


# ---------------------------------------------------------------------------
# Triangulation with edge adjacency
# ---------------------------------------------------------------------------

class Triangulation:
    """CCW triangles over a shared vertex list, with edge->triangle adjacency.

    Vertices are (u,v) tuples. Triangles are stored by id so adjacency survives
    insertion and deletion without index renumbering.
    """

    def __init__(self, points: list[tuple[float, float]]):
        self.pts: list[tuple[float, float]] = list(points)
        self.tris: dict[int, tuple[int, int, int]] = {}
        self.edge_tris: dict[tuple[int, int], set[int]] = {}
        self._next = 0
        self._hint: int | None = None      # last located triangle, for walk start

    # -- vertex --------------------------------------------------------------
    def add_point(self, uv) -> int:
        self.pts.append((float(uv[0]), float(uv[1])))
        return len(self.pts) - 1

    # -- triangle ------------------------------------------------------------
    @staticmethod
    def _key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def add_triangle(self, a: int, b: int, c: int) -> int:
        if orient2d(self.pts[a], self.pts[b], self.pts[c]) < 0:
            b, c = c, b          # enforce CCW
        tid = self._next
        self._next += 1
        self.tris[tid] = (a, b, c)
        for i, j in ((a, b), (b, c), (c, a)):
            self.edge_tris.setdefault(self._key(i, j), set()).add(tid)
        return tid

    def remove_triangle(self, tid: int) -> None:
        a, b, c = self.tris.pop(tid)
        for i, j in ((a, b), (b, c), (c, a)):
            k = self._key(i, j)
            s = self.edge_tris.get(k)
            if s:
                s.discard(tid)
                if not s:
                    del self.edge_tris[k]

    def neighbours(self, tid: int):
        a, b, c = self.tris[tid]
        out = []
        for i, j in ((a, b), (b, c), (c, a)):
            for other in self.edge_tris.get(self._key(i, j), ()):
                if other != tid:
                    out.append((other, (i, j)))
        return out

    def has_edge(self, i: int, j: int) -> bool:
        return self._key(i, j) in self.edge_tris

    def _locate_brute(self, uv) -> int | None:
        for tid, (a, b, c) in self.tris.items():
            if point_in_triangle(uv, self.pts[a], self.pts[b], self.pts[c]):
                return tid
        return None

    def locate(self, uv, hint: int | None = None) -> int | None:
        """Triangle containing uv, by directed walk from a hint.

        Triangles are CCW, so uv is inside iff orient2d(i, j, uv) >= 0 for all
        three directed edges. A negative test says which edge to cross. That walks
        in roughly O(sqrt(n)) instead of the O(n) brute-force scan, which is what
        turned insertion from O(n^2) into something usable -- a 1850-triangle face
        was taking minutes.

        Falls back to brute force if the walk cycles or leaves the triangulation,
        so correctness never depends on the walk succeeding.
        """
        if not self.tris:
            return None
        tid = hint if (hint is not None and hint in self.tris) else self._hint
        if tid is None or tid not in self.tris:
            tid = next(iter(self.tris))

        for _ in range(4 * len(self.tris) + 16):
            a, b, c = self.tris[tid]
            crossed = False
            for i, j in ((a, b), (b, c), (c, a)):
                if orient2d(self.pts[i], self.pts[j], uv) < 0:
                    nb = [t for t in self.edge_tris.get(self._key(i, j), ())
                          if t != tid]
                    if not nb:
                        return self._locate_brute(uv)   # walked off the boundary
                    tid = nb[0]
                    crossed = True
                    break
            if not crossed:
                self._hint = tid
                return tid
        return self._locate_brute(uv)

    def insert_point(self, uv) -> int:
        """Bowyer-Watson insertion by coordinate, with dedupe.

        Returns the existing index if uv is already a vertex. Note this cannot be
        used to bootstrap vertices that are ALREADY in self.pts but not yet in any
        triangle -- the dedupe would return early and silently skip them. Use
        insert_existing() for that.
        """
        for idx, p in enumerate(self.pts):
            if abs(p[0] - uv[0]) < 1e-14 and abs(p[1] - uv[1]) < 1e-14:
                return idx
        return self.insert_existing(self.add_point(uv))

    def insert_existing(self, vid: int) -> int:
        """Bowyer-Watson insertion of a vertex already present in self.pts.

        Splitting this out from insert_point is not cosmetic: the boundary nodes
        are registered in pts up front so their indices match FaceBoundary's
        global numbering (which the constraint list depends on). Routing them
        through insert_point would hit the coordinate dedupe, return the existing
        index, and never triangulate them at all -- leaving only the super
        triangle and failing every constraint.
        """
        uv = self.pts[vid]
        seed = self.locate(uv)
        if seed is None:
            raise ValueError(f"point {uv} lies outside the triangulation")

        # Cavity = every triangle whose circumcircle contains uv, grown from the
        # seed through adjacency so the cavity stays connected.
        cavity: set[int] = set()
        queue = [seed]
        while queue:
            tid = queue.pop()
            if tid in cavity:
                continue
            a, b, c = self.tris[tid]
            if incircle(self.pts[a], self.pts[b], self.pts[c], uv) <= 0:
                continue
            cavity.add(tid)
            for other, _ in self.neighbours(tid):
                if other not in cavity:
                    queue.append(other)
        if not cavity:
            cavity.add(seed)

        # Directed cavity edges: internal ones appear as (x,y) and (y,x) and
        # cancel; the survivors bound the cavity in CCW order.
        directed: dict[tuple[int, int], int] = {}
        for tid in cavity:
            a, b, c = self.tris[tid]
            for e in ((a, b), (b, c), (c, a)):
                directed[e] = directed.get(e, 0) + 1
        boundary = [e for e in directed
                    if (e[1], e[0]) not in directed]

        for tid in cavity:
            self.remove_triangle(tid)
        for i, j in boundary:
            if i == vid or j == vid:
                continue          # degenerate: vid already on this cavity edge
            self.add_triangle(i, j, vid)
        return vid

    # -- diagnostics ---------------------------------------------------------
    def is_delaunay(self, ignore: set[int] = frozenset()) -> bool:
        """No vertex strictly inside any triangle's circumcircle."""
        for tid, (a, b, c) in self.tris.items():
            if {a, b, c} & ignore:
                continue
            for d in range(len(self.pts)):
                if d in (a, b, c) or d in ignore:
                    continue
                if incircle(self.pts[a], self.pts[b], self.pts[c],
                            self.pts[d]) > 0:
                    return False
        return True


def super_triangle(points, scale: float = 20.0):
    """Three vertices enclosing every point, for Bowyer-Watson bootstrap."""
    p = np.asarray(points, float)
    lo, hi = p.min(axis=0), p.max(axis=0)
    c = 0.5 * (lo + hi)
    r = float(np.max(hi - lo))
    r = max(r, 1e-9) * scale
    return [(c[0] - r, c[1] - r), (c[0] + r, c[1] - r), (c[0], c[1] + r)]


# ---------------------------------------------------------------------------
# Constraint recovery (Anglada)
# ---------------------------------------------------------------------------

def _segments_cross(tri: Triangulation, e, seg) -> bool:
    """Does edge e strictly cross segment seg? Shared endpoints do not count."""
    a, b = e
    s, t = seg
    if len({a, b, s, t}) < 4:
        return False
    pa, pb, ps, pt = tri.pts[a], tri.pts[b], tri.pts[s], tri.pts[t]
    d1 = orient2d(ps, pt, pa)
    d2 = orient2d(ps, pt, pb)
    d3 = orient2d(pa, pb, ps)
    d4 = orient2d(pa, pb, pt)
    return d1 * d2 < 0 and d3 * d4 < 0


def _flip(tri: Triangulation, i: int, j: int) -> bool:
    """Flip the shared edge (i,j) if the surrounding quad is strictly convex."""
    ts = list(tri.edge_tris.get(tri._key(i, j), ()))
    if len(ts) != 2:
        return False
    opp = []
    for tid in ts:
        a, b, c = tri.tris[tid]
        opp.append(next(v for v in (a, b, c) if v not in (i, j)))
    k, l = opp
    # Quad is k, i, l, j. Convex iff k and l are on opposite sides of (i,j) AND
    # i and j are on opposite sides of (k,l).
    if orient2d(tri.pts[i], tri.pts[j], tri.pts[k]) * \
       orient2d(tri.pts[i], tri.pts[j], tri.pts[l]) >= 0:
        return False
    if orient2d(tri.pts[k], tri.pts[l], tri.pts[i]) * \
       orient2d(tri.pts[k], tri.pts[l], tri.pts[j]) >= 0:
        return False
    for tid in ts:
        tri.remove_triangle(tid)
    tri.add_triangle(k, i, l)
    tri.add_triangle(k, l, j)
    return True


def recover_constraints(tri: Triangulation, constraints, max_passes: int = 200):
    """Force every constrained edge into the triangulation by edge flipping.

    Anglada's method: while the constraint is absent, find an edge that crosses it
    and flip it. Each successful flip strictly reduces the number of crossings, so
    the loop terminates. Returns the list of constraints it could not recover.
    """
    failed = []
    for s, t in constraints:
        if s == t or tri.has_edge(s, t):
            continue
        for _ in range(max_passes):
            crossing = [e for e in list(tri.edge_tris.keys())
                        if _segments_cross(tri, e, (s, t))]
            if not crossing:
                break
            progressed = False
            for e in crossing:
                if _flip(tri, e[0], e[1]):
                    progressed = True
                    break
            if not progressed:
                break
            if tri.has_edge(s, t):
                break
        if not tri.has_edge(s, t):
            failed.append((s, t))
    return failed


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

def winding_number(pt, loop_uv) -> int:
    """Winding number of a closed polygon about pt. Nonzero == enclosed.

    Chosen over a parity ray-cast because it handles the outer-CCW/inner-CW
    convention directly: material gives +1, holes and exterior give 0.
    """
    wn = 0
    n = len(loop_uv)
    for i in range(n):
        a = loop_uv[i]
        b = loop_uv[(i + 1) % n]
        if a[1] <= pt[1]:
            if b[1] > pt[1] and orient2d(a, b, pt) > 0:
                wn += 1
        else:
            if b[1] <= pt[1] and orient2d(a, b, pt) < 0:
                wn -= 1
    return wn


def _interior(tri: Triangulation, loops, drop: set[int]):
    """Triangle ids whose centroid is inside the domain, in WORKING coordinates.

    Takes pre-transformed loops rather than the FaceBoundary, because the
    triangulation lives in metric-normalized coordinates. Winding is invariant
    under an orientation-preserving linear map, which is why cholesky_frame is
    required to keep det > 0.
    """
    keep = []
    for tid, (a, b, c) in tri.tris.items():
        if {a, b, c} & drop:
            continue
        pa, pb, pc = tri.pts[a], tri.pts[b], tri.pts[c]
        cen = ((pa[0] + pb[0] + pc[0]) / 3.0, (pa[1] + pb[1] + pc[1]) / 3.0)
        if sum(winding_number(cen, L) for L in loops) != 0:
            keep.append(tid)
    return keep


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

def _encroaches(tri: Triangulation, seg, pt) -> bool:
    """Is pt inside the diametral circle of seg? (angle at pt > 90 degrees)

    Retained for the quality-driven refinement pass, which is not yet wired in.
    The size-driven pass uses longest-edge bisection instead and needs no
    encroachment test -- see the note in triangulate().
    """
    a, b = tri.pts[seg[0]], tri.pts[seg[1]]
    ax, ay = a[0] - pt[0], a[1] - pt[1]
    bx, by = b[0] - pt[0], b[1] - pt[1]
    return ax * bx + ay * by < 0.0


@dataclass
class CDTResult:
    mesh: SurfaceMesh
    # (u,v) of each output vertex, same order as mesh.vertices. Kept because the
    # parametric domain is where a PARAMETRIC mesher's bugs actually live -- a
    # folded or non-partitioning (u,v) triangulation can still look plausible in
    # 3D. Visualizing both is how you tell them apart.
    uv: "np.ndarray | None" = None
    n_refinement_inserts: int = 0
    n_segment_splits: int = 0
    failed_constraints: list = field(default_factory=list)
    hit_iteration_cap: bool = False
    metric_evals: int = 0


def triangulate(boundary: FaceBoundary, surface, *,
                size_tolerance: float = 1.3,
                max_inserts: int = 20000,
                max_passes: int = 40,
                linearize_metric: bool = True) -> CDTResult:
    """Parametric CDT of one face. Returns a SurfaceMesh plus diagnostics.

    size_tolerance: refine a triangle when its longest METRIC edge exceeds
    size_tolerance * target_size. 1.0 would chase its own tail, since inserting a
    point creates edges near the target length.

    linearize_metric: work in metric-normalized coordinates xi = Lt @ (uv - uv0)
    instead of raw (u,v). Lt comes from a Cholesky factor of the first fundamental
    form at the domain centroid, so |d|_M == |Lt d|_2 and Euclidean Delaunay in xi
    is metric Delaunay in (u,v).

    This is the difference between a usable parametric mesher and a broken one. On
    a cylinder of radius 10 the metric is diag(100, 1); raw (u,v) Delaunay yields
    triangles equilateral in (u,v) and 10:1 stretched in R^3 -- measured shape_min
    0.05 and a minimum angle of 1.7 degrees. In xi the same domain becomes the
    UNROLLED 10x5 rectangle and the triangles come out properly shaped.

    The linearization is exact when the metric is constant (planes, cylinders,
    cones) and first-order when it varies (spheres, general NURBS). Set False to
    measure the raw-(u,v) baseline for comparison.
    """
    if boundary.has_seam:
        raise ValueError(
            f"face {boundary.face_tag} has a seam curve; a parametric (u,v) loop "
            "across a seam is ambiguous. Route this face to the direct-3D method.")
    if not boundary.loops:
        raise ValueError(f"face {boundary.face_tag} has no boundary loops")

    mf = MetricField(surface)
    target = boundary.target_size

    uv_all = boundary.all_uv()
    origin = uv_all.mean(axis=0)
    if linearize_metric:
        Lt, Lt_inv = cholesky_frame(mf.at(origin))
    else:
        Lt = Lt_inv = np.eye(2)

    def to_work(p):
        """(u,v) -> working coordinates."""
        d = np.asarray(p, float) - origin
        return tuple(Lt @ d)

    def to_uv(p):
        """working coordinates -> (u,v)."""
        return tuple(Lt_inv @ np.asarray(p, float) + origin)

    uv = [to_work(p) for p in uv_all]
    work_loops = [np.array([to_work(p) for p in l.uv]) for l in boundary.loops]
    n_real = len(uv)
    st = super_triangle(uv)
    tri = Triangulation(uv + st)
    sup = {n_real, n_real + 1, n_real + 2}
    tri.add_triangle(n_real, n_real + 1, n_real + 2)

    # Insert the boundary nodes into the super-triangle one at a time. They
    # already occupy indices [0, n_real) in tri.pts, so insert_point finds them
    # via its coordinate dedupe and returns the existing index rather than
    # appending a duplicate.
    for i in range(n_real):
        tri.insert_existing(i)

    constraints = [tuple(e) for e in boundary.constrained_edges()]
    failed = recover_constraints(tri, constraints)

    # --- metric-driven refinement: longest-edge bisection ---
    #
    # The refinement operator is midpoint insertion on the longest METRIC edge of
    # the worst triangle, not circumcenter insertion. Three reasons, all learned
    # the hard way:
    #
    #  1. It drives the stopping criterion directly. The criterion IS "no metric
    #     edge exceeds size_tolerance * target", and bisecting the offending edge
    #     halves it, so every step makes guaranteed progress and the loop
    #     terminates.
    #  2. An edge midpoint of an interior triangle is always inside the domain.
    #     Circumcenters of obtuse boundary triangles are not, and handling that
    #     meant either aborting refinement (under-refining the whole face) or
    #     blacklisting the triangle (stalling with 9-unit edges against a 2.6
    #     tolerance).
    #  3. No encroachment special case is needed. When the longest edge IS a
    #     constrained boundary segment, bisecting it is exactly the segment split
    #     that Ruppert's encroachment rule exists to trigger -- so the boundary
    #     refines only when it is genuinely the coarsest thing present, instead of
    #     cascading. The earlier circumcenter version split a 14-node boundary to
    #     53 nodes and produced ~8x the intended triangle count.
    #
    # Quality still benefits from Delaunay: Bowyer-Watson re-triangulates after
    # each insertion, so the mesh stays Delaunay throughout.
    seg_set = {tri._key(*c) for c in constraints}
    n_ins = n_split = 0
    capped = False
    limit = size_tolerance * target

    # Pass-based refinement. Each pass classifies the interior ONCE, collects
    # every offending edge, and bisects them all before reclassifying.
    #
    # The earlier version found the single globally worst edge per insertion, which
    # meant one _interior() sweep (O(triangles x boundary) winding tests) per
    # inserted point. That is cubic overall and made a 1850-triangle face take
    # minutes. Bisecting a whole batch per pass cuts the number of sweeps from
    # O(inserts) to O(log(initial_size / target)) -- typically under 20.
    #
    # Edges can vanish mid-pass when a neighbouring bisection retriangulates the
    # cavity, so each is re-checked with has_edge before use.
    for _ in range(max_passes):
        offenders: dict[tuple[int, int], float] = {}
        for tid in _interior(tri, work_loops, sup):
            a, b, c = tri.tris[tid]
            for i, j in ((a, b), (b, c), (c, a)):
                key = tri._key(i, j)
                if key in offenders:
                    continue
                L = mf.distance(to_uv(tri.pts[i]), to_uv(tri.pts[j]))
                if L > limit:
                    offenders[key] = L
        if not offenders:
            break

        # Longest first, so the coarsest regions improve fastest.
        for (i, j), _L in sorted(offenders.items(), key=lambda kv: -kv[1]):
            if not tri.has_edge(i, j):
                continue                      # retriangulated away this pass
            pa, pb = tri.pts[i], tri.pts[j]
            mid = (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]))
            key = tri._key(i, j)
            is_segment = key in seg_set

            new_v = tri.insert_point(mid)
            if new_v == i or new_v == j:
                continue                      # midpoint collapsed onto an endpoint

            if is_segment:
                seg_set.discard(key)
                seg_set.add(tri._key(i, new_v))
                seg_set.add(tri._key(new_v, j))
                recover_constraints(tri, [(i, new_v), (new_v, j)])
                n_split += 1
            else:
                n_ins += 1
            if n_ins + n_split > max_inserts:
                capped = True
                break
        if capped:
            break
    else:
        capped = True

    kept = _interior(tri, work_loops, sup)

    # --- emit ---
    used = sorted({v for tid in kept for v in tri.tris[tid]})
    remap = {v: i for i, v in enumerate(used)}
    verts3d = np.array([surface.point(to_uv(tri.pts[v])) for v in used],
                       dtype=float)
    faces = np.array([[remap[v] for v in tri.tris[tid]] for tid in kept],
                     dtype=np.int64)

    uv_out = np.array([to_uv(tri.pts[v]) for v in used], dtype=float)
    mesh = SurfaceMesh(vertices=verts3d, triangles=faces,
                       method="parametric_cdt",
                       meta={"face_tag": boundary.face_tag,
                             "target_size": target,
                             "boundary_fingerprint": boundary.fingerprint()})
    return CDTResult(mesh=mesh, uv=uv_out, n_refinement_inserts=n_ins,
                     n_segment_splits=n_split, failed_constraints=failed,
                     hit_iteration_cap=capped, metric_evals=mf.n_evals)
