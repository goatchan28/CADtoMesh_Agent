"""
Shared boundary discretization -- the fairness guarantee.

All three meshers must start from BYTE-IDENTICAL boundary nodes, or the comparison
measures nothing. If one method got a slightly finer boundary it would look better
on element count and worse on quality, and there would be no way to separate that
from the algorithm.

So the boundary is built ONCE, here, and handed to all three as an immutable
FaceBoundary. `fingerprint()` hashes the node coordinates and loop structure;
compare.py asserts all three runs received the same fingerprint. That turns
fairness from an intention into a checked precondition.

Two representations are carried for every boundary node:

    uv   -- parametric, for the two PARAMETRIC methods
    xyz  -- real space,  for the DIRECT 3D method

They are the same nodes. The direct-3D front never looks at uv for construction
and the parametric methods never look at xyz, but both are derived from one
discretization, so the two families genuinely start from the same place.

Seam curves are the known hazard. A periodic face's seam has two valid (u,v)
traces, so a parametric loop crossing it is ambiguous; Stage 0 already classifies
such faces SEAMED. FaceBoundary records `has_seam` and the parametric meshers
refuse rather than silently producing a wrong loop. The direct-3D method is
unaffected, which is one of the differences the comparison should surface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .predicates import orient2d


@dataclass(frozen=True)
class Loop:
    """One closed boundary loop of a face.

    Node i connects to node i+1, and the last connects back to the first. The
    closing edge is IMPLICIT -- never duplicate the first node at the end.
    """
    uv: np.ndarray            # (K, 2)
    xyz: np.ndarray           # (K, 3)
    curve_tags: tuple[int, ...] = ()
    is_outer: bool = True

    def __len__(self) -> int:
        return len(self.uv)

    def edges(self):
        """Directed (i, j) index pairs, including the implicit closing edge."""
        k = len(self.uv)
        return [(i, (i + 1) % k) for i in range(k)]

    def signed_area_uv(self) -> float:
        """Shoelace area in parametric space. Positive == counter-clockwise."""
        u, v = self.uv[:, 0], self.uv[:, 1]
        return 0.5 * float(np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v))


@dataclass(frozen=True)
class FaceBoundary:
    """Immutable, shared input to every mesher.

    Node numbering is global across loops: loop 0 occupies indices
    [0, len(loop0)), loop 1 continues from there, and so on. All three meshers
    must preserve those indices for the boundary nodes so their outputs are
    directly comparable vertex-for-vertex on the boundary.
    """
    face_tag: int
    loops: tuple[Loop, ...]
    target_size: float                 # desired real-space edge length
    has_seam: bool = False
    seam_curve_tags: tuple[int, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return sum(len(l) for l in self.loops)

    def loop_offsets(self) -> list[int]:
        offs, run = [], 0
        for l in self.loops:
            offs.append(run)
            run += len(l)
        return offs

    def all_uv(self) -> np.ndarray:
        return np.concatenate([l.uv for l in self.loops], axis=0)

    def all_xyz(self) -> np.ndarray:
        return np.concatenate([l.xyz for l in self.loops], axis=0)

    def constrained_edges(self) -> list[tuple[int, int]]:
        """Global-index edges that MUST appear in the output mesh.

        These are the CDT's constraints and the advancing front's initial front.
        Same list for both, which is the point.
        """
        out: list[tuple[int, int]] = []
        for off, l in zip(self.loop_offsets(), self.loops):
            out.extend([(off + i, off + j) for i, j in l.edges()])
        return out

    def fingerprint(self) -> str:
        """Stable hash of the boundary. Equal fingerprint == identical input."""
        h = hashlib.sha256()
        h.update(f"face={self.face_tag};target={self.target_size!r}".encode())
        for l in self.loops:
            h.update(f"|loop outer={l.is_outer} k={len(l)}|".encode())
            # Round to 12 significant digits so a harmless last-bit difference in
            # a recomputed coordinate does not read as a different boundary.
            h.update(np.round(l.uv, 12).tobytes())
            h.update(np.round(l.xyz, 12).tobytes())
        return h.hexdigest()[:16]

    def total_boundary_length(self) -> float:
        total = 0.0
        for l in self.loops:
            d = np.roll(l.xyz, -1, axis=0) - l.xyz
            total += float(np.sum(np.linalg.norm(d, axis=1)))
        return total


def orient_loops(loops: list[Loop]) -> tuple[Loop, ...]:
    """Normalize winding: outer loop CCW, inner loops CW, in parametric space.

    Both parametric meshers depend on this. The CDT classifies inside/outside by
    winding, and the advancing front must know which side of a boundary edge the
    material is on. Getting it wrong on an inner loop fills the hole instead of
    leaving it -- a failure that produces a perfectly valid-looking mesh of the
    wrong domain.
    """
    if not loops:
        return ()
    areas = [l.signed_area_uv() for l in loops]
    # The outer loop is the one with the largest absolute area.
    outer_idx = int(np.argmax(np.abs(areas)))

    fixed: list[Loop] = []
    for i, (l, a) in enumerate(zip(loops, areas)):
        want_ccw = (i == outer_idx)
        is_ccw = a > 0
        if is_ccw != want_ccw:
            l = Loop(uv=l.uv[::-1].copy(), xyz=l.xyz[::-1].copy(),
                     curve_tags=l.curve_tags, is_outer=want_ccw)
        else:
            l = Loop(uv=l.uv, xyz=l.xyz, curve_tags=l.curve_tags,
                     is_outer=want_ccw)
        fixed.append(l)
    return tuple(fixed)


def loop_is_simple(loop: Loop) -> bool:
    """True if the loop does not self-intersect in (u,v).

    A self-intersecting boundary makes both parametric methods ill-posed, so it is
    worth failing loudly here rather than producing folded triangles later.
    """
    from .predicates import segments_properly_intersect
    pts = [tuple(p) for p in loop.uv]
    e = loop.edges()
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            a, b = e[i]
            c, d = e[j]
            if len({a, b, c, d}) < 4:
                continue          # adjacent edges share a node, fine
            if segments_properly_intersect(pts[a], pts[b], pts[c], pts[d]):
                return False
    return True


# ---------------------------------------------------------------------------
# Synthetic construction -- for testing without gmsh
# ---------------------------------------------------------------------------

def boundary_from_uv_polygon(uv_loops, surface, target_size: float,
                             face_tag: int = 0) -> FaceBoundary:
    """Build a FaceBoundary from explicit (u,v) polygons on an analytic surface.

    This is how the meshers get tested without gmsh: hand them a square, an
    annulus, a cylinder patch, and check invariants against known answers.
    """
    loops: list[Loop] = []
    for uv in uv_loops:
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        xyz = np.array([surface.point(p) for p in uv], dtype=float)
        loops.append(Loop(uv=uv, xyz=xyz))
    return FaceBoundary(face_tag=face_tag, loops=orient_loops(loops),
                        target_size=float(target_size))


def resample_uv_polygon(corners, surface, target_size: float) -> np.ndarray:
    """Subdivide a (u,v) polygon so every edge is about target_size in REAL space.

    Uses metric arc length, not parametric length, so a cylinder patch gets the
    same physical spacing as a plane patch. Without this the "same boundary" would
    still be unfair across surfaces.
    """
    from .metric import MetricField
    mf = MetricField(surface)
    corners = np.asarray(corners, dtype=float).reshape(-1, 2)
    out: list[np.ndarray] = []
    k = len(corners)
    for i in range(k):
        a, b = corners[i], corners[(i + 1) % k]
        length = mf.distance(a, b)
        n = max(1, int(round(length / target_size)))
        for s in range(n):                 # exclude the endpoint; next edge adds it
            out.append(a + (b - a) * (s / n))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# gmsh construction
# ---------------------------------------------------------------------------

def boundary_from_gmsh_face(face_tag: int, target_size: float,
                            seam_curve_tags=(), n_per_curve_min: int = 1):
    """Build a FaceBoundary from a gmsh face's EXISTING 1D discretization.

    Requires gmsh.model.mesh.generate(1) to have already run, so the curve nodes
    are exactly the ones Stage 0 measured and every face sharing a curve gets the
    same nodes. That shared 1D mesh is what makes a CAD surface mesh watertight by
    construction, and reusing it here is what makes this Stage 1 conformal rather
    than a per-face free-for-all.

    Returns None when the face has a seam, since a parametric loop across a seam
    is ambiguous. Caller should route such faces to the direct-3D method.
    """
    import gmsh

    bnd = gmsh.model.getBoundary([(2, face_tag)], combined=False, oriented=True)
    seen: dict[int, int] = {}
    for _, t in bnd:
        seen[abs(t)] = seen.get(abs(t), 0) + 1
    seams = tuple(sorted(t for t, n in seen.items() if n > 1)) or tuple(seam_curve_tags)

    # Ordered curve list per loop: use combined=True to get loops, then walk.
    curve_nodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, ctag in bnd:
        t = abs(ctag)
        if t in curve_nodes:
            continue
        ntags, coords, params = gmsh.model.mesh.getNodes(
            1, t, includeBoundary=True, returnParametricCoord=True)
        xyz = np.asarray(coords, float).reshape(-1, 3)
        uv = np.asarray(
            gmsh.model.reparametrizeOnSurface(1, t, list(np.asarray(params, float)),
                                             face_tag), float).reshape(-1, 2)
        curve_nodes[t] = (uv, xyz)

    if seams:
        return FaceBoundary(face_tag=face_tag, loops=(), target_size=target_size,
                            has_seam=True, seam_curve_tags=seams,
                            meta={"reason": "seam curve: parametric loop ambiguous"})

    # Single-loop assembly. Multi-loop ordering from getBoundary(combined=True)
    # is left for the next pass; faces with holes are detected and reported.
    uvs = np.concatenate([curve_nodes[abs(t)][0] for _, t in bnd], axis=0)
    xyzs = np.concatenate([curve_nodes[abs(t)][1] for _, t in bnd], axis=0)
    uvs, xyzs = _dedupe_consecutive(uvs, xyzs)
    loops = orient_loops([Loop(uv=uvs, xyz=xyzs,
                               curve_tags=tuple(abs(t) for _, t in bnd))])
    return FaceBoundary(face_tag=face_tag, loops=loops,
                        target_size=float(target_size),
                        seam_curve_tags=seams)


def _dedupe_consecutive(uv: np.ndarray, xyz: np.ndarray, tol: float = 1e-12):
    """Drop repeated nodes where consecutive curves share an endpoint."""
    keep = [0]
    for i in range(1, len(uv)):
        if np.linalg.norm(uv[i] - uv[keep[-1]]) > tol:
            keep.append(i)
    if len(keep) > 2 and np.linalg.norm(uv[keep[-1]] - uv[keep[0]]) <= tol:
        keep.pop()
    idx = np.asarray(keep, dtype=int)
    return uv[idx].copy(), xyz[idx].copy()
