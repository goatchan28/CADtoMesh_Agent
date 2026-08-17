"""
Direct 3D Advancing Front. Our own implementation.

No parametric domain. The front lives in R^3 on the surface, and advances by
stepping along the surface's tangent plane and projecting back. The only surface
operations used are project() and normal().

Why this method exists alongside the parametric one
---------------------------------------------------
Both parametric methods share two structural weaknesses that are properties of
working in (u,v), not implementation defects:

  * SEAMS. A periodic face's (u,v) loop is ambiguous where it crosses the seam, so
    both parametric methods must refuse such faces outright. This method never
    forms a (u,v) loop, so a seam is not a special case at all.
  * METRIC VARIATION. Metric normalization uses one Cholesky factor at the domain
    centroid -- exact for constant-metric surfaces (planes, cylinders, cones),
    first-order everywhere else. Near a sphere pole the parametric CDT measured
    shape_min 0.535 and a 19.3 degree minimum angle against 0.630 / 26.8 on a
    mid-latitude band of the same sphere. This method has no metric to linearize:
    every step is measured in real space.

The cost is that "perpendicular" and "does this triangle overlap the front" are no
longer plain 2D questions. Both are answered in a LOCAL TANGENT FRAME built at each
front edge, which is where the shared 2D validation from af_core gets reused.

Orientation
-----------
For a front edge i->j with outward surface normal n, the in-surface direction
pointing into un-meshed material is p = n x e, where e is the unit edge direction.
Check with n = +z, e = +x: n x e = +y, which is indeed to the left of +x seen from
+z. The same material-on-the-left convention as the parametric method, so the
shared front bookkeeping applies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .af_core import (AFConfig, AFStats, Front, apex_height, order_candidates,
                      triangle_is_valid_2d)
from .boundary import FaceBoundary
from .cdt_parametric import winding_number
from quality.core import SurfaceMesh


@dataclass
class AF3DResult:
    mesh: SurfaceMesh
    uv: np.ndarray | None = None
    stats: AFStats = field(default_factory=AFStats)


def _unit(v: np.ndarray) -> np.ndarray:
    L = float(np.linalg.norm(v))
    return v / L if L > 0 else v


class _Counter:
    """Counts surface evaluations, so cost is comparable between methods."""

    def __init__(self, surface):
        self.s = surface
        self.n = 0

    def project(self, xyz):
        self.n += 1
        return self.s.project(xyz)

    def frame_normal(self, uv):
        """Surface normal CONSISTENT WITH THE UV ORIENTATION: unit(dX/du x dX/dv).

        Deliberately NOT surface.normal(), which is free to report the
        geometrically outward normal. The two can be opposite: this sphere's
        parametrization has du x dv pointing INWARD while normal() points outward.

        The front's inward direction is p = n x e, and material-on-the-left was
        established by orient_loops IN UV. So n must be the uv-consistent normal or
        p points out of the patch and the front escapes. That is exactly what
        happened: the sphere cases meshed 538 and 599 triangles against ideals of
        149 and 78, with 240% and 630% area error, because the front marched off the
        patch and wrapped around the sphere. The plane and cylinder were unaffected
        only because their two normals happen to agree.

        Local derivatives are seam-safe -- it is the (u,v) LOOP that is ambiguous at
        a seam, not the tangent frame at a point.
        """
        self.n += 1
        du, dv = self.s.derivatives(uv)
        return _unit(np.cross(np.asarray(du, float), np.asarray(dv, float)))

    def is_inside(self, uv):
        return self.s.is_inside(uv)


def mesh_face(boundary: FaceBoundary, surface, *,
              config: AFConfig | None = None) -> AF3DResult:
    """Advancing-front surface mesh of one face, marching directly in R^3.

    Uses only the xyz side of the boundary loops. The uv side is ignored entirely,
    which is what lets this method accept faces the parametric ones refuse.
    """
    cfg = config or AFConfig()
    if not boundary.loops:
        raise ValueError(f"face {boundary.face_tag} has no boundary loops")

    surf = _Counter(surface)
    target = boundary.target_size

    xyz = boundary.all_xyz()
    front = Front(xyz, cell=target, dim=3)
    front.seed_from_loops(boundary.constrained_edges())

    # Cached (u,v) per node, needed only for normals and the trim test.
    node_uv: dict[int, np.ndarray] = {}
    uv_seed = np.concatenate([l.uv for l in boundary.loops], axis=0)
    for idx in range(len(uv_seed)):
        node_uv[idx] = uv_seed[idx]

    def uv_of(idx: int) -> np.ndarray:
        uv = node_uv.get(idx)
        if uv is None:
            _, uv = surf.project(front.pts[idx])
            node_uv[idx] = uv
        return uv

    # A uv winding test is available whenever the loops are unambiguous. It is a
    # GUARD, not part of the construction: the front is supposed to confine itself,
    # but a single orientation error lets it escape and silently wrap the surface,
    # and the local front-crossing test cannot see a front edge on the far side.
    # Seam faces skip it, which is precisely the case this method exists for.
    uv_loops = None if boundary.has_seam else [l.uv for l in boundary.loops]

    def apex_allowed(uv) -> bool:
        if not surf.is_inside(uv):
            return False
        if uv_loops is None:
            return True
        return sum(winding_number(tuple(uv), L) for L in uv_loops) != 0

    _assert_front_points_inward(front, boundary, surf, apex_allowed, target)

    stats = AFStats()
    reuse_r = cfg.reuse_radius_factor * target
    min_dist = cfg.min_node_dist_factor * target

    while front.edges and stats.n_triangles < cfg.max_triangles:
        progressed = False
        for min_q, relaxed in ((cfg.min_quality_strict, False),
                               (cfg.min_quality_relaxed, True)):
            for (i, j) in front.shortest_edges():
                if (i, j) not in front.edges:
                    continue

                pi3, pj3 = front.pts[i], front.pts[j]
                L = float(np.linalg.norm(pj3 - pi3))
                if L <= 0:
                    front.remove_edge(i, j)
                    continue

                e = _unit(pj3 - pi3)
                mid3 = 0.5 * (pi3 + pj3)
                # Anchor the frame ON the surface: the chord midpoint sits slightly
                # off a curved surface, and building the tangent frame at the
                # off-surface point tilts every subsequent measurement.
                mid_on, mid_uv = surf.project(mid3)
                n = surf.frame_normal(mid_uv)
                p = _unit(np.cross(n, e))          # into the material
                if float(np.linalg.norm(p)) == 0.0:
                    continue

                # Local tangent frame: origin mid_on, axes (e, p). The front edge
                # maps to roughly (-L/2,0)-(+L/2,0) and material to +y, so the
                # shared 2D validator sees exactly the layout it expects.
                def to2d(x3, _o=mid_on, _e=e, _p=p):
                    d = np.asarray(x3, float) - _o
                    return np.array([float(d @ _e), float(d @ _p)])

                def proj(idx, _f=to2d):
                    return _f(front.pts[idx])

                apex3 = mid_on + p * apex_height(L, target)
                apex_on, apex_uv = surf.project(apex3)

                existing = [k for k in front.front_nodes_near(apex_on, reuse_r)
                            if k not in (i, j)]
                pi2, pj2 = proj(i), proj(j)
                candidates = order_candidates(pi2, pj2, to2d(apex_on), existing,
                                              proj, target, cfg)

                chosen = None
                for node, pos2 in candidates:
                    if node is None and not apex_allowed(apex_uv):
                        continue
                    probe = node if node is not None else -1
                    if triangle_is_valid_2d(front, i, j, probe, pos2, proj,
                                            existing, min_quality=min_q,
                                            min_node_dist=min_dist):
                        if node is None:
                            chosen = front.add_node(apex_on)
                            node_uv[chosen] = apex_uv
                            stats.n_new_nodes += 1
                        else:
                            chosen = node
                            stats.n_existing_node_reuse += 1
                        break

                if chosen is None:
                    stats.n_edge_retries += 1
                    continue

                front.commit(i, j, chosen)
                stats.n_triangles += 1
                if relaxed:
                    stats.n_relaxed_accepts += 1
                progressed = True
                break
            if progressed:
                break

        if not progressed:
            stats.stalled_edges = len(front.edges)
            break

    stats.front_remaining = len(front.edges)
    stats.completed = len(front.edges) == 0
    stats.surface_evals = surf.n

    used = sorted({v for t in front.triangles for v in t})
    remap = {v: k for k, v in enumerate(used)}
    verts = np.array([front.pts[v] for v in used], dtype=float) if used \
        else np.zeros((0, 3))
    tris = np.array([[remap[v] for v in t] for t in front.triangles],
                    dtype=np.int64).reshape(-1, 3)
    uv_out = np.array([uv_of(v) for v in used], dtype=float) if used \
        else np.zeros((0, 2))

    mesh = SurfaceMesh(vertices=verts, triangles=tris, method="direct3d_af",
                       meta={"face_tag": boundary.face_tag, "target_size": target,
                             "boundary_fingerprint": boundary.fingerprint()})
    return AF3DResult(mesh=mesh, uv=uv_out, stats=stats)


def _assert_front_points_inward(front: Front, boundary: FaceBoundary, surf,
                                apex_allowed, target: float) -> None:
    """Verify p = n x e actually points INTO the domain, on real boundary edges.

    Cheap, and it converts the single most damaging possible error -- a flipped
    orientation convention -- from a silently wrong mesh into an immediate,
    named failure. Checks several edges because one could sit at a spot where the
    trim test is inconclusive.
    """
    edges = list(boundary.constrained_edges())[:8]
    inward = outward = 0
    for i, j in edges:
        pi3, pj3 = front.pts[i], front.pts[j]
        L = float(np.linalg.norm(pj3 - pi3))
        if L <= 0:
            continue
        e = _unit(pj3 - pi3)
        mid_on, mid_uv = surf.project(0.5 * (pi3 + pj3))
        n = surf.frame_normal(mid_uv)
        p = _unit(np.cross(n, e))
        step = 0.15 * min(target, L)
        _, uv_in = surf.project(mid_on + p * step)
        _, uv_out = surf.project(mid_on - p * step)
        if apex_allowed(uv_in) and not apex_allowed(uv_out):
            inward += 1
        elif apex_allowed(uv_out) and not apex_allowed(uv_in):
            outward += 1
    if outward > inward:
        raise ValueError(
            "direct3d front orientation is inverted: n x e points OUT of the "
            f"domain on {outward} of {inward + outward} conclusive boundary edges. "
            "The surface normal used must be unit(dX/du x dX/dv) so it agrees with "
            "the uv winding that orient_loops established.")
