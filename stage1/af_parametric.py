"""
Parametric Advancing Front. Our own implementation.

Not gmsh Algorithm 6. No gmsh meshing code is used or consulted; the only gmsh
contact is through the Surface protocol, which supplies data about the CAD face.

Strategy: advance the front in METRIC-NORMALIZED parametric coordinates
xi = Lt @ (uv - uv0), where Lt is a Cholesky factor of the first fundamental form
at the domain centroid. In xi, |d|_2 == |d|_M, so an equilateral triangle in xi is
an equilateral triangle in R^3 -- which means the ordinary 2D advancing-front
construction (perpendicular offset, isoceles apex) is directly correct.

Why not advance in raw (u,v): the parametric CDT measurement makes the case
concretely. On a cylinder of radius 10 the metric is diag(100, 1); raw (u,v)
geometry produced a minimum angle of 1.8 degrees where the normalized version
produced 28.4. An advancing front in raw (u,v) fails the same way, and worse --
its "perpendicular" direction is not perpendicular on the surface at all, so the
front marches off at an angle.

Known limitation, shared with the parametric CDT: a face with a SEAM has an
ambiguous (u,v) loop and is refused. Faces whose metric varies strongly (sphere
poles) get a single global linearization, which is first-order rather than exact.
The Direct 3D Advancing Front exists to handle both cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .af_core import (AFConfig, AFStats, Front, ideal_apex_2d,
                      order_candidates, triangle_is_valid_2d)
from .boundary import FaceBoundary
from .cdt_parametric import winding_number
from .metric import MetricField, cholesky_frame
from .quality import SurfaceMesh


@dataclass
class AFResult:
    mesh: SurfaceMesh
    uv: np.ndarray | None = None
    stats: AFStats = field(default_factory=AFStats)


def mesh_face(boundary: FaceBoundary, surface, *,
              config: AFConfig | None = None,
              linearize_metric: bool = True) -> AFResult:
    """Advancing-front surface mesh of one face, in parametric space."""
    cfg = config or AFConfig()
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
    Lt, Lt_inv = (cholesky_frame(mf.at(origin)) if linearize_metric
                  else (np.eye(2), np.eye(2)))

    def to_work(p):
        return Lt @ (np.asarray(p, float) - origin)

    def to_uv(p):
        return Lt_inv @ np.asarray(p, float) + origin

    work_loops = [np.array([to_work(p) for p in l.uv]) for l in boundary.loops]
    front = Front([to_work(p) for p in uv_all], cell=target, dim=2)
    front.seed_from_loops(boundary.constrained_edges())

    def proj(idx):
        return front.pts[idx]

    def inside(pw) -> bool:
        uv = to_uv(pw)
        if not surface.is_inside(uv):
            return False
        return sum(winding_number(tuple(pw), L) for L in work_loops) != 0

    stats = AFStats()
    reuse_r = cfg.reuse_radius_factor * target
    min_dist = cfg.min_node_dist_factor * target

    while front.edges and stats.n_triangles < cfg.max_triangles:
        progressed = False
        # Two sweeps: strict quality first, then relaxed. Relaxing only after every
        # edge has been tried strictly keeps poor triangles as a last resort rather
        # than a convenience, which matters because one accepted sliver tends to
        # force more.
        for min_q, relaxed in ((cfg.min_quality_strict, False),
                               (cfg.min_quality_relaxed, True)):
            for (i, j) in front.shortest_edges():
                if (i, j) not in front.edges:
                    continue
                pi, pj = front.pts[i], front.pts[j]
                ideal = ideal_apex_2d(pi, pj, target)

                existing = [n for n in front.front_nodes_near(ideal, reuse_r)
                            if n not in (i, j)]
                candidates = order_candidates(pi, pj, ideal, existing, proj,
                                              target, cfg)

                chosen = None
                for node, pos in candidates:
                    if node is None and not inside(pos):
                        continue
                    probe = node if node is not None else -1
                    if triangle_is_valid_2d(front, i, j, probe, pos, proj, existing,
                                            min_quality=min_q,
                                            min_node_dist=min_dist):
                        if node is None:
                            chosen = front.add_node(pos)
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
    stats.surface_evals = mf.n_evals

    uv_out = np.array([to_uv(p) for p in front.pts], dtype=float)
    verts = np.array([surface.point(p) for p in uv_out], dtype=float)
    tris = np.array(front.triangles, dtype=np.int64).reshape(-1, 3)
    used = sorted({v for t in front.triangles for v in t})
    remap = {v: n for n, v in enumerate(used)}
    mesh = SurfaceMesh(
        vertices=verts[used] if used else np.zeros((0, 3)),
        triangles=np.array([[remap[v] for v in t] for t in front.triangles],
                           dtype=np.int64).reshape(-1, 3),
        method="parametric_af",
        meta={"face_tag": boundary.face_tag, "target_size": target,
              "boundary_fingerprint": boundary.fingerprint(),
              "linearized": linearize_metric})
    return AFResult(mesh=mesh, uv=uv_out[used] if used else np.zeros((0, 2)),
                    stats=stats)
