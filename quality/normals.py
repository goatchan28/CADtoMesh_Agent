"""
CAD normal deviation: angle between each facet's normal and the true surface normal.

Why this is not redundant with chordal deviation
------------------------------------------------
Chordal deviation asks "how far is the facet from the surface?" Normal deviation
asks "does the facet point the same way as the surface?" They are independent, and
the second is the stronger statement.

The Schwarz lantern makes this concrete. An inscribed triangulation of a cylinder
can be refined so that every vertex lies exactly on the surface and chordal
deviation goes to zero, while total area converges to the wrong value or diverges.
What fails in that construction is precisely NORMAL convergence -- the facets
become long thin slivers whose normals spin away from the surface normal. Our own
measurements showed area approaching 50 from ABOVE (50.83, 50.15, 50.06), which is
the same phenomenon in mild form.

That matters for simulation, not just for geometry:
  * surface integrals -- pressure loads, fluxes, contact -- are computed against
    facet normals, so normal error is load error
  * shell and plate formulations use the facet normal as the director
  * anything involving reflection, radiation, or view factors is normal-driven

Sign convention
---------------
The angle is measured UNSIGNED, to the nearer of +n or -n. Global orientation is a
separate concern already covered by topology_checks (n_inconsistent_normals), and
conflating the two here would be actively misleading: this project has already been
bitten once by a surface whose analytic normal() pointed opposite to
unit(dX/du x dX/dv). On that sphere every facet would have reported ~180 degrees of
"deviation" while being geometrically perfect. So this metric answers "is the facet
parallel to the surface", and orientation consistency is answered elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .core import SurfaceMesh, triangle_areas, triangle_normals


@dataclass
class NormalReport:
    max_deg: float
    mean_deg: float
    p99_deg: float
    area_weighted_mean_deg: float
    frac_above_limit: float
    limit_deg: float
    n_sampled: int
    worst_triangles: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"normal dev: max {self.max_deg:.2f} deg, mean {self.mean_deg:.2f}, "
                f"p99 {self.p99_deg:.2f}, {self.frac_above_limit * 100:.1f}% above "
                f"{self.limit_deg:.1f} deg")


def _unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def normal_deviation(mesh: SurfaceMesh, surface, *, limit_deg: float = 15.0,
                     max_triangles: int | None = None, seed: int = 0,
                     n_worst: int = 20) -> NormalReport:
    """Angle between facet normals and the true surface normal at each centroid.

    Args:
        surface: needs .project(xyz) -> (closest_xyz, uv) and .normal(uv). Both are
            part of the Surface protocol, so this stays backend-independent: the
            same evaluator scores a gmsh mesh and a research mesh.
        limit_deg: reporting threshold. 15 degrees is a common default for
            visualization-grade tessellation; simulation surfaces carrying pressure
            loads generally want tighter.
        max_triangles: subsample for speed. Each sample costs one projection plus
            one normal evaluation, and against a gmsh-backed surface both are OCC
            calls.

    Returns NormalReport. Exactly zero on a planar mesh, which is the check that
    projection and normal evaluation agree.
    """
    if mesh.n_triangles == 0:
        return NormalReport(0.0, 0.0, 0.0, 0.0, 0.0, limit_deg, 0)

    a, b, c = mesh.corners()
    idx = np.arange(len(a))
    if max_triangles is not None and len(idx) > max_triangles:
        idx = np.random.default_rng(seed).choice(idx, max_triangles, replace=False)
        idx.sort()

    facet_n = _unit_rows(triangle_normals(a[idx], b[idx], c[idx]))
    centroids = (a[idx] + b[idx] + c[idx]) / 3.0

    surf_n = np.empty_like(facet_n)
    for k, p in enumerate(centroids):
        _, uv = surface.project(p)
        surf_n[k] = np.asarray(surface.normal(uv), float)
    surf_n = _unit_rows(surf_n)

    # abs() is the unsigned convention: measure to the nearer of +n or -n.
    cos = np.abs(np.einsum("ij,ij->i", facet_n, surf_n))
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    areas = triangle_areas(a[idx], b[idx], c[idx])
    tot = float(areas.sum())
    aw = float((ang * areas).sum() / tot) if tot > 0 else 0.0

    order = np.argsort(ang)[::-1][:n_worst]
    worst = [{"tri": int(idx[i]), "deg": float(ang[i])} for i in order]

    return NormalReport(
        max_deg=float(ang.max()), mean_deg=float(ang.mean()),
        p99_deg=float(np.percentile(ang, 99.0)),
        area_weighted_mean_deg=aw,
        frac_above_limit=float(np.mean(ang > limit_deg)),
        limit_deg=float(limit_deg), n_sampled=len(idx), worst_triangles=worst)


def facet_angle_bound(radius: float, chord: float) -> float:
    """Exact facet-to-surface angle for a chord on a circle of given radius, degrees.

    A chord subtending angle theta = 2*asin(c/2R) has its own direction rotated
    theta/2 from the tangent at each endpoint, so the maximum facet-to-surface angle
    along that chord is theta/2. This is the closed-form reference that
    normal_deviation must reproduce on a cylinder or sphere -- the analogue of
    quality.core.sagitta for chordal deviation.

    Note the scaling: the angle is O(c/R) while sagitta is O(c^2/R). Halving element
    size halves normal error but QUARTERS chordal error, so normal deviation is the
    binding constraint at coarse sizes and the slower one to improve.
    """
    half = 0.5 * chord
    if half >= radius:
        return 90.0
    return math.degrees(math.asin(half / radius))
