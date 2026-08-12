"""
Stage 0b -- 3D to 2D reduction.

Every OCC face carries a native parametrization X(u,v): R^2 -> R^3. Gmsh's 2D
algorithms (MeshAdapt, Delaunay, Frontal-Delaunay) mesh in that (u,v) domain and
map the result back. That is fast and exact -- but only when the map is well
behaved. This module measures how well behaved it is, per face, so the agent can
choose an algorithm per face instead of globally.

The instrument is the first fundamental form of the map:

    I = [[E, F], [F, G]],  E = Xu.Xu,  F = Xu.Xv,  G = Xv.Xv

From it:
  * sqrt(det I) = sqrt(EG - F^2)  -- local area scaling of the map
  * sqrt(lambda_max / lambda_min) -- local anisotropy (stretch ratio)

Two failure signatures matter:
  * det I -> 0 somewhere: the map collapses. Sphere poles, cone apexes.
    Parametric-space meshing puts unboundedly many elements near the pole.
  * area-scale ratio across the face >> 1: a uniform (u,v) mesh becomes wildly
    non-uniform in R^3. Fixable by metric-driven sizing, but worth knowing.

Note the distinction from element quality: a face can be geometrically simple and
still parametrize terribly. Trimmed NURBS from a CAD kernel routinely do.
"""

from __future__ import annotations

import math

import gmsh
import numpy as np

from .config import Thresholds, DEFAULTS
from .metrics import metric_from_derivatives
from .report import Stage0Report, FaceParametrization, ParamClass, Severity

# Thresholds now live in config.Thresholds so they are calibratable and are
# recorded in every report. Module constants remain only as fallbacks.


def _seam_curves(face_tag: int) -> list[int]:
    """A seam curve appears twice in a periodic face's oriented boundary.

    Cylinders, spheres and tori all have one. Gmsh handles seams internally, but
    reparametrizeOnSurface() is ambiguous on them, so anything that maps curve
    parameters onto a face must special-case these.
    """
    bnd = gmsh.model.getBoundary([(2, face_tag)], combined=False, oriented=True)
    seen: dict[int, int] = {}
    for _, tag in bnd:
        seen[abs(tag)] = seen.get(abs(tag), 0) + 1
    return sorted(t for t, n in seen.items() if n > 1)


def _sample_domain(face_tag: int, n: int = 12):
    """Grid-sample the parametric rectangle, keeping only points inside the trim.

    Critical: getParametrizationBounds gives the *untrimmed* rectangle. A trimmed
    face occupies a subset of it, and derivatives outside the trim are unreliable.
    isInside() with parametric=True filters correctly.
    """
    lo, hi = gmsh.model.getParametrizationBounds(2, face_tag)
    us = np.linspace(lo[0], hi[0], n)
    vs = np.linspace(lo[1], hi[1], n)

    inside = []
    for u in us:
        for v in vs:
            if gmsh.model.isInside(2, face_tag, [u, v], parametric=True):
                inside.append((u, v))
    return (lo, hi), inside


def analyze_face(face_tag: int, n_grid: int = 12,
                 th: Thresholds = DEFAULTS) -> FaceParametrization | None:
    (lo, hi), uv = _sample_domain(face_tag, n_grid)
    seams = _seam_curves(face_tag)

    if len(uv) < 4:
        return FaceParametrization(
            tag=face_tag, param_class=ParamClass.UNPARAMETRIZABLE,
            u_range=(lo[0], hi[0]), v_range=(lo[1], hi[1]),
            n_samples_in_domain=len(uv),
            area_scale_min=0.0, area_scale_max=0.0, area_scale_ratio=math.inf,
            anisotropy_median=math.inf, anisotropy_max=math.inf,
            n_degenerate_samples=len(uv), seam_curve_tags=seams,
        )

    flat = [c for p in uv for c in p]
    # VERIFY ON FIRST RUN: for dim=2 this should return 6 floats per point,
    # ordered [dXdu(3), dXdv(3)]. Assert the length so a layout change is loud.
    d = gmsh.model.getDerivative(2, face_tag, flat)
    assert len(d) == 6 * len(uv), (
        f"unexpected getDerivative layout: {len(d)} values for {len(uv)} points"
    )
    d = np.asarray(d, dtype=float).reshape(len(uv), 6)
    du, dv = d[:, 0:3], d[:, 3:6]

    area_scale, anisotropy = metric_from_derivatives(du, dv)

    med = float(np.median(area_scale[area_scale > 0])) if np.any(area_scale > 0) else 0.0
    n_degen = int(np.sum(area_scale <= th.degenerate_det_rel * max(med, 1e-300)))
    pos = area_scale[area_scale > 0]
    a_min = float(pos.min()) if pos.size else 0.0
    a_max = float(area_scale.max())
    ratio = (a_max / a_min) if a_min > 0 else math.inf

    finite_aniso = anisotropy[np.isfinite(anisotropy)]
    aniso_med = float(np.median(finite_aniso)) if finite_aniso.size else math.inf
    aniso_max = float(finite_aniso.max()) if finite_aniso.size else math.inf

    if n_degen > 0:
        cls = ParamClass.DEGENERATE
    elif ratio > th.distortion_ratio_warn or aniso_max > th.anisotropy_warn:
        cls = ParamClass.DISTORTED
    elif seams:
        cls = ParamClass.SEAMED
    else:
        cls = ParamClass.CLEAN

    return FaceParametrization(
        tag=face_tag, param_class=cls,
        u_range=(lo[0], hi[0]), v_range=(lo[1], hi[1]),
        n_samples_in_domain=len(uv),
        area_scale_min=a_min, area_scale_max=a_max, area_scale_ratio=ratio,
        anisotropy_median=aniso_med, anisotropy_max=aniso_max,
        n_degenerate_samples=n_degen, seam_curve_tags=seams,
    )


def analyze(report: Stage0Report, th: Thresholds = DEFAULTS) -> None:
    for _, tag in gmsh.model.getEntities(2):
        fp = analyze_face(tag, th.param_grid, th)
        if fp is None:
            continue
        report.parametrizations.append(fp)

        if fp.param_class is ParamClass.UNPARAMETRIZABLE:
            report.add("param.unparametrizable", Severity.BLOCK,
                       f"only {fp.n_samples_in_domain} valid sample(s) in the "
                       "parametric domain; needs discrete reparametrization",
                       dim=2, tag=tag)
        elif fp.param_class is ParamClass.DEGENERATE:
            report.add("param.degenerate", Severity.WARN,
                       f"metric collapses at {fp.n_degenerate_samples} sample(s) "
                       "(pole/apex); use MeshAdapt (Mesh.Algorithm=1) on this face",
                       dim=2, tag=tag)
        elif fp.param_class is ParamClass.DISTORTED:
            report.add("param.distorted", Severity.WARN,
                       f"area-scale ratio {fp.area_scale_ratio:.3g}, max anisotropy "
                       f"{fp.anisotropy_max:.3g}; uniform (u,v) sizing will map to "
                       "very non-uniform 3D elements",
                       dim=2, tag=tag,
                       threshold_name="distortion_ratio_warn",
                       threshold=th.distortion_ratio_warn,
                       measured=float(fp.area_scale_ratio))
