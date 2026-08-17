"""
Extract backend-neutral EntityDescriptors from a live gmsh/OCC model.

This is the only file that knows both gmsh and pipeline.entities. Everything the
registry does is pure and testable; this thin adapter is what cannot be tested
without a real OCC kernel, so it is kept as small as possible and its assumptions
are stated explicitly.

Gmsh API used (all public, all read-only):
    getEntities, getBoundingBox, getBoundary
    occ.getMass, occ.getCenterOfMass
    getNormal, getParametrizationBounds, isInside   (faces only)
"""

from __future__ import annotations

import numpy as np

from .entities import EntityDescriptor, ModelFrame


def model_frame() -> ModelFrame:
    """Normalization frame from the whole model's bounding box.

    Capture this ONCE, from the first snapshot, and pass the same frame to every
    later registration -- see EntityRegistry.register.
    """
    import gmsh
    return ModelFrame.from_bbox(*gmsh.model.getBoundingBox(-1, -1))


def _face_normal_signature(face_tag: int, n_samples: int = 5) -> tuple[float, ...]:
    """Mean unit normal over the face, as a coincidence discriminator.

    Two solids meeting at a shared face produce faces with identical mass,
    centroid and bbox. Their OUTWARD normals are opposite, and that is the only
    geometric quantity that separates them. Without it the registry has to fall
    back on ambiguity deferral, which still works but records less.

    Returns () when the face has no usable parametric domain rather than guessing;
    a wrong normal is worse than an absent one.
    """
    import gmsh
    try:
        lo, hi = gmsh.model.getParametrizationBounds(2, face_tag)
    except Exception:
        return ()
    us = np.linspace(lo[0], hi[0], n_samples + 2)[1:-1]
    vs = np.linspace(lo[1], hi[1], n_samples + 2)[1:-1]

    acc = np.zeros(3)
    n_ok = 0
    for u in us:
        for v in vs:
            try:
                if not gmsh.model.isInside(2, face_tag, [float(u), float(v)],
                                           parametric=True):
                    continue
                n = np.asarray(gmsh.model.getNormal(face_tag, [float(u), float(v)]),
                               float).reshape(-1, 3)[0]
            except Exception:
                continue
            L = float(np.linalg.norm(n))
            if L > 0:
                acc += n / L
                n_ok += 1
    if n_ok == 0:
        return ()
    m = acc / n_ok
    L = float(np.linalg.norm(m))
    if L < 1e-9:
        return ()                       # normals cancel: a closed or folded face
    return tuple(np.round(m / L, 6))


def _curvature_signature(face_tag: int, n_samples: int = 3) -> tuple[float, ...]:
    """Mean absolute principal curvatures, as a shape discriminator.

    Separates a planar face from a cylindrical one of equal area and centroid.
    Reported unnormalized; fingerprint() multiplies by model scale to make it
    dimensionless.
    """
    import gmsh
    try:
        lo, hi = gmsh.model.getParametrizationBounds(2, face_tag)
    except Exception:
        return ()
    pts: list[float] = []
    for u in np.linspace(lo[0], hi[0], n_samples + 2)[1:-1]:
        for v in np.linspace(lo[1], hi[1], n_samples + 2)[1:-1]:
            try:
                if gmsh.model.isInside(2, face_tag, [float(u), float(v)],
                                       parametric=True):
                    pts.extend([float(u), float(v)])
            except Exception:
                pass
    if not pts:
        return ()
    try:
        cmax, cmin, _, _ = gmsh.model.getPrincipalCurvatures(face_tag, pts)
    except Exception:
        return ()
    a = np.abs(np.asarray(cmax, float))
    b = np.abs(np.asarray(cmin, float))
    if a.size == 0:
        return ()
    return (float(np.round(a.mean(), 9)), float(np.round(b.mean(), 9)))


def snapshot(dims=(1, 2, 3), *, with_normals: bool = True,
             with_curvature: bool = True) -> list[EntityDescriptor]:
    """Describe every entity of the given dimensions in the current model.

    Vertices (dim 0) are excluded by default: they carry no mass, so their identity
    reduces to position alone, and the pipeline references faces and edges.
    """
    import gmsh
    out: list[EntityDescriptor] = []
    for dim in dims:
        for _, tag in gmsh.model.getEntities(dim):
            try:
                mass = float(gmsh.model.occ.getMass(dim, tag))
                com = tuple(float(x) for x in gmsh.model.occ.getCenterOfMass(dim, tag))
            except Exception:
                continue
            bbox = tuple(float(x) for x in gmsh.model.getBoundingBox(dim, tag))
            try:
                nb = len(gmsh.model.getBoundary([(dim, tag)], combined=False,
                                                oriented=False))
            except Exception:
                nb = 0
            perim = 0.0
            if dim == 2:
                try:
                    bnd = gmsh.model.getBoundary([(dim, tag)], combined=False,
                                                 oriented=False)
                    # Distinct curves: a seam appears twice and double-counting it
                    # would deflate circularity and fake a sliver.
                    perim = sum(float(gmsh.model.occ.getMass(1, c))
                                for c in {abs(c) for _, c in bnd})
                except Exception:
                    perim = 0.0
            curv = nrm = ()
            if dim == 2:
                if with_curvature:
                    curv = _curvature_signature(tag)
                if with_normals:
                    nrm = _face_normal_signature(tag)
            out.append(EntityDescriptor(dim=dim, tag=int(tag), mass=mass,
                                        centroid=com, bbox=bbox, n_boundary=nb,
                                        curvature_sig=curv, normal_sig=nrm,
                                        perimeter=perim))
    return out
