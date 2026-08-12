"""
Pure differential geometry. No gmsh, no I/O -- so it can be unit-tested against
closed-form surfaces.

Everything here is about the first fundamental form of a parametrization
X(u,v): R^2 -> R^3:

    I = [[E, F], [F, G]],   E = Xu.Xu,  F = Xu.Xv,  G = Xv.Xv

I is the pullback of the Euclidean metric. Its eigenvalues are the squared
singular values of the Jacobian [Xu Xv], so:

    sqrt(det I) = sqrt(EG - F^2)          local area scaling
    sqrt(l_max / l_min)                    local anisotropy (stretch ratio)

Note sqrt(EG - F^2), not sqrt(EG): the -F^2 term is the shear correction. Drop it
and every non-orthogonal parametrization reads as having more area than it does,
which is exactly the case (trimmed NURBS) you most need to be right about.
"""

from __future__ import annotations

import numpy as np


def first_fundamental_form(du: np.ndarray, dv: np.ndarray):
    """E, F, G for a batch of sample points.

    Args:
        du, dv: (N, 3) arrays of dX/du and dX/dv.
    Returns:
        (E, F, G), each shape (N,).
    """
    du = np.atleast_2d(np.asarray(du, dtype=float))
    dv = np.atleast_2d(np.asarray(dv, dtype=float))
    E = np.einsum("ij,ij->i", du, du)
    G = np.einsum("ij,ij->i", dv, dv)
    F = np.einsum("ij,ij->i", du, dv)
    return E, F, G


def metric_from_derivatives(du: np.ndarray, dv: np.ndarray):
    """Local area scaling and anisotropy of the (u,v) -> R^3 map.

    Returns:
        (area_scale, anisotropy), each shape (N,).
        area_scale is 0 where the map degenerates (pole/apex);
        anisotropy is +inf there.
    """
    E, F, G = first_fundamental_form(du, dv)

    det = np.maximum(E * G - F * F, 0.0)      # clamp fp noise, never sqrt(<0)
    area_scale = np.sqrt(det)

    # Eigenvalues of symmetric 2x2 [[E,F],[F,G]], computed stably.
    tr = E + G
    disc = np.sqrt(np.maximum((E - G) ** 2 + 4.0 * F * F, 0.0))
    lam_max = 0.5 * (tr + disc)
    lam_min = 0.5 * (tr - disc)
    lam_min = np.maximum(lam_min, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lam_min > 0.0, lam_max / lam_min, np.inf)
    anisotropy = np.sqrt(ratio)
    return area_scale, anisotropy


def target_size_in_parametric_space(du: np.ndarray, dv: np.ndarray,
                                    target_h_3d: float):
    """Convert a desired 3D element size into (u,v)-space sizes.

    This is the payoff of the 3D->2D reduction: a 2D algorithm working in
    parametric space needs to know that a step of du here is not the same
    physical distance as a step of du there. h_u = target / |Xu|, h_v = target / |Xv|.

    Returns (h_u, h_v), each shape (N,); inf where the corresponding derivative
    vanishes.
    """
    E, _, G = first_fundamental_form(du, dv)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_u = np.where(E > 0, target_h_3d / np.sqrt(E), np.inf)
        h_v = np.where(G > 0, target_h_3d / np.sqrt(G), np.inf)
    return h_u, h_v
