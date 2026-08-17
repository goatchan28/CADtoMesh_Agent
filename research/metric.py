"""
Metric-aware geometry in parametric space. PURE -- no gmsh.

The core problem for both PARAMETRIC methods: (u,v) space is not real space. A
step of 0.1 in u may be 1.2 mm here and 40 mm there, and may not even be
perpendicular to a step in v. A mesher that does Euclidean geometry in (u,v)
produces triangles that are equilateral in (u,v) and stretched garbage in R^3.

The fix is the first fundamental form I = [[E,F],[F,G]] from stage0.metrics. It
gives the squared length of a parametric step d:

    |d|_M^2 = d^T I d

The elegant move is a local change of coordinates. Cholesky-factor I = L L^T;
then

    xi = L^T d      =>      |d|_M = |xi|_2

So inside a small neighbourhood, mapping (u,v) -> xi turns metric geometry into
ORDINARY EUCLIDEAN geometry. Both parametric meshers exploit this: they do their
construction (perpendicular offsets for the front, circumcircles for Delaunay) in
xi where the textbook formulas are correct, then map back to (u,v).

This is what makes the parametric methods legitimately comparable to the direct-3D
method rather than just worse.
"""

from __future__ import annotations

import math

import numpy as np


def metric_tensor(du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """First fundamental form at one point. Returns 2x2 [[E,F],[F,G]]."""
    E = float(np.dot(du, du))
    F = float(np.dot(du, dv))
    G = float(np.dot(dv, dv))
    return np.array([[E, F], [F, G]], dtype=float)


def metric_at(surface, uv) -> np.ndarray:
    du, dv = surface.derivatives(uv)
    return metric_tensor(np.asarray(du, float), np.asarray(dv, float))


def cholesky_frame(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Lt, Lt_inv) with |d|_M == |Lt @ d|_2.

    M = L L^T, so d^T M d = d^T L L^T d = |L^T d|^2. Hence Lt = L^T.

    Degenerate M (a sphere pole, where det I -> 0) has no usable factorization.
    Rather than raising, regularize by clamping the small eigenvalue to a tiny
    fraction of the large one. That lets a parametric mesher LIMP through a pole
    instead of crashing -- and the resulting quality drop is exactly the effect the
    three-way comparison is meant to expose.
    """
    M = np.asarray(M, float)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 0.0)
    wmax = float(w.max())
    if wmax <= 0.0:
        return np.eye(2), np.eye(2)
    floor = 1e-10 * wmax
    w = np.maximum(w, floor)
    sqrt_w = np.sqrt(w)
    # M = V diag(w) V^T, so a valid "L^T" is diag(sqrt(w)) V^T.
    Lt = (V * sqrt_w).T
    Lt_inv = V / sqrt_w
    # Force det(Lt) > 0. eigh's eigenvector matrix may have determinant -1, which
    # would make the change of coordinates a REFLECTION -- silently reversing
    # every triangle's winding and inverting the whole mesh. Any mesher using this
    # frame as a working coordinate system depends on orientation surviving.
    if np.linalg.det(Lt) < 0:
        Lt = Lt.copy()
        Lt_inv = Lt_inv.copy()
        Lt[0, :] *= -1.0
        Lt_inv[:, 0] *= -1.0
    return Lt, Lt_inv


def metric_distance(M: np.ndarray, d: np.ndarray) -> float:
    """sqrt(d^T M d) for a single parametric offset d."""
    d = np.asarray(d, float)
    val = float(d @ np.asarray(M, float) @ d)
    return math.sqrt(max(val, 0.0))


def parametric_step_for_length(M: np.ndarray, direction: np.ndarray,
                              length: float) -> np.ndarray:
    """Scale a parametric direction so it spans `length` in real space."""
    direction = np.asarray(direction, float)
    d = metric_distance(M, direction)
    if d <= 0.0:
        return np.zeros(2)
    return direction * (length / d)


def is_degenerate(M: np.ndarray, rel: float = 1e-8) -> bool:
    """True if the metric has collapsed (pole/apex) at this point."""
    M = np.asarray(M, float)
    w = np.linalg.eigvalsh(M)
    wmax = float(np.max(np.abs(w)))
    if wmax <= 0.0:
        return True
    return float(np.min(w)) <= rel * wmax


def anisotropy(M: np.ndarray) -> float:
    """sqrt(lambda_max/lambda_min). 1.0 isotropic, inf degenerate."""
    w = np.linalg.eigvalsh(np.asarray(M, float))
    lo, hi = float(max(w.min(), 0.0)), float(w.max())
    if lo <= 0.0:
        return math.inf
    return math.sqrt(hi / lo)


class MetricField:
    """Caches the metric on a surface, keyed by rounded (u,v).

    Advancing front and Delaunay refinement both query the metric repeatedly at
    nearly the same locations, and for GmshSurface each query is an OCC call.
    Caching keeps the parametric methods from being unfairly slow relative to the
    direct-3D method purely on evaluation count.
    """

    def __init__(self, surface, cache_digits: int = 9):
        self.surface = surface
        self._digits = cache_digits
        self._cache: dict[tuple, np.ndarray] = {}
        self.n_evals = 0

    def at(self, uv) -> np.ndarray:
        key = (round(float(uv[0]), self._digits), round(float(uv[1]), self._digits))
        M = self._cache.get(key)
        if M is None:
            M = metric_at(self.surface, uv)
            self.n_evals += 1
            self._cache[key] = M
        return M

    def frame_at(self, uv):
        return cholesky_frame(self.at(uv))

    def distance(self, uv_a, uv_b) -> float:
        """Approximate real-space distance between two parametric points.

        Uses the metric at the midpoint -- a one-point quadrature of the true arc
        length. Accurate when the step is small relative to curvature, which is
        the regime a mesher operates in anyway.
        """
        a = np.asarray(uv_a, float)
        b = np.asarray(uv_b, float)
        mid = 0.5 * (a + b)
        return metric_distance(self.at(mid), b - a)

    def chord_distance(self, uv_a, uv_b) -> float:
        """Exact straight-line 3D distance between the two surface points.

        Cheaper to trust than `distance` for validation, but not usable as a
        length measure during construction because it ignores curvature.
        """
        pa = np.asarray(self.surface.point(uv_a), float)
        pb = np.asarray(self.surface.point(uv_b), float)
        return float(np.linalg.norm(pb - pa))
