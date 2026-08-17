"""
Size adequation and gradation. PURE -- arrays in, numbers out. No gmsh, no surface.

These are the two metrics that no per-element shape measure can see, and both are
things a solver cares about directly.

SIZE ADEQUATION -- did the mesher honour the size field?
    A mesher can produce flawless equilateral triangles at entirely the wrong size.
    Shape quality is scale-invariant, so it reports 0.95 either way. That is a real
    failure mode for advancing front in particular, which grows outward from the
    boundary and can drift: our own parametric AF hit maxL/target = 4.77 near a
    sphere pole while still reporting shape_mean 0.862.

GRADATION -- how fast does element size change between neighbours?
    Every per-element metric is blind to this because it is a property of PAIRS.
    Abrupt size transitions degrade solver conditioning and interpolation accuracy
    even when both elements are individually well shaped. FEM practice generally
    wants adjacent element sizes within about 1.5-2x.

Both are reported as distributions, not single numbers. A max-only report cannot
distinguish one bad element from a systematically wrong mesh, and the adaptive loop
needs to tell those apart: the first is a local fix, the second means the size
field itself is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import SurfaceMesh, edge_lengths, triangle_areas


def element_size(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Representative size of each triangle: RMS edge length.

    RMS rather than mean or max. The size field is a length scale, and RMS is what
    relates cleanly to area for a near-equilateral triangle (area = sqrt(3)/4 * h^2
    exactly when all three edges equal h). Max edge would systematically
    over-report on any anisotropic element; mean under-reports.
    """
    L = edge_lengths(a, b, c)
    return np.sqrt(np.mean(L * L, axis=1))


def size_from_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Equivalent-equilateral edge length from area: h = sqrt(4A/sqrt(3)).

    A useful cross-check on element_size. The two agree for equilateral elements
    and diverge for distorted ones, so a large gap between them is itself a
    distortion signal.
    """
    A = triangle_areas(a, b, c)
    return np.sqrt(np.maximum(A, 0.0) * 4.0 / np.sqrt(3.0))


@dataclass
class SizeReport:
    """How well actual element size matches the requested size field."""
    target_mode: str                  # "uniform" or "field"
    ratio_min: float                  # min(actual/target)
    ratio_max: float
    ratio_mean: float
    ratio_p01: float
    ratio_p99: float
    frac_undersized: float            # ratio < 1/tolerance
    frac_oversized: float             # ratio > tolerance
    frac_in_band: float
    tolerance: float
    n_elements: int
    worst_oversized: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"size: ratio {self.ratio_min:.2f}-{self.ratio_max:.2f} "
                f"(mean {self.ratio_mean:.2f}), {self.frac_in_band * 100:.1f}% "
                f"within {self.tolerance:.2f}x, "
                f"{self.frac_oversized * 100:.1f}% oversized")


def size_adequation(mesh: SurfaceMesh, target, *, tolerance: float = 1.5,
                    n_worst: int = 20) -> SizeReport:
    """Compare actual element size against the requested size.

    Args:
        target: a scalar target size, or a callable taking an (N,3) array of
            element centroids and returning an (N,) array of local target sizes.
            The callable form is what lets this evaluate curvature-adaptive or
            proximity-driven size fields rather than only uniform ones.
        tolerance: an element is "in band" when 1/tolerance <= ratio <= tolerance.

    Note this is deliberately SYMMETRIC. Oversized elements lose accuracy;
    undersized ones waste solve time and, near a size-field discontinuity, also
    wreck gradation. Reporting only the oversized tail would hide a mesher that
    systematically over-refines -- which is exactly what our parametric CDT did
    (1.3-1.6x the ideal element count) while every shape metric looked fine.
    """
    if mesh.n_triangles == 0:
        return SizeReport("uniform", 0, 0, 0, 0, 0, 0, 0, 0, tolerance, 0)

    a, b, c = mesh.corners()
    h = element_size(a, b, c)

    if callable(target):
        centroids = (a + b + c) / 3.0
        t = np.asarray(target(centroids), float).reshape(-1)
        mode = "field"
    else:
        t = np.full(len(h), float(target))
        mode = "uniform"
    t = np.where(t > 0, t, np.nan)
    ratio = h / t

    hi = ratio > tolerance
    lo = ratio < 1.0 / tolerance
    order = np.argsort(ratio)[::-1][:n_worst]
    worst = [{"tri": int(i), "ratio": float(ratio[i]), "size": float(h[i])}
             for i in order]

    return SizeReport(
        target_mode=mode,
        ratio_min=float(np.nanmin(ratio)), ratio_max=float(np.nanmax(ratio)),
        ratio_mean=float(np.nanmean(ratio)),
        ratio_p01=float(np.nanpercentile(ratio, 1.0)),
        ratio_p99=float(np.nanpercentile(ratio, 99.0)),
        frac_undersized=float(np.mean(lo)), frac_oversized=float(np.mean(hi)),
        frac_in_band=float(np.mean(~hi & ~lo)),
        tolerance=float(tolerance), n_elements=len(h), worst_oversized=worst)


@dataclass
class GradationReport:
    """Size ratio across shared edges: how abruptly element size changes."""
    max_ratio: float
    mean_ratio: float
    p99_ratio: float
    frac_above_limit: float
    limit: float
    n_pairs: int
    worst_pairs: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"gradation: max {self.max_ratio:.2f}, p99 {self.p99_ratio:.2f}, "
                f"{self.frac_above_limit * 100:.1f}% of {self.n_pairs} pairs "
                f"above {self.limit:.2f}")


def _edge_adjacency(mesh: SurfaceMesh):
    """Pairs of triangle indices sharing an edge."""
    tris = mesh.triangles
    edge_map: dict[tuple[int, int], list[int]] = {}
    for ti, t in enumerate(tris.tolist()):
        for i, j in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_map.setdefault((i, j) if i < j else (j, i), []).append(ti)
    return [(v[0], v[1]) for v in edge_map.values() if len(v) == 2]


def gradation(mesh: SurfaceMesh, *, limit: float = 1.5,
              n_worst: int = 20) -> GradationReport:
    """Ratio of neighbouring element sizes across every shared edge.

    Always >= 1 by construction (larger over smaller), so 1.0 is a perfectly
    uniform mesh. Measured across shared EDGES rather than shared vertices: an edge
    neighbour is where the solver actually needs continuity of the interpolant, and
    vertex-based adjacency would also flag the benign size change around a
    refinement fan.
    """
    if mesh.n_triangles == 0:
        return GradationReport(1.0, 1.0, 1.0, 0.0, limit, 0)

    a, b, c = mesh.corners()
    h = element_size(a, b, c)
    pairs = _edge_adjacency(mesh)
    if not pairs:
        return GradationReport(1.0, 1.0, 1.0, 0.0, limit, 0)

    p = np.asarray(pairs, dtype=np.int64)
    h1, h2 = h[p[:, 0]], h[p[:, 1]]
    small = np.minimum(h1, h2)
    ratio = np.where(small > 0, np.maximum(h1, h2) / np.where(small > 0, small, 1.0),
                     np.inf)

    order = np.argsort(ratio)[::-1][:n_worst]
    worst = [{"tris": [int(p[i, 0]), int(p[i, 1])], "ratio": float(ratio[i])}
             for i in order]

    finite = ratio[np.isfinite(ratio)]
    return GradationReport(
        max_ratio=float(ratio.max()),
        mean_ratio=float(finite.mean()) if finite.size else float("inf"),
        p99_ratio=float(np.percentile(finite, 99.0)) if finite.size else float("inf"),
        frac_above_limit=float(np.mean(ratio > limit)),
        limit=float(limit), n_pairs=len(pairs), worst_pairs=worst)
