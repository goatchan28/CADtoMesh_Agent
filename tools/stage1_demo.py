"""
Stage 1 demo: mesh known analytic surfaces, print quality, export for viewing.

    python -m tools.stage1_demo --case all
    python -m tools.stage1_demo --case cylinder --target 1.0
    python -m tools.stage1_demo --case cylinder --target 1.0 --no-linearize

Writes TWO views per case, both with zero extra dependencies:

  out/<case>.stl   3D surface. Open with `gmsh out/<case>.stl` (gmsh is already
                   installed) or `open out/<case>.stl` -- macOS Quick Look renders
                   STL natively.

  out/<case>.svg   The (u,v) triangulation, shaded by shape quality. Open with
                   `open out/<case>.svg` in any browser.

The SVG is the one people skip and shouldn't. This is a PARAMETRIC mesher, so its
failure modes live in parametric space: a folded, overlapping, or non-partitioning
(u,v) triangulation can still map to a 3D mesh that looks perfectly reasonable.
Seeing both is how you tell a real mesh from a plausible-looking one.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from stage1 import quality as Q
from stage1.boundary import (FaceBoundary, Loop, boundary_from_uv_polygon,
                             orient_loops, resample_uv_polygon)
from stage1.cdt_parametric import triangulate
from stage1.surface import CylinderSurface, PlaneSurface, SphereSurface


# ---------------------------------------------------------------------------
# Cases: each returns (surface, FaceBoundary, exact_area or None)
# ---------------------------------------------------------------------------

def case_plane(target):
    s = PlaneSurface()
    poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return s, boundary_from_uv_polygon(
        [resample_uv_polygon(poly, s, target)], s, target), 1.0


def case_lshape(target):
    s = PlaneSurface()
    poly = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    return s, boundary_from_uv_polygon(
        [resample_uv_polygon(poly, s, target)], s, target), 3.0


def case_hole(target):
    s = PlaneSurface()
    outer = resample_uv_polygon([(0, 0), (3, 0), (3, 3), (0, 3)], s, target)
    inner = resample_uv_polygon([(1, 1), (2, 1), (2, 2), (1, 2)], s, target)
    loops = orient_loops([
        Loop(uv=outer, xyz=np.array([s.point(p) for p in outer])),
        Loop(uv=inner, xyz=np.array([s.point(p) for p in inner])),
    ])
    return s, FaceBoundary(face_tag=0, loops=loops, target_size=target), 8.0


def case_cylinder(target):
    R = 10.0
    s = CylinderSurface(radius=R)
    poly = [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)]
    return s, boundary_from_uv_polygon(
        [resample_uv_polygon(poly, s, target)], s, target), 1.0 * R * 5.0


def case_sphere(target):
    """Mid-latitude band: avoids the poles, so the metric stays well conditioned."""
    R = 10.0
    s = SphereSurface(radius=R)
    poly = [(0.0, 0.9), (1.2, 0.9), (1.2, 2.2), (0.0, 2.2)]
    exact = R * R * 1.2 * (math.cos(0.9) - math.cos(2.2))
    return s, boundary_from_uv_polygon(
        [resample_uv_polygon(poly, s, target)], s, target), exact


def case_sphere_pole(target):
    """Reaches to v=0.05, near the pole where det(I) collapses.

    Expect visibly worse quality than the band. This is the case the Direct 3D
    Advancing Front should handle better, and the reason it is worth building.
    """
    R = 10.0
    s = SphereSurface(radius=R)
    poly = [(0.0, 0.05), (1.2, 0.05), (1.2, 1.2), (0.0, 1.2)]
    exact = R * R * 1.2 * (math.cos(0.05) - math.cos(1.2))
    return s, boundary_from_uv_polygon(
        [resample_uv_polygon(poly, s, target)], s, target), exact


CASES = {
    "plane": case_plane,
    "lshape": case_lshape,
    "hole": case_hole,
    "cylinder": case_cylinder,
    "sphere": case_sphere,
    "sphere_pole": case_sphere_pole,
}
DEFAULT_TARGET = {"plane": 0.15, "lshape": 0.3, "hole": 0.25,
                  "cylinder": 1.0, "sphere": 1.5, "sphere_pole": 1.5}


# ---------------------------------------------------------------------------
# Exporters -- no dependencies beyond numpy
# ---------------------------------------------------------------------------

def write_stl(mesh, path: Path, name: str = "mesh") -> None:
    a, b, c = mesh.corners()
    n = Q.triangle_normals(a, b, c)
    lines = [f"solid {name}"]
    for i in range(len(a)):
        lines.append(f"  facet normal {n[i,0]:.6e} {n[i,1]:.6e} {n[i,2]:.6e}")
        lines.append("    outer loop")
        for p in (a[i], b[i], c[i]):
            lines.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines))


def _quality_colour(q: float) -> str:
    """Red (bad) -> yellow -> green (good). q in [0,1], 1.0 == equilateral."""
    q = max(0.0, min(1.0, q))
    if q < 0.5:
        r, g = 220, int(60 + 300 * q)
    else:
        r, g = int(220 - 300 * (q - 0.5)), 210
    return f"rgb({max(0,min(255,r))},{max(0,min(255,g))},70)"


def write_uv_svg(uv: np.ndarray, mesh, path: Path, width: int = 900,
                 max_height: int = 1100) -> None:
    """The (u,v) triangulation, each triangle shaded by its 3D shape quality.

    Shading by the 3D metric while drawing in (u,v) is the point: it shows WHERE in
    parametric space the mesh is geometrically poor. On an anisotropic face the bad
    triangles cluster, and that clustering tells you whether the metric handling is
    working.
    """
    a3, b3, c3 = mesh.corners()
    q = Q.shape_quality(a3, b3, c3)

    lo, hi = uv.min(axis=0), uv.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    pad = 0.04 * float(span.max())
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    height = int(width * span[1] / span[0])
    # Scale UNIFORMLY. A tall/narrow (u,v) domain (a cylinder patch is 1:5 in
    # (u,v) while being 10:5 physically) must not be squashed to fit: non-uniform
    # scaling would distort exactly the triangle shapes this plot exists to show.
    # So cap the overall size by shrinking width instead.
    if height > max_height:
        width = max(120, int(width * max_height / height))
        height = max_height

    def sx(u):
        return (u - lo[0]) / span[0] * width

    def sy(v):
        return height - (v - lo[1]) / span[1] * height   # flip: SVG y grows down

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>']
    for t, qi in zip(mesh.triangles, q):
        pts = " ".join(f"{sx(uv[i,0]):.2f},{sy(uv[i,1]):.2f}" for i in t)
        out.append(f'<polygon points="{pts}" fill="{_quality_colour(float(qi))}" '
                   f'stroke="#333" stroke-width="0.6"/>')
    out.append(f'<text x="8" y="18" font-family="monospace" font-size="13" '
               f'fill="#000">{mesh.method} | {mesh.n_triangles} tris | '
               f'shape min {float(q.min()):.3f} mean {float(q.mean()):.3f} '
               f'| red=poor green=equilateral</text>')
    out.append(f'<text x="8" y="34" font-family="monospace" font-size="11" '
               f'fill="#555">(u,v) domain, uniform scale. shading is 3D shape '
               f'quality, so stretch here = metric anisotropy</text>')
    out.append("</svg>")
    path.write_text("\n".join(out))


# ---------------------------------------------------------------------------

HEADER = (f"{'case':<13}{'tris':>6}{'area err%':>10}{'maxL/lim':>10}"
          f"{'shape min':>10}{'mean':>7}{'minang':>7}{'chord max':>10}"
          f"{'chord/h':>9}{'inv':>4}")


def run_case(name: str, target: float, outdir: Path, linearize: bool,
             quiet: bool = False):
    surface, fb, exact = CASES[name](target)
    res = triangulate(fb, surface, linearize_metric=linearize)
    m = res.mesh
    a, b, c = m.corners()
    rep = Q.evaluate(m)
    area = float(Q.triangle_areas(a, b, c).sum())
    err = abs(area - exact) / exact * 100.0 if exact else float("nan")
    maxL = float(Q.edge_lengths(a, b, c).max())

    suffix = "" if linearize else "_raw"
    outdir.mkdir(parents=True, exist_ok=True)
    stl = outdir / f"{name}{suffix}.stl"
    svg = outdir / f"{name}{suffix}.svg"
    write_stl(m, stl, name)
    write_uv_svg(res.uv, m, svg)

    # Geometric fidelity. Sampled, so cap the cost on large meshes.
    ch = Q.chordal_deviation(m, surface, target_size=target, max_triangles=1500)

    if not quiet:
        print(f"{name + suffix:<13}{m.n_triangles:>6}{err:>10.3f}"
              f"{maxL/(1.3*target):>10.3f}{rep.shape_min:>10.3f}"
              f"{rep.shape_mean:>7.3f}{rep.min_angle:>7.1f}"
              f"{ch.max_deviation:>10.4f}{ch.max_relative:>9.4f}"
              f"{rep.n_inverted:>4}")
    return res, rep, stl, svg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1 CDT demo and exporter")
    ap.add_argument("--case", default="all", choices=sorted(CASES) + ["all"])
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-linearize", action="store_true",
                    help="disable metric normalization, to see the raw-(u,v) "
                         "baseline the linearization fixes")
    args = ap.parse_args(argv)

    names = sorted(CASES) if args.case == "all" else [args.case]
    outdir = Path(args.out)

    print("maxL/lim < 1.0 means the size criterion was met. inv must be 0.")
    print("chord max = max distance from facet to true surface; chord/h is that")
    print("relative to element size -- the scale-free geometric fidelity number.\n")
    print(HEADER)
    print("-" * len(HEADER))
    written = []
    for n in names:
        t = args.target if args.target is not None else DEFAULT_TARGET[n]
        _, _, stl, svg = run_case(n, t, outdir, not args.no_linearize)
        written += [stl, svg]

    print(f"\nwrote {len(written)} files to {outdir}/")
    print(f"  3D : gmsh {written[0]}        (or: open {written[0]})")
    print(f"  uv : open {written[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
