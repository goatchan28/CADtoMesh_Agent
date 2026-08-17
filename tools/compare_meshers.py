"""
Three-way surface-mesher comparison.

    python -m tools.compare_meshers
    python -m tools.compare_meshers --case sphere_pole --target 1.5
    python -m tools.compare_meshers --out out --svg

Runs Parametric CDT, Parametric Advancing Front, and Direct 3D Advancing Front on
the SAME FaceBoundary and scores every result through the same quality functions.

Fairness is enforced, not assumed
---------------------------------
Each mesh records the fingerprint of the boundary it was built from, and this
driver asserts all three match before reporting anything. Without that check a
future change to one mesher could quietly re-derive or re-space the boundary, and
the resulting difference in element count and quality would be indistinguishable
from an algorithmic difference. The three also share: the same target size, the
same apex_height function, the same AFConfig, the same metric linearization, and
the same quality metrics.

Gmsh appears as an EXTERNAL REFERENCE, in a separate section
-----------------------------------------------------------
With --gmsh, gmsh's built-in surface mesher is run on geometrically equivalent OCC
geometry and scored with the same quality functions. It is a black box: only
generate(2) is called and only the output mesh is read.

It is reported apart from the three because it generates its OWN 1D
discretization, so it is not a peer. Part of any difference in its element count
and quality comes from boundary placement rather than from the 2D algorithm, and
folding it into the fair table would quietly break the one guarantee that table
provides. Boundary node counts are printed for both groups so the size of the
asymmetry is visible.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from research import af_direct3d as AF3
from research import af_parametric as AFP
from research import cdt_parametric as CDT
from quality import core as Q
from research.boundary import boundary_from_uv_polygon, resample_uv_polygon
from research.surface import CylinderSurface, PlaneSurface, SphereSurface

R = 10.0


def _sphere_band_area(v0, v1, du=1.2):
    return R * R * du * (math.cos(v0) - math.cos(v1))


CASES = {
    # name: (surface, uv polygon, exact area, default target)
    "plane": (PlaneSurface(), [(0, 0), (1, 0), (1, 1), (0, 1)], 1.0, 0.15),
    "lshape": (PlaneSurface(), [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)],
               3.0, 0.3),
    "cylinder": (CylinderSurface(radius=R), [(0, 0), (1.0, 0), (1.0, 5.0), (0, 5.0)],
                 1.0 * R * 5.0, 1.0),
    "sphere_band": (SphereSurface(radius=R),
                    [(0, 0.9), (1.2, 0.9), (1.2, 2.2), (0, 2.2)],
                    _sphere_band_area(0.9, 2.2), 1.5),
    "sphere_pole": (SphereSurface(radius=R),
                    [(0, 0.05), (1.2, 0.05), (1.2, 1.2), (0, 1.2)],
                    _sphere_band_area(0.05, 1.2), 1.5),
}

METHODS = ("parametric_cdt", "parametric_af", "direct3d_af")

HDR = (f"{'method':<18}{'tris':>6}{'ideal':>7}{'bnd':>5}{'area%':>8}{'shp_min':>9}"
       f"{'mean':>7}{'minang':>8}{'maxL/h':>8}{'chord/h':>9}{'inv':>5}"
       f"{'time':>8}  state")


def score_row(label, mesh, surface, exact, target, elapsed, n_bnd, state):
    """One report line. Identical scoring for our meshers and for gmsh."""
    a, b, c = mesh.corners()
    rep = Q.evaluate(mesh)
    ch = Q.chordal_deviation(mesh, surface, target_size=target, max_triangles=800)
    area_err = abs(float(Q.triangle_areas(a, b, c).sum()) - exact) / exact * 100
    maxL = float(Q.edge_lengths(a, b, c).max()) / target
    ideal = exact / (math.sqrt(3) / 4 * target * target)
    print(f"{label:<18}{mesh.n_triangles:6d}{ideal:7.0f}{n_bnd:5d}{area_err:8.3f}"
          f"{rep.shape_min:9.3f}{rep.shape_mean:7.3f}{rep.min_angle:8.1f}"
          f"{maxL:8.2f}{ch.max_relative:9.4f}{rep.n_inverted:5d}"
          f"{elapsed:7.2f}s  {state}")


def run_method(name: str, fb, surface):
    t0 = time.time()
    if name == "parametric_cdt":
        r = CDT.triangulate(fb, surface)
        stats = None
    elif name == "parametric_af":
        r = AFP.mesh_face(fb, surface)
        stats = r.stats
    else:
        r = AF3.mesh_face(fb, surface)
        stats = r.stats
    return r, stats, time.time() - t0


def report(case: str, target: float, outdir: Path | None, want_svg: bool,
           with_gmsh: bool = False, gmsh_algos=(6,), curvature_adapt: int = 0):
    surface, poly, exact, default_t = CASES[case]
    t = target if target is not None else default_t
    fb = boundary_from_uv_polygon(
        [resample_uv_polygon(poly, surface, t)], surface, t)

    print(f"\n=== {case} | target {t} | boundary fingerprint {fb.fingerprint()}")
    print(f"--- our implementations: SHARED boundary, {fb.n_nodes} nodes (fair) ---")
    print(HDR)
    print("-" * len(HDR))

    rows = []
    for name in METHODS:
        try:
            r, stats, elapsed = run_method(name, fb, surface)
        except ValueError as e:
            print(f"{name:<18}  REFUSED: {e}")
            continue

        fp = r.mesh.meta.get("boundary_fingerprint")
        assert fp == fb.fingerprint(), (
            f"{name} used a DIFFERENT boundary ({fp} vs {fb.fingerprint()}); "
            "the comparison would be meaningless")

        if r.mesh.n_triangles == 0:
            print(f"{name:<18}  EMPTY MESH")
            continue
        state = "closed" if (stats is None or stats.completed) \
            else f"STALL({stats.front_remaining})"
        score_row(name, r.mesh, surface, exact, t, elapsed, fb.n_nodes, state)
        rows.append((name, r, r.mesh))

        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            from tools.research_demo import write_stl, write_uv_svg
            write_stl(r.mesh, outdir / f"{case}__{name}.stl", f"{case}_{name}")
            if want_svg and getattr(r, "uv", None) is not None and len(r.uv):
                write_uv_svg(r.uv, r.mesh, outdir / f"{case}__{name}.svg")

    if with_gmsh:
        _gmsh_section(case, surface, poly, exact, t, gmsh_algos,
                      curvature_adapt, outdir)
    return rows


def _gmsh_section(case, surface, poly, exact, t, algos, curvature_adapt, outdir):
    """External reference. Separate section because gmsh does its own 1D pass."""
    from tools import gmsh_reference as GREF

    print(f"--- external reference: gmsh chooses its OWN boundary "
          f"(not a peer comparison) ---")
    if not GREF.available():
        print("gmsh is not installed; skipping the reference benchmark")
        return
    if case not in GREF.BUILDERS:
        print(f"no equivalent gmsh geometry for case {case!r}; skipping")
        return

    for algo in algos:
        label = f"gmsh:{GREF.ALGORITHMS.get(algo, algo)}"
        try:
            t0 = time.time()
            gr = GREF.mesh_case(case, t, surface, poly, algorithm=algo,
                                curvature_adapt=curvature_adapt)
            elapsed = time.time() - t0
        except Exception as e:                      # black box: report, never crash
            print(f"{label:<18}  FAILED: {type(e).__name__}: {e}")
            continue
        if gr.mesh.n_triangles == 0:
            print(f"{label:<18}  no triangles returned "
                  f"(algorithm {algo} may produce quads)")
            continue
        score_row(label, gr.mesh, surface, exact, t, elapsed,
                  gr.n_boundary_nodes, "n/a")
        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            from tools.research_demo import write_stl
            write_stl(gr.mesh, outdir / f"{case}__gmsh_algo{algo}.stl",
                      f"{case}_gmsh{algo}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare the three surface meshers")
    ap.add_argument("--case", default="all", choices=sorted(CASES) + ["all"])
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--out", default=None, help="write STL/SVG here")
    ap.add_argument("--svg", action="store_true", help="also write (u,v) SVGs")
    ap.add_argument("--gmsh", action="store_true",
                    help="also run gmsh's built-in surface mesher as an external "
                         "reference (black box; it generates its own 1D mesh)")
    ap.add_argument("--gmsh-algo", type=int, action="append", default=None,
                    help="Mesh.Algorithm to benchmark; repeatable. Default 6.")
    ap.add_argument("--gmsh-curvature", type=int, default=0,
                    help="Mesh.MeshSizeFromCurvature for the reference. Default 0 "
                         "(uniform), matching our meshers. Try 12 to see what "
                         "gmsh's curvature adaptation buys on chordal error.")
    args = ap.parse_args(argv)

    names = sorted(CASES) if args.case == "all" else [args.case]
    outdir = Path(args.out) if args.out else None

    print("Our three meshers receive the IDENTICAL FaceBoundary; the driver asserts")
    print("matching fingerprints before reporting.  ideal = area/(sqrt3/4 h^2).")
    print("chord/h is chordal deviation over element size -- geometric fidelity.")
    print("bnd = boundary nodes the mesher worked from.")
    if args.gmsh:
        print("\ngmsh is an EXTERNAL BLACK-BOX REFERENCE in its own section: it")
        print("generates its own 1D discretization, so it is a benchmark, not a peer.")

    algos = tuple(args.gmsh_algo) if args.gmsh_algo else (6,)
    for n in names:
        report(n, args.target, outdir, args.svg, with_gmsh=args.gmsh,
               gmsh_algos=algos, curvature_adapt=args.gmsh_curvature)

    if outdir:
        print(f"\nwrote meshes to {outdir}/  (gmsh {outdir}/<case>__<method>.stl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
