"""
Exercise the gmsh production backend end to end on real CAD.

    python -m tools.verify_backend models/block_hole.step
    python -m tools.verify_backend models/*.step --json out/backend.json
    python -m tools.verify_backend models/two_blocks.step --fidelity

Runs: import -> (imprint) -> resolve size field -> gmsh surface mesh -> assess ->
attribute per face. Everything the adaptive loop will do, minus the adaptation.

What to check in the output
---------------------------
  size_in_band   Whether gmsh honoured the per-face sizes we requested. This is the
                 calibration this tool exists for: gmsh satisfies a size field
                 APPROXIMATELY, and until the bias is measured, the size_in_band
                 objective threshold is a guess. Expect systematic offset, not noise.

  faces_empty    Any face producing no triangles is a hard problem, not a quality
                 one -- a lost face means lost geometry.

  worst faces    Per-PID attribution. If this is empty or lumps everything into one
                 face, localized adaptation is impossible and the loop can only make
                 global changes.

  watertight     Only meaningful for closed solids. An open sheet legitimately has
                 boundary edges, so it is reported rather than asserted here; the
                 topology contract belongs in the pipeline, not the backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run_one(path: str, *, base_frac: float, algorithm: str, fidelity: bool,
            heal: bool, verbose: bool, budget: int) -> dict:
    from backends.base import MeshRequest, MeshingStrategy, attribute_by_face, worst_faces
    from backends.gmsh_backend import GmshBackend
    from pipeline.sizefield import SizeFieldSpec
    from pipeline.sizefield import estimate_element_count as SFest
    from quality import core as Q
    from quality.criteria import Criteria, evaluate as assess

    out: dict = {"model": path}
    with GmshBackend(verbose=verbose) as be:
        strategy = MeshingStrategy(algorithm=algorithm, heal=heal, imprint=True)
        out["load"] = be.load(path, strategy)
        scale = be.model_scale()
        out["model_scale"] = scale

        spec = SizeFieldSpec.default(scale, base_frac=base_frac)
        out["size_field"] = spec.to_dict()
        out["size_field_fingerprint"] = spec.fingerprint()

        # Measure wall thickness BEFORE resolving, or the thickness rule is inert.
        thickness = be.measure_thickness()
        out["thickness"] = {"n_faces": len(thickness),
                            "min": float(min(thickness.values())) if thickness else None}

        info = be.face_info()
        adj = be.face_adjacency()
        sizes = spec.resolve(info, smooth=False)
        areas = {pid: float(i.get("area") or 0.0) for pid, i in info.items()}
        estimate = SFest(sizes, areas)
        out["estimated_elements"] = estimate
        out["over_budget"] = estimate > budget
        out["n_ramp_faces"] = sum(1 for h in sizes.values()
                                  if h < max(sizes.values()) * 0.999)
        out["requested_sizes"] = {
            "n_faces": len(sizes),
            "min": float(min(sizes.values())) if sizes else None,
            "max": float(max(sizes.values())) if sizes else None,
            "curved_faces": sum(1 for i in info.values()
                                if (i.get("curvature_max") or 0) > 0),
        }

        result = be.mesh(MeshRequest(spec, strategy))
        out["mesh"] = result.to_report()

        mesh = result.mesh
        if mesh.n_triangles == 0:
            out["error"] = "no triangles produced"
            return out

        # Score against the REQUESTED per-face size, not a global nominal.
        target = result.provenance.size_field_callable()
        crit = Criteria(require_watertight=False)
        surface_for = be.surface_for if fidelity else None
        verdict = assess(mesh, criteria=crit, target_size=target,
                         interface_triangles=result.provenance.interface_triangles())
        out["verdict"] = {
            "is_valid": verdict.is_valid, "ready": verdict.ready,
            "score": verdict.score,
            "failures": [{"code": f.code, "count": f.count} for f in verdict.failures],
            "objectives": [{"name": o.name, "value": o.value, "target": o.target,
                            "satisfied": o.satisfied} for o in verdict.objectives],
        }
        topo = Q.topology_checks(mesh,
                                 result.provenance.interface_triangles())
        out["interface_faces"] = len(result.provenance.interface_pids)
        out["topology"] = topo

        scores = attribute_by_face(mesh, result.provenance,
                                   surface_for=surface_for,
                                   max_fidelity_samples=60)
        out["n_faces_attributed"] = len(scores)
        out["worst_by_shape"] = [s.to_dict() for s in scores[:5]]
        if fidelity:
            out["worst_by_normal"] = worst_faces(scores, "normal_p99_deg", 5)
            out["worst_by_chordal"] = worst_faces(scores, "chordal_relative", 5)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify the gmsh meshing backend")
    ap.add_argument("cad", nargs="+")
    ap.add_argument("--base-frac", type=float, default=0.05)
    ap.add_argument("--algorithm", default="frontal",
                    choices=("meshadapt", "delaunay", "frontal", "default"))
    ap.add_argument("--heal", action="store_true")
    ap.add_argument("--fidelity", action="store_true",
                    help="also measure chordal/normal deviation against the CAD "
                         "faces. Slow: one OCC projection per sample.")
    ap.add_argument("--budget", type=int, default=250_000,
                    help="element budget. A size field implying more than this is "
                         "reported as a decision rather than silently meshed.")
    ap.add_argument("--json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        import gmsh  # noqa: F401
    except ImportError:
        print("gmsh is not installed; this tool needs a real OCC kernel")
        return 1

    hdr = (f"{'model':<16}{'faces':>6}{'ifc':>4}{'tris':>7}{'empty':>6}{'valid':>6}"
           f"{'score':>7}{'shp_min':>8}{'minang':>7}{'in_band':>8}{'grad':>6}"
           f"{'wtr':>5}{'est':>10}{'a/e':>7}{'sec':>7}")
    print("ifc = internal interface faces; topology is checked on the exterior skin.")
    print("est is a LOWER BOUND: it counts each face at its own size and ignores")
    print("the ramp band that Threshold fields put on neighbouring coarse faces.")
    print("Watch the actual/est ratio -- that is the calibration for --budget.")
    print("in_band = fraction of elements within tolerance of the REQUESTED size.")
    print("That is the calibration number: gmsh honours a size field only")
    print("approximately, and the bias must be measured before the objective")
    print("threshold means anything.\n")
    print(hdr)
    print("-" * len(hdr))

    results = []
    ok = True
    for path in args.cad:
        try:
            r = run_one(path, base_frac=args.base_frac, algorithm=args.algorithm,
                        fidelity=args.fidelity, heal=args.heal,
                        verbose=args.verbose, budget=args.budget)
        except Exception as e:
            print(f"{Path(path).stem:<16}  FAILED: {type(e).__name__}: {e}")
            ok = False
            continue
        results.append(r)
        name = Path(path).stem
        if "error" in r:
            print(f"{name:<16}  {r['error']}")
            ok = False
            continue
        v, m, t = r["verdict"], r["mesh"], r["topology"]
        obj = {o["name"]: o for o in v["objectives"]}

        def val(n):
            return obj[n]["value"] if n in obj else float("nan")

        print(f"{name:<16}{m['faces']:>6}{r.get('interface_faces', 0):>4}"
              f"{m['stats']['n_triangles']:>7}"
              f"{m['stats']['n_faces_empty']:>6}"
              f"{'yes' if v['is_valid'] else 'NO':>6}{v['score']:>7.3f}"
              f"{val('min_shape'):>8.3f}{val('min_angle'):>7.1f}"
              f"{val('size_in_band'):>8.3f}{val('gradation'):>6.2f}"
              f"{'y' if t['is_watertight'] else 'n':>5}"
              f"{r.get('estimated_elements', 0):>10,}"
              f"{(m['stats']['n_triangles'] / max(r.get('estimated_elements', 1), 1)):>7.2f}"
              f"{m['stats']['seconds']:>7.2f}")
        if r.get("over_budget"):
            print(f"{'':<16}  !! size field implies "
                  f"{r['estimated_elements']:,} elements, over the "
                  f"{args.budget:,} budget")
            ok = False
        if not v["is_valid"]:
            for f in v["failures"]:
                print(f"{'':<16}  HARD: {f['code']} x{f['count']}")
            ok = False
        if m["stats"]["n_faces_empty"]:
            print(f"{'':<16}  !! {m['stats']['n_faces_empty']} face(s) lost")
            ok = False

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nfull report -> {args.json}")

    print(f"\n{'PASS' if ok else 'ATTENTION'}: "
          f"{'all models meshed and validated' if ok else 'see the flagged rows'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
