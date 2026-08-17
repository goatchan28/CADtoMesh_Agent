"""
Run the adaptive surface-meshing loop on real CAD.

    python -m tools.run_pipeline models/block_hole.step
    python -m tools.run_pipeline models/*.step --json out/run.json
    python -m tools.run_pipeline models/sliver_block.step --max-iterations 6 -v

Surface meshing only. Volume meshing, boundary conditions and solver integration
are out of scope.

The loop terminates on quality/criteria.py, which is deterministic. No LLM is
involved anywhere: pipeline.policy generates a closed set of candidate actions and
DeterministicSelector picks among them. The LLMSelector seam exists and is
intentionally unimplemented.

Reading the output
------------------
  accepted     valid AND every objective satisfied -- the mesh is simulation-ready
  escalated    the policy asked for a human decision, e.g. defeaturing. This is a
               SUCCESSFUL diagnosis, not a crash: some geometry cannot be meshed
               without changing the feature contract
  plateau      the score stopped moving; the policy has no lever left
  oscillation  a parameter set came back around

Whatever the outcome, the record reports the BEST VALID iteration, which is often
not the last -- the final iteration is usually the most aggressive edit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_one(path: str, *, budget: int, max_iterations: int, fidelity: bool,
            base_frac: float, verbose: bool, approve: set,
            auto_approve_area_frac: float,
            auto_approve_circularity: float):
    from backends.base import MeshingStrategy
    from backends.gmsh_backend import GmshBackend
    from pipeline.loop import AdaptiveMesher
    from pipeline.sizefield import SizeFieldSpec
    from quality.criteria import Criteria

    written = []
    with GmshBackend(verbose=verbose) as be:
        mesher = AdaptiveMesher(be, criteria=Criteria(), budget=budget,
                                max_iterations=max_iterations, fidelity=fidelity,
                                approved_removals=approve,
                                auto_approve_area_frac=auto_approve_area_frac,
                                auto_approve_circularity=auto_approve_circularity)
        # The spec needs the model scale, which needs the model loaded. The loop
        # builds a default when none is supplied; override only if asked.
        spec = None
        if base_frac is not None:
            be.load(path, MeshingStrategy())
            spec = SizeFieldSpec.default(be.model_scale(), base_frac=base_frac)
        rec = mesher.run(path, spec=spec)
        return rec, mesher.best_result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adaptive surface meshing pipeline")
    ap.add_argument("cad", nargs="+")
    ap.add_argument("--budget", type=int, default=250_000,
                    help="element budget; a size field implying more is refused "
                         "BEFORE meshing")
    ap.add_argument("--max-iterations", type=int, default=8)
    ap.add_argument("--base-frac", type=float, default=None,
                    help="initial base size as a fraction of model scale")
    ap.add_argument("--no-fidelity", action="store_true",
                    help="skip chordal/normal measurement (one OCC projection per "
                         "sample; slow on large models)")
    ap.add_argument("--approve-removal", action="append", metavar="PID",
                    help="explicitly approve a CAD face that healing removed. "
                         "Repeatable. Approving a removal changes the feature "
                         "contract, so it is a decision, not a parameter.")
    ap.add_argument("--auto-approve-area", type=float, default=1e-6,
                    help="removals below this fraction of model_scale^2 are treated "
                         "as artifacts and approved automatically")
    ap.add_argument("--auto-approve-circularity", type=float, default=1e-2,
                    help="removals SHAPED like slivers (4*pi*A/P^2 below this) are "
                         "approved automatically. Default 1e-2 is about 300:1 "
                         "aspect. Area is the wrong test: a 50x0.005 rail is thin, "
                         "not small.")
    ap.add_argument("--out", default="out/meshes",
                    help="write the accepted mesh (.msh + .stl + manifest) here. "
                         "The .msh carries a physical group per CAD face so face "
                         "identity survives into volume meshing.")
    ap.add_argument("--no-stl", action="store_true",
                    help="skip the STL; it is inspection-only and carries no "
                         "face tags")
    ap.add_argument("--json")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="table only, no per-iteration detail")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        import gmsh  # noqa: F401
    except ImportError:
        print("gmsh is not installed; this tool needs a real OCC kernel")
        return 1

    hdr = (f"{'model':<16}{'outcome':<14}{'iters':>6}{'best':>5}{'tris':>9}"
           f"{'score':>7}{'ready':>6}{'sec':>7}")
    print("A run that ends 'escalated' has diagnosed something, not crashed.")
    print("A * on the triangle count means that mesh was INVALID.")
    print("Contracts compare the mesh against the CAD: a missing face or an")
    print("unapproved removal is a HARD failure, not a quality shortfall.\n")
    print(hdr)
    print("-" * len(hdr))

    records, ok = [], True
    for path in args.cad:
        try:
            rec, best_result = run_one(path, budget=args.budget,
                          max_iterations=args.max_iterations,
                          fidelity=not args.no_fidelity,
                          base_frac=args.base_frac, verbose=args.verbose,
                          approve=set(args.approve_removal or ()),
                          auto_approve_area_frac=args.auto_approve_area,
                          auto_approve_circularity=args.auto_approve_circularity)
        except Exception as e:
            print(f"{Path(path).stem:<16}  FAILED: {type(e).__name__}: {e}")
            ok = False
            continue
        records.append(rec.to_dict())

        # Write the mesh. Without this the run validates geometry and throws it
        # away, and nothing downstream can consume the result.
        if best_result is not None and args.out:
            from pipeline.export import export_result
            best = rec.best
            paths = export_result(
                best_result, rec, args.out, Path(path).stem,
                criteria=rec.criteria,
                contract=(best.contract if best else None),
                iteration=(best.index if best else -1),
                write_stl_too=not args.no_stl)
            print(f"    -> {paths[0]}")
        best = rec.best
        # With no valid mesh, report the last attempt's size rather than 0 --
        # "produced 10,386 invalid triangles" and "produced nothing" are different
        # diagnoses. A trailing * marks that the count is from an invalid mesh.
        if best is not None:
            shown = f"{best.n_triangles:,}"
        elif rec.iterations:
            shown = f"{rec.iterations[-1].n_triangles:,}*"
        else:
            shown = "0"
        print(f"{Path(path).stem:<16}{rec.outcome.value:<14}"
              f"{len(rec.iterations):>6}"
              f"{(best.index if best else -1):>5}"
              f"{shown:>9}"
              f"{(best.score if best else 0.0):>7.3f}"
              f"{('yes' if best and best.ready else 'no'):>6}"
              f"{rec.elapsed:>7.2f}")
        if not args.quiet:
            for line in rec.summary().splitlines()[1:]:
                print(f"    {line}")
        if best is None:
            ok = False

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(records, indent=2, default=str))
        print(f"\nfull record -> {args.json}")

    print(f"\n{'PASS' if ok else 'ATTENTION'}: "
          f"{'every model produced a valid mesh' if ok else 'see the flagged rows'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
