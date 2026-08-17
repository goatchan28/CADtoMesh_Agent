"""
Prove entity-id stability against REAL OCC geometry.

    python -m tools.verify_entity_ids models/block_hole.step
    python -m tools.verify_entity_ids models/two_blocks.step --json out/ids.json
    python -m tools.verify_entity_ids models/*.step

The registry's logic is unit-tested without gmsh (tests/test_entities.py). What
cannot be tested there is whether real OCC operations perturb geometry more than
the fingerprint quantization absorbs. That is what this measures.

Four scenarios, each a question the pipeline depends on:

  A. RE-IMPORT     Same file, fresh session. Ids MUST be 100% preserved, or two
                   runs of the pipeline disagree about what a face is.
  B. TOLERANCE     Geometry.Tolerance changed by 10x. Healing-scale perturbations
                   must not break identity.
  C. HEAL          healShapes() plus the Fix* options. Expect some DELETED (that
                   is healing doing its job); expect no false SAME.
  D. IMPRINT       fragment() on a multi-body assembly. Expect MERGE at coincident
                   faces, with both parents traceable to the child.

The number that matters is `preserved_frac` for A and B: anything below 1.0 there
means the fingerprint is too fragile for a report key, and the quantization
(entities.DEFAULT_DIGITS) needs loosening.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.entities import EntityRegistry, Relation


def _import(path: str, *, tolerance: float | None, heal: bool):
    import gmsh
    gmsh.clear()
    if tolerance is not None:
        gmsh.option.setNumber("Geometry.Tolerance", tolerance)
    flag = 1 if heal else 0
    for opt in ("OCCFixDegenerated", "OCCFixSmallEdges", "OCCFixSmallFaces",
                "OCCSewFaces"):
        gmsh.option.setNumber(f"Geometry.{opt}", flag)
    gmsh.model.occ.importShapes(path)
    if heal:
        gmsh.model.occ.healShapes()
    gmsh.model.occ.synchronize()


def _counts(reg: EntityRegistry, dim: int) -> dict:
    live = reg.live(dim=dim)
    by: dict[str, int] = {}
    for r in live:
        by[r.relation.value] = by.get(r.relation.value, 0) + 1
    dead = [r for r in reg.records.values() if r.dim == dim and not r.alive]
    return {"live": len(live), "dead": len(dead), "by_relation": by,
            "preserved_frac": (by.get("same", 0) / len(live)) if live else 0.0}


def run_case(path: str, dim: int, verbose: bool) -> dict:
    import gmsh
    from pipeline import gmsh_entities as GE

    result: dict = {"model": path, "dim": dim, "scenarios": {}}
    reg = EntityRegistry()

    # --- baseline ---
    _import(path, tolerance=None, heal=False)
    frame = GE.model_frame()
    reg.register(GE.snapshot(dims=(dim,)), frame, "raw")
    base_pids = {r.pid for r in reg.live(dim=dim)}
    result["baseline_entities"] = len(base_pids)

    # --- A: re-import ---
    _import(path, tolerance=None, heal=False)
    reg.register(GE.snapshot(dims=(dim,)), frame, "reimport")
    a = _counts(reg, dim)
    a["identical_pid_set"] = ({r.pid for r in reg.live(dim=dim)} == base_pids)
    result["scenarios"]["A_reimport"] = a

    # --- B: tolerance change ---
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
    _import(path, tolerance=1e-3 * diag, heal=False)
    reg.register(GE.snapshot(dims=(dim,)), frame, "tolerance_10x")
    result["scenarios"]["B_tolerance"] = _counts(reg, dim)

    # --- C: healing ---
    _import(path, tolerance=None, heal=True)
    reg.register(GE.snapshot(dims=(dim,)), frame, "healed")
    result["scenarios"]["C_heal"] = _counts(reg, dim)

    # --- D: imprint (multi-body only) ---
    _import(path, tolerance=None, heal=False)
    vols = gmsh.model.getEntities(3)
    if len(vols) >= 2:
        gmsh.model.occ.fragment(vols, [])
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        reg.register(GE.snapshot(dims=(dim,)), frame, "imprinted")
        d = _counts(reg, dim)
        merges = [r for r in reg.live(dim=dim) if r.relation is Relation.MERGE]
        d["n_merge"] = len(merges)
        d["merge_parents_traceable"] = all(
            all(reg.descendants(p) for p in m.parents) for m in merges)
        result["scenarios"]["D_imprint"] = d
    else:
        result["scenarios"]["D_imprint"] = {"skipped": "single body"}

    result["registry"] = reg.to_dict() if verbose else None
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify CAD entity id stability")
    ap.add_argument("cad", nargs="+")
    ap.add_argument("--dim", type=int, default=2, choices=(1, 2, 3))
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        import gmsh
    except ImportError:
        print("gmsh is not installed; this tool needs a real OCC kernel")
        return 1

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if args.verbose else 0)
    all_results = []
    ok = True
    try:
        hdr = (f"{'model':<22}{'scenario':<16}{'live':>6}{'dead':>6}"
               f"{'same':>6}{'split':>6}{'merge':>6}{'new':>5}{'preserved':>11}")
        print("A and B MUST show preserved = 1.000. C and D may legitimately differ.")
        print(f"\n{hdr}\n{'-' * len(hdr)}")
        for path in args.cad:
            r = run_case(path, args.dim, args.verbose)
            all_results.append(r)
            name = Path(path).stem
            for sc, v in r["scenarios"].items():
                if "skipped" in v:
                    print(f"{name:<22}{sc:<16}  skipped ({v['skipped']})")
                    continue
                b = v["by_relation"]
                print(f"{name:<22}{sc:<16}{v['live']:>6}{v['dead']:>6}"
                      f"{b.get('same', 0):>6}{b.get('split', 0):>6}"
                      f"{b.get('merge', 0):>6}{b.get('created', 0):>5}"
                      f"{v['preserved_frac']:>11.3f}")
                if sc in ("A_reimport", "B_tolerance") and v["preserved_frac"] < 1.0:
                    ok = False
            a = r["scenarios"]["A_reimport"]
            if not a.get("identical_pid_set", False):
                ok = False
                print(f"{name:<22}  !! re-import produced a DIFFERENT id set")
            d = r["scenarios"].get("D_imprint", {})
            if d.get("n_merge", 0) and not d.get("merge_parents_traceable", True):
                ok = False
                print(f"{name:<22}  !! merge parents are not traceable")
    finally:
        gmsh.finalize()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(all_results, indent=2))
        print(f"\nfull report -> {args.json}")

    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{'identity is stable where it must be' if ok else 'see !! lines above'}")
    if not ok:
        print("If A/B fall short, loosen pipeline.entities.DEFAULT_DIGITS and re-run;")
        print("if that does not fix it, OCC is perturbing geometry more than a")
        print("fingerprint can absorb and provenance matching must carry more load.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
