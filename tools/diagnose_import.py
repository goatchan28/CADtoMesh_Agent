"""
Isolate WHERE a solid gets lost: STEP export, or import healing.

    python -m tools.diagnose_import models/sphere.step
    python -m tools.diagnose_import models/two_blocks.step

Both parts reported `topology.no_volume`. Two candidate causes, and they need
different fixes:

  A. Import healing destroys the solid. OCCFixDegenerated strips a sphere's
     degenerate pole edges; OCCSewFaces sews coincident faces of touching solids.
     Fix: do not heal on import (already done -- heal is now opt-in).

  B. STEP export never wrote a solid. Fix: change make_test_parts.py.

This tells them apart by building the shape in memory, counting volumes BEFORE
writing, then re-importing the written file under four option sets. If the
in-memory count is 1 and a raw re-import gives 1, the culprit was healing. If the
in-memory count is 1 but every re-import gives 0, the export dropped it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh

# name -> (FixDegenerated, FixSmallEdges, FixSmallFaces, SewFaces, healShapes)
MODES: dict[str, tuple[int, int, int, int, bool]] = {
    "raw            ": (0, 0, 0, 0, False),
    "healShapes only": (0, 0, 0, 0, True),
    "sew only       ": (0, 0, 0, 1, False),
    "fixdegen only  ": (1, 0, 0, 0, False),
    "full heal      ": (1, 1, 1, 1, True),
}


def counts() -> tuple[int, int, int, int]:
    return tuple(len(gmsh.model.getEntities(d)) for d in (3, 2, 1, 0))


def import_with(path: str, mode: tuple) -> tuple[int, int, int, int]:
    fd, fse, fsf, sew, heal = mode
    gmsh.clear()
    gmsh.option.setNumber("Geometry.OCCFixDegenerated", fd)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", fse)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", fsf)
    gmsh.option.setNumber("Geometry.OCCSewFaces", sew)
    gmsh.model.occ.importShapes(path)
    if heal:
        gmsh.model.occ.healShapes()
    gmsh.model.occ.synchronize()
    return counts()


def in_memory_reference(name: str) -> tuple[int, int, int, int] | None:
    """Rebuild the primitive in memory and count volumes before any file I/O."""
    gmsh.clear()
    if name == "sphere":
        gmsh.model.occ.addSphere(0, 0, 0, 15.0)
    elif name == "two_blocks":
        gmsh.model.occ.addBox(0, 0, 0, 30, 30, 20)
        gmsh.model.occ.addBox(30, 0, 0, 30, 30, 20)
    elif name == "cylinder":
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 40, 12.0)
    else:
        return None
    gmsh.model.occ.synchronize()
    return counts()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Isolate export vs import damage")
    ap.add_argument("cad")
    args = ap.parse_args(argv)

    stem = Path(args.cad).stem
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        ref = in_memory_reference(stem)
        print(f"model: {args.cad}\n")
        print(f"{'stage':<18} {'vol':>4} {'face':>5} {'curve':>6} {'pt':>4}")
        print("-" * 42)
        if ref:
            print(f"{'IN MEMORY (pre-write)':<18} {ref[0]:>4} {ref[1]:>5} "
                  f"{ref[2]:>6} {ref[3]:>4}")
        results: dict[str, tuple] = {}
        for label, mode in MODES.items():
            c = import_with(args.cad, mode)
            results[label.strip()] = c
            print(f"{label:<18} {c[0]:>4} {c[1]:>5} {c[2]:>6} {c[3]:>4}")

        print("\nverdict:")
        raw_vol = results["raw"][0]
        healed_vol = results["full heal"][0]
        if ref is None:
            print("  (no in-memory reference for this part; compare rows above)")
        elif ref[0] > 0 and raw_vol > 0 and healed_vol == 0:
            print("  CAUSE A -- import healing destroyed the solid. The STEP is")
            print("  fine and a raw import recovers it. Already fixed: heal is")
            print("  now opt-in via --heal.")
            for label in ("sew only", "fixdegen only", "healShapes only"):
                if results[label][0] == 0:
                    print(f"    culprit isolated to: {label}")
        elif ref[0] > 0 and raw_vol == 0:
            print("  CAUSE B -- STEP export dropped the solid; every import mode")
            print("  gives 0 volumes although the shape had one in memory.")
            print("  Fix make_test_parts.py, not the importer.")
        elif ref[0] == 0:
            print("  the shape had no volume even IN MEMORY -- the builder in")
            print("  make_test_parts.py is wrong.")
        else:
            print("  no volume loss detected; all modes agree.")
    finally:
        gmsh.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
