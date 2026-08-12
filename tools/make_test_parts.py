"""
Generate synthetic STEP parts with known-by-construction properties.

    python -m tools.make_test_parts models/

Each part is designed to exercise one thing, so a Stage 0 finding can be checked
against an answer you already know:

  cylinder.step        |Xu| == radius, |Xv| == 1. Verifies getDerivative layout.
                       Also has a seam curve (periodic in u).
  sphere.step          Two pole degeneracies. Must classify DEGENERATE.
  block_hole.step      Chunky, well behaved. Everything should read CLEAN/SEAMED.
  two_blocks.step      Two solids sharing a face. Tests fragment()/imprinting.
  thin_plate.step      1.5 mm wall on a 100 mm plate. Must trip the thickness guard.
  sliver_block.step    Block with a sub-tolerance sliver face. Tests face.sliver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh

R_CYL = 12.0
H_CYL = 40.0
R_SPH = 15.0


def _write(path: Path) -> None:
    gmsh.model.occ.synchronize()
    gmsh.write(str(path))
    print(f"  wrote {path.name}")


def make_cylinder(outdir: Path) -> None:
    """Known: OCC parametrizes a cylinder as (R cos u, R sin u, v).

    So |Xu| == R_CYL everywhere and |Xv| == 1 everywhere. That is the ground
    truth for verifying getDerivative's output layout -- if you get 1 and 12
    instead of 12 and 1, the du/dv blocks are swapped.
    """
    gmsh.model.add("cylinder")
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H_CYL, R_CYL)
    _write(outdir / "cylinder.step")
    gmsh.model.remove()


def make_sphere(outdir: Path) -> None:
    """Known: metric collapses at both poles (area scale ~ sin v)."""
    gmsh.model.add("sphere")
    gmsh.model.occ.addSphere(0, 0, 0, R_SPH)
    _write(outdir / "sphere.step")
    gmsh.model.remove()


def make_block_hole(outdir: Path) -> None:
    """Chunky reference part: 60x40x25 block, through hole, filleted edges."""
    gmsh.model.add("block_hole")
    box = gmsh.model.occ.addBox(0, 0, 0, 60, 40, 25)
    hole = gmsh.model.occ.addCylinder(30, 20, -5, 0, 0, 35, 8)
    out, _ = gmsh.model.occ.cut([(3, box)], [(3, hole)])
    gmsh.model.occ.synchronize()

    vol = out[0][1]
    # Fillet the four vertical outer edges. Selecting by geometry rather than by
    # tag, because tags are not stable across boolean operations.
    edges = gmsh.model.getBoundary(
        gmsh.model.getBoundary([(3, vol)], combined=False, oriented=False),
        combined=False, oriented=False)
    vertical = []
    for _, tag in {(d, abs(t)) for d, t in edges}:
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, tag)
        dz = abs(z1 - z0)
        near_axis = abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6
        on_outer = min(x0, y0) < 1e-6 or x1 > 59.99 or y1 > 39.99
        if near_axis and dz > 24.0 and on_outer:
            vertical.append(tag)
    if vertical:
        try:
            gmsh.model.occ.fillet([vol], vertical, [4.0])
        except Exception as e:      # OCC refuses some edge sets; part is still valid
            print(f"  (fillet skipped: {e})")
    _write(outdir / "block_hole.step")
    gmsh.model.remove()


def make_two_blocks(outdir: Path) -> None:
    """Two solids sharing a coincident face.

    Without fragment()/removeAllDuplicates(), these mesh independently and share
    no nodes -- the silent-wrong-answer case. Stage 0 must report a shared
    interface face after imprinting.
    """
    gmsh.model.add("two_blocks")
    gmsh.model.occ.addBox(0, 0, 0, 30, 30, 20)
    gmsh.model.occ.addBox(30, 0, 0, 30, 30, 20)     # face-to-face at x=30
    _write(outdir / "two_blocks.step")
    gmsh.model.remove()


def make_thin_plate(outdir: Path) -> None:
    """100 x 100 x 1.5 plate. bbox diagonal ~141, so reference h ~5.7.

    p01 thickness (1.5) is far below 3*h, so thickness.check_against_element_size
    must return BLOCK. If it does not, the guard is broken.
    """
    gmsh.model.add("thin_plate")
    gmsh.model.occ.addBox(0, 0, 0, 100, 100, 1.5)
    _write(outdir / "thin_plate.step")
    gmsh.model.remove()


def make_sliver_block(outdir: Path) -> None:
    """Block with a very thin step cut into one face -> sliver faces."""
    gmsh.model.add("sliver_block")
    box = gmsh.model.occ.addBox(0, 0, 0, 50, 50, 30)
    sliver = gmsh.model.occ.addBox(0, 0, 30, 50, 0.005, 0.005)
    out, _ = gmsh.model.occ.fuse([(3, box)], [(3, sliver)])
    _write(outdir / "sliver_block.step")
    gmsh.model.remove()


BUILDERS = {
    "cylinder": make_cylinder,
    "sphere": make_sphere,
    "block_hole": make_block_hole,
    "two_blocks": make_two_blocks,
    "thin_plate": make_thin_plate,
    "sliver_block": make_sliver_block,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic STEP test parts")
    ap.add_argument("outdir", nargs="?", default="models")
    ap.add_argument("--only", choices=sorted(BUILDERS), action="append")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        for name in (args.only or sorted(BUILDERS)):
            print(f"{name}:")
            BUILDERS[name](outdir)
    finally:
        gmsh.finalize()
    print(f"\ndone -> {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
