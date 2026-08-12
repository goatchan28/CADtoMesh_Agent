"""
Stage 0 driver.

    python -m stage0.run part.step --json out/part.stage0.json
    python -m stage0.run part.step --heal --config thresholds.toml

Order matters. Audit first (a blocking CAD defect makes everything after it
meaningless), then parametrization, then 1D, then thickness.

Healing is OPT-IN via --heal. See geometry_audit.import_cad() for why: forcing
repair on import destroyed valid spheres and touching-solid assemblies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gmsh

from . import boundary_discretization, geometry_audit, parametric_map, thickness
from .config import Thresholds, DEFAULTS
from .report import Stage0Report


def run(path: str, *, tolerance: float | None = None, verbose: bool = False,
        heal: bool = False, skip_thickness: bool = False,
        th: Thresholds = DEFAULTS) -> Stage0Report:
    report = Stage0Report(
        source_path=str(path),
        thresholds=th.to_dict(),   # provenance: the report records its own config
        healed=heal,
        tolerance=tolerance,
    )

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Geometry.ReparamOnFaceRobust", 1)

        geometry_audit.import_cad(path, tolerance=tolerance, heal=heal)
        geometry_audit.enforce_conformal_assembly(report)
        geometry_audit.audit(report, th)

        if report.blocking:
            # Stop here. Parametrizing a broken solid produces noise that buries
            # the real finding, which is the failure mode we are avoiding.
            return report

        parametric_map.analyze(report, th)
        boundary_discretization.discretize(report, th)

        if not skip_thickness:
            gmsh.model.mesh.clear()
            thickness.measure(report, th)
    finally:
        gmsh.finalize()

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0 CAD geometry audit")
    ap.add_argument("cad", help="STEP/IGES/BREP file")
    ap.add_argument("--json", help="write full report here")
    ap.add_argument("--config", help="TOML file overriding thresholds")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="OCC Geometry.Tolerance override")
    ap.add_argument("--heal", action="store_true",
                    help="enable OCC shape healing. OFF by default: healing "
                         "strips a sphere's degenerate pole edges and sews "
                         "coincident faces of touching solids, destroying valid "
                         "solids. Use only when a raw import shows a real defect.")
    ap.add_argument("--skip-thickness", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    th = Thresholds.from_toml(args.config) if args.config else DEFAULTS

    report = run(args.cad, tolerance=args.tolerance, verbose=args.verbose,
                 heal=args.heal, skip_thickness=args.skip_thickness, th=th)

    print(report.summary())
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(report.to_json())
        print(f"\nfull report -> {args.json}")

    return 2 if report.blocking else 0


if __name__ == "__main__":
    sys.exit(main())
