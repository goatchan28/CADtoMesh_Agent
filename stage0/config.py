"""
Central threshold configuration.

Every tunable number in Stage 0 lives here, for two reasons:

1. Thresholds were guesses. They need calibrating against real parts, and you
   cannot calibrate constants scattered across five modules.
2. The values used are written into every report. A report that does not record
   its own thresholds is not reproducible six months later -- you would not know
   whether a finding changed because the geometry changed or because you moved a
   number.

All length/area thresholds are FRACTIONS OF MODEL SCALE (bbox diagonal), never
absolute. A 5 mm bracket and a 5 m weldment share one config.

Override from TOML:

    [thresholds]
    sliver_circularity = 0.02
    tiny_curve_frac = 5e-5

    python -m stage0.run part.step --config thresholds.toml
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    # --- geometry audit ---
    tiny_curve_frac: float = 1e-4
    """Curve length / model scale below which a curve cannot carry a segment."""

    degenerate_curve_frac: float = 1e-9
    """Below this, treat as an intentional degenerate edge (pole/apex), not a defect."""

    sliver_circularity: float = 5e-2
    """Isoperimetric quotient 4*pi*A/P^2. Scale-free shape measure:
        circle          1.000
        square          0.785
        10:1 rectangle  0.260
        60:1 rectangle  0.051   <- roughly the default cut
        100:1 rectangle 0.031
        300:1 rectangle 0.010
    For a rectangle of aspect ratio r, circularity = pi*r/(1+r)^2, so this
    default flags faces worse than about 60:1. Raise it to catch more, lower it
    if long legitimate faces (ribs, flanges) trip it on your parts.

    NOTE: inner loops (holes) count toward perimeter, so a face with a large hole
    reads lower than its outline alone. block_hole's top face is 0.46, still well
    clear of the cut."""

    tiny_face_area_frac: float = 1e-7
    """Area / model scale^2. Catches genuinely SMALL faces. Kept alongside
    circularity because the two failure modes are different: a 0.001 x 0.001
    patch is small but not thin, a 50 x 0.005 rail is thin but not small."""

    # --- parametrization ---
    param_grid: int = 12
    """Samples per axis per face when probing the (u,v) domain."""

    distortion_ratio_warn: float = 100.0
    """max/min of sqrt(det I) across a face."""

    anisotropy_warn: float = 20.0
    """max sqrt(lambda_max/lambda_min) across a face."""

    degenerate_det_rel: float = 1e-12
    """sqrt(det I) below this fraction of the face median counts as collapsed."""

    # --- reference discretization ---
    reference_size_frac: float = 0.04
    """Uniform 1D size as a fraction of model scale. Diagnostic, not a deliverable."""

    min_segments_per_curve: int = 1

    # --- reporting ---
    borderline_factor: float = 2.0
    """A finding whose measured value is within this factor of its threshold is
    marked borderline. Those are the rows where the threshold actually decided the
    outcome, so they are where calibration attention belongs. thin_plate's sliver
    faces measured 4.574e-2 against a 5e-2 cut -- a ratio of 0.91. A slightly
    different default would have reported nothing at all."""

    # --- thickness ---
    thickness_size_frac: float = 0.03
    thickness_max_samples: int = 4000
    min_elements_through_thickness: int = 3

    @classmethod
    def from_toml(cls, path: str | Path) -> "Thresholds":
        data = tomllib.loads(Path(path).read_text())
        section = data.get("thresholds", data)
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ValueError(
                f"unknown threshold(s) in {path}: {sorted(unknown)}. "
                f"valid keys: {sorted(known)}")
        return cls(**section)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULTS = Thresholds()
