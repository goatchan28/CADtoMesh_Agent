"""
Stage 0 report schema.

This is the contract between Stage 0 (deterministic geometry inspection) and
everything downstream. The agent never sees raw gmsh state -- it sees this.

Design rule: every finding carries an entity tag and a severity, so the agent
can reason about *where* a problem is, not just that one exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from .config import DEFAULTS


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"       # meshable, but quality or element count at risk
    BLOCK = "block"     # will fail or silently produce a wrong model


class ParamClass(str, Enum):
    """How a CAD face's native (u,v) parametrization behaves."""
    CLEAN = "clean"                 # safe for parametric-space 2D algorithms
    SEAMED = "seamed"               # periodic surface; seam curve needs care
    DEGENERATE = "degenerate"       # pole/apex: metric collapses somewhere
    DISTORTED = "distorted"         # usable but badly stretched; reparametrize
    UNPARAMETRIZABLE = "unparam"    # fall back to discrete reparametrization


@dataclass
class Finding:
    code: str
    severity: Severity
    message: str
    dim: int | None = None
    tag: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    # --- threshold provenance ---
    # Recorded so tools/calibrate.py can retro-fit thresholds from saved reports
    # without re-running Stage 0. A finding that cannot say what it tripped and
    # by how much is not calibratable.
    threshold_name: str | None = None
    threshold: float | None = None
    measured: float | None = None

    @property
    def margin(self) -> float | None:
        """measured / threshold. <1 means below the cut, >1 above."""
        if self.threshold in (None, 0) or self.measured is None:
            return None
        return self.measured / self.threshold

    @property
    def borderline(self) -> bool:
        """True when the threshold, not the geometry, decided this finding.

        A margin near 1.0 means a small change to the number would flip the
        result. These rows are where calibration effort pays off; a finding three
        orders of magnitude clear of its cut tells you nothing about the cut.
        """
        m = self.margin
        if m is None or m <= 0:
            return False
        f = DEFAULTS.borderline_factor
        return 1.0 / f <= m <= f


@dataclass
class FaceParametrization:
    tag: int
    param_class: ParamClass
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    n_samples_in_domain: int
    # sqrt(det I) = local area scaling of the (u,v) -> R^3 map
    area_scale_min: float
    area_scale_max: float
    area_scale_ratio: float
    # sqrt(lambda_max / lambda_min) of the first fundamental form
    anisotropy_median: float
    anisotropy_max: float
    n_degenerate_samples: int
    is_periodic_u: bool = False
    is_periodic_v: bool = False
    seam_curve_tags: list[int] = field(default_factory=list)


@dataclass
class CurveDiscretization:
    tag: int
    length: float
    n_segments: int
    min_segment: float
    max_segment: float
    adjacent_face_tags: list[int] = field(default_factory=list)
    is_degenerate: bool = False
    n_incidences: int = 0        # counted WITH multiplicity; a seam gives 2 on 1 face
    edge_class: str = "manifold"  # see topology.EdgeClass


@dataclass
class ThicknessSample:
    x: float
    y: float
    z: float
    thickness: float
    face_tag: int


@dataclass
class Stage0Report:
    source_path: str
    # Thresholds this run used. Without these the report is not reproducible --
    # you could not tell a geometry change from a threshold change.
    thresholds: dict[str, Any] = field(default_factory=dict)
    healed: bool = False
    tolerance: float | None = None
    # Faces bounding 2+ volumes after imprinting. Consumed by the 1D edge check
    # (to tell expected assembly non-manifoldness from a real defect) and by the
    # thickness pass (to exclude internal walls from ray casting).
    shared_interface_faces: list[int] = field(default_factory=list)
    # --- topology / tolerance audit ---
    n_volumes: int = 0
    n_faces: int = 0
    n_curves: int = 0
    n_points: int = 0
    bbox: tuple[float, float, float, float, float, float] | None = None
    total_volume: float = 0.0
    model_scale: float = 0.0          # bbox diagonal; the length unit for all thresholds
    # --- artifacts for downstream stages ---
    parametrizations: list[FaceParametrization] = field(default_factory=list)
    curves: list[CurveDiscretization] = field(default_factory=list)
    thickness_samples: list[ThicknessSample] = field(default_factory=list)
    thickness_p01: float | None = None    # 1st percentile local thickness
    thickness_median: float | None = None
    # --- findings ---
    findings: list[Finding] = field(default_factory=list)

    def add(self, code: str, severity: Severity, message: str, *,
            threshold_name: str | None = None, threshold: float | None = None,
            measured: float | None = None, **kw) -> None:
        self.findings.append(Finding(
            code=code, severity=severity, message=message,
            threshold_name=threshold_name, threshold=threshold,
            measured=measured, **kw))

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.BLOCK]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    def summary(self) -> str:
        """Compact human/LLM-readable digest. Full detail stays in the JSON."""
        lines = [
            f"model: {self.source_path}",
            f"topology: {self.n_volumes} vol / {self.n_faces} face / "
            f"{self.n_curves} curve / {self.n_points} pt",
            f"scale (bbox diagonal): {self.model_scale:.4g}",
            f"import: {'HEALED' if self.healed else 'raw'}"
            + (f", tolerance={self.tolerance:g}" if self.tolerance else ""),
        ]
        if self.thickness_median is not None:
            lines.append(
                f"thickness: median {self.thickness_median:.4g}, "
                f"p01 {self.thickness_p01:.4g}"
            )
        by_class: dict[str, int] = {}
        for p in self.parametrizations:
            by_class[p.param_class.value] = by_class.get(p.param_class.value, 0) + 1
        if by_class:
            lines.append("face parametrization: " + ", ".join(
                f"{k}={v}" for k, v in sorted(by_class.items())))
        n_border = sum(1 for f in self.findings
                       if f.borderline and f.severity != Severity.INFO)
        if n_border:
            lines.append(f"borderline findings: {n_border} (threshold decided the "
                         "outcome; see calibration)")
        for sev in (Severity.BLOCK, Severity.WARN):
            fs = [f for f in self.findings if f.severity == sev]
            if fs:
                lines.append(f"{sev.value.upper()} ({len(fs)}):")
                for f in fs[:15]:
                    loc = f" [{f.dim}D tag {f.tag}]" if f.tag is not None else ""
                    mark = ""
                    if f.borderline:
                        mark = f"  <-- BORDERLINE (margin {f.margin:.2f})"
                    lines.append(f"  {f.code}{loc}: {f.message}{mark}")
                if len(fs) > 15:
                    lines.append(f"  ... {len(fs) - 15} more")
        return "\n".join(lines)
