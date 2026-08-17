"""
Contracts: what the mesh must preserve about the CAD. PURE -- no gmsh.

Two questions that the quality metrics cannot answer, because both compare the mesh
against the INPUT rather than against itself:

  TOPOLOGY   Is the mesh closed when it should be, and open where it should be?
  FEATURE    Is every required CAD face still present, and was anything removed?

Why the topology expectation is DERIVED, not configured
-------------------------------------------------------
"require watertight" as a flag is wrong in both directions. A closed solid that
comes out open is a defect; a sheet body that comes out closed is also a defect, and
a sheet body legitimately has boundary edges. Guessing from the mesh is circular --
the mesh is the thing under test. So the expectation comes from the CAD: a body with
a positive-volume solid must produce a closed skin, a shell body must produce a
boundary that matches its own free edges, and an assembly must do the former per body
with conformal interfaces.

Why removals need APPROVAL
--------------------------
Healing deletes geometry. On sliver_block that was exactly right: two 0.005 mm rail
faces vanished, element count fell 10,386 -> 1,858, and the mesh became valid. But
the run reported "accepted" with the removal recorded only inside a load report.

Nothing distinguished that from healing quietly removing a REAL small fillet. The
output would look identical and the accepted mesh would be of geometry that is not
the input geometry. A removal is therefore a violation until it is approved --
automatically when it is small enough to be unambiguous, explicitly otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum


def circularity(area: float, perimeter: float) -> float:
    """Isoperimetric quotient 4*pi*A/P^2. Same formula stage0 uses for face slivers.

    1.0 for a circle, 0.785 for a square, approaching 0 for a sliver. For a
    rectangle of aspect ratio r it equals pi*r/(1+r)^2.
    """
    if perimeter <= 0.0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def aspect_from_circularity(c: float) -> float:
    """Invert circularity for a rectangle, for readable messages."""
    if c <= 0:
        return math.inf
    if c >= math.pi / 4:
        return 1.0
    b = 2.0 - math.pi / c
    disc = b * b - 4.0
    return 1.0 if disc <= 0 else (-b + math.sqrt(disc)) / 2.0


class BodyKind(str, Enum):
    SOLID = "solid"          # one closed volume
    SHEET = "sheet"          # faces with no volume: legitimately open
    ASSEMBLY = "assembly"    # 2+ volumes, conformal interfaces expected
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Removal:
    """A CAD face that disappeared between snapshots."""
    pid: str
    area: float
    area_frac: float               # area / model_scale^2
    perimeter: float = 0.0
    reason: str = ""
    approved: bool = False
    approved_by: str = ""

    @property
    def circularity(self) -> float:
        return circularity(self.area, self.perimeter)

    @property
    def aspect_ratio(self) -> float:
        return aspect_from_circularity(self.circularity)

    def describe(self) -> str:
        return (f"{self.pid}  area {self.area:.4g} ({self.area_frac:.2e} of "
                f"scale^2)  circularity {self.circularity:.3e} "
                f"(~{self.aspect_ratio:.0f}:1)")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    pids: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "pids": list(self.pids), "detail": dict(self.detail)}


@dataclass(frozen=True)
class TopologyContract:
    body_kind: BodyKind
    n_volumes: int
    require_closed: bool
    expected_free_edge_length: float = 0.0
    """Total length of CAD curves bounding exactly one face. Zero for a solid.

    Compared against the mesh's boundary-edge length rather than an edge COUNT,
    because the count depends on element size while the length does not."""
    interface_pids: tuple[str, ...] = ()
    length_tolerance: float = 0.05
    """Fractional tolerance on boundary length. A discretized boundary is slightly
    shorter than the true curve (chords cut corners), so an exact match is the wrong
    test."""

    @staticmethod
    def derive(n_volumes: int, n_faces: int, free_edge_length: float,
               interface_pids=()) -> "TopologyContract":
        """Decide the expectation from the CAD alone."""
        if n_volumes >= 2:
            kind, closed = BodyKind.ASSEMBLY, True
        elif n_volumes == 1:
            kind, closed = BodyKind.SOLID, True
        elif n_faces > 0:
            kind, closed = BodyKind.SHEET, False
        else:
            kind, closed = BodyKind.UNKNOWN, False
        return TopologyContract(
            body_kind=kind, n_volumes=n_volumes, require_closed=closed,
            expected_free_edge_length=(0.0 if closed else float(free_edge_length)),
            interface_pids=tuple(interface_pids))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["body_kind"] = self.body_kind.value
        return d


@dataclass(frozen=True)
class FeatureContract:
    """Which CAD faces must survive into the mesh, and what may be removed."""
    required_pids: tuple[str, ...] = ()
    removals: tuple[Removal, ...] = ()
    auto_approve_circularity: float = 1e-2
    """Removals SHAPED like slivers are approved automatically.

    Area is the wrong test, and getting this wrong once already cost us. The rail
    faces on sliver_block are 50 x 0.005: aspect ratio 10000:1, but area 0.25 mm^2,
    which is 4.2e-5 of scale^2 and nowhere near "tiny". They are THIN, not SMALL --
    exactly the mistake stage0's face.sliver detector made before area was replaced
    with circularity.

    A face that thin cannot carry an element at any size and is an artifact by
    construction. Threshold 1e-2 is about a 300:1 aspect ratio -- deliberately
    tighter than stage0's 5e-2 detection cut, because auto-approving a REMOVAL is a
    stronger claim than flagging a face for attention. A 100:1 face reads 0.031 and
    still requires a human."""

    auto_approve_area_frac: float = 1e-6
    """Secondary test: removals below this fraction of model_scale^2 are approved
    regardless of shape, because a face that small cannot carry an element either."""

    @staticmethod
    def derive(required_pids, removals=(), auto_approve_area_frac: float = 1e-6,
               auto_approve_circularity: float = 1e-2) -> "FeatureContract":
        out = []
        for r in removals:
            why = ""
            if r.approved:
                why = r.approved_by
            elif r.perimeter > 0 and r.circularity <= auto_approve_circularity:
                why = (f"auto: sliver, circularity {r.circularity:.2e} "
                       f"(~{r.aspect_ratio:.0f}:1)")
            elif r.area_frac <= auto_approve_area_frac:
                why = f"auto: area {r.area_frac:.2e} of scale^2 is below the cut"
            out.append(Removal(pid=r.pid, area=r.area, area_frac=r.area_frac,
                               perimeter=r.perimeter, reason=r.reason,
                               approved=bool(why), approved_by=why))
        return FeatureContract(required_pids=tuple(sorted(required_pids)),
                               removals=tuple(out),
                               auto_approve_area_frac=auto_approve_area_frac,
                               auto_approve_circularity=auto_approve_circularity)

    def approve(self, pid: str, by: str = "operator") -> "FeatureContract":
        """Explicitly acknowledge one removal. Returns a new contract."""
        return FeatureContract(
            required_pids=self.required_pids,
            removals=tuple(
                Removal(r.pid, r.area, r.area_frac, r.perimeter, r.reason, True, by)
                if r.pid == pid else r for r in self.removals),
            auto_approve_area_frac=self.auto_approve_area_frac,
            auto_approve_circularity=self.auto_approve_circularity)

    @property
    def unapproved(self) -> tuple[Removal, ...]:
        return tuple(r for r in self.removals if not r.approved)

    def to_dict(self) -> dict:
        return {"required_pids": list(self.required_pids),
                "auto_approve_area_frac": self.auto_approve_area_frac,
                "auto_approve_circularity": self.auto_approve_circularity,
                "removals": [dict(r.to_dict(), circularity=r.circularity,
                                  aspect_ratio=r.aspect_ratio)
                             for r in self.removals]}


@dataclass
class ContractReport:
    topology: TopologyContract | None = None
    feature: FeatureContract | None = None
    violations: list[Violation] = field(default_factory=list)
    meshed_pids: tuple[str, ...] = ()
    boundary_length: float = 0.0

    @property
    def satisfied(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {"satisfied": self.satisfied,
                "topology": self.topology.to_dict() if self.topology else None,
                "feature": self.feature.to_dict() if self.feature else None,
                "boundary_length": self.boundary_length,
                "n_meshed_faces": len(self.meshed_pids),
                "violations": [v.to_dict() for v in self.violations]}

    def summary(self) -> str:
        if self.satisfied:
            kind = self.topology.body_kind.value if self.topology else "?"
            return f"contracts satisfied ({kind}, {len(self.meshed_pids)} faces)"
        return "contract violations:\n" + "\n".join(
            f"  [{v.code}] {v.message}" for v in self.violations)


# ---------------------------------------------------------------------------

def verify(topology: TopologyContract, feature: FeatureContract, *,
           meshed_pids, boundary_edge_length: float,
           n_nonmanifold_skin: int = 0,
           n_inconsistent_normals: int = 0) -> ContractReport:
    """Check a mesh against both contracts. Pure: takes measurements, not a mesh.

    boundary_edge_length is measured on the EXTERIOR SKIN, with assembly interface
    triangles excluded -- an internal wall's edges are not part of the boundary.
    """
    rep = ContractReport(topology=topology, feature=feature,
                         meshed_pids=tuple(sorted(meshed_pids)),
                         boundary_length=float(boundary_edge_length))
    meshed = set(meshed_pids)

    # --- feature: every required face must have produced elements ---
    missing = sorted(set(feature.required_pids) - meshed)
    if missing:
        rep.violations.append(Violation(
            "contract.missing_face",
            f"{len(missing)} required CAD face(s) produced no elements; that is "
            "lost geometry, not a quality shortfall",
            pids=tuple(missing)))

    # --- feature: removals need approval ---
    unapproved = feature.unapproved
    if unapproved:
        rep.violations.append(Violation(
            "contract.unapproved_removal",
            f"{len(unapproved)} CAD face(s) were removed from the model without "
            "approval; the mesh is of different geometry than the input: "
            + "; ".join(r.describe() for r in unapproved[:4]),
            pids=tuple(r.pid for r in unapproved),
            detail={"removals": [r.to_dict() for r in unapproved]}))

    # --- topology ---
    if topology.require_closed:
        if boundary_edge_length > 0:
            rep.violations.append(Violation(
                "contract.unexpected_boundary",
                f"{topology.body_kind.value} must produce a closed skin, but the "
                f"mesh has open boundary of length {boundary_edge_length:.4g}",
                detail={"boundary_length": boundary_edge_length}))
    else:
        expect = topology.expected_free_edge_length
        if expect > 0:
            err = abs(boundary_edge_length - expect) / expect
            if err > topology.length_tolerance:
                rep.violations.append(Violation(
                    "contract.boundary_mismatch",
                    f"sheet body boundary is {boundary_edge_length:.4g} against an "
                    f"expected {expect:.4g} ({err * 100:.1f}% off); the mesh is not "
                    "open where the CAD is open",
                    detail={"actual": boundary_edge_length, "expected": expect,
                            "relative_error": err}))
        elif boundary_edge_length <= 0:
            rep.violations.append(Violation(
                "contract.boundary_mismatch",
                "sheet body produced a closed mesh, but a shell has free edges",
                detail={"actual": boundary_edge_length}))

    if topology.body_kind is BodyKind.ASSEMBLY:
        unmeshed_iface = sorted(set(topology.interface_pids) - meshed)
        if unmeshed_iface:
            rep.violations.append(Violation(
                "contract.interface_not_conformal",
                f"{len(unmeshed_iface)} imprinted interface face(s) produced no "
                "elements; the bodies are not sharing nodes and the assembly "
                "transmits no load across the joint",
                pids=tuple(unmeshed_iface)))

    if n_nonmanifold_skin > 0:
        rep.violations.append(Violation(
            "contract.nonmanifold_skin",
            f"{n_nonmanifold_skin} non-manifold edge(s) on the exterior skin, "
            "after excluding assembly interfaces",
            detail={"count": n_nonmanifold_skin}))
    if n_inconsistent_normals > 0:
        rep.violations.append(Violation(
            "contract.inconsistent_orientation",
            f"{n_inconsistent_normals} adjacent pair(s) disagree on which way is "
            "out; surface integrals and any outward-normal convention break",
            detail={"count": n_inconsistent_normals}))
    return rep


def as_failures(report: ContractReport):
    """Contract violations as quality.criteria Failure objects.

    Violations are HARD failures, never objectives. A mesh missing a required CAD
    face is not a lower-quality mesh -- it is a mesh of the wrong part, and no
    amount of refinement changes that.
    """
    from quality.criteria import Failure
    return [Failure(code=v.code, message=v.message, count=len(v.pids) or 1,
                    detail=dict(v.detail, pids=list(v.pids)))
            for v in report.violations]
