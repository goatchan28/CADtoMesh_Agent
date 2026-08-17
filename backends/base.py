"""
Meshing backend protocol and its request/result types. PURE -- no gmsh.

Defining this separately from the gmsh implementation is not ceremony. It fixes
what a backend is allowed to be asked and what it must report, so that:

  * the adaptive loop is written against a protocol, not against gmsh options
  * the research meshers can be wrapped in the same interface for A/B comparison
  * every result carries the request that produced it, which is what makes a run
    record reproducible rather than merely descriptive

Per-face attribution is here too, because it is pure list arithmetic over
provenance and it is what the adaptation policy actually needs. A whole-mesh verdict
says "chordal error is 0.09"; the policy needs "face PID abc123 owns it", or its
only available action is a global size reduction -- which is a parameter sweep, not
adaptation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Protocol

import numpy as np

from pipeline.sizefield import SizeFieldSpec
from quality.core import SurfaceMesh


CAD_SUFFIXES = frozenset({".step", ".stp", ".iges", ".igs", ".brep"})
"""Formats OCC's importShapes can read. STEP is the tested path."""

MESH_SUFFIXES = frozenset({".stl", ".obj", ".ply", ".off", ".msh", ".vtk", ".3mf"})
NATIVE_CAD_SUFFIXES = frozenset({".sldprt", ".sldasm", ".ipt", ".iam", ".prt",
                                 ".catpart", ".catproduct", ".x_t", ".x_b",
                                 ".sat", ".3dm", ".f3d"})


class UnsupportedFormat(ValueError):
    pass


def check_cad_format(path) -> str:
    """Reject unreadable formats with a NAMED error before OCC is touched.

    Without this a .stl fails somewhere inside importShapes with a bare OCC
    message, which reads like a bug in the pipeline rather than the wrong input.

    Mesh formats are called out separately because the reason is deeper than a
    missing reader: the whole pipeline is built on B-rep. Entity PIDs fingerprint
    face mass and curvature, the size field resolves per CAD face, and contracts
    compare against CAD topology. A triangle soup has no faces for any of that to
    attach to, so a converter would not help.
    """
    from pathlib import Path as _P
    suffix = _P(path).suffix.lower()
    if suffix in CAD_SUFFIXES:
        return suffix
    if suffix in MESH_SUFFIXES:
        raise UnsupportedFormat(
            f"'{suffix}' is a MESH format. This pipeline requires B-rep CAD: "
            "entity identity, per-face sizing and the topology/feature contracts "
            "are all keyed to CAD faces, which a triangle soup does not have. "
            f"Supported: {', '.join(sorted(CAD_SUFFIXES))}")
    if suffix in NATIVE_CAD_SUFFIXES:
        raise UnsupportedFormat(
            f"'{suffix}' is a native CAD format requiring a Parasolid or ACIS "
            "licence that open OpenCASCADE does not have. Export to STEP "
            "(AP214 or AP242) first.")
    raise UnsupportedFormat(
        f"unrecognized extension '{suffix}'. "
        f"Supported: {', '.join(sorted(CAD_SUFFIXES))}")


@dataclass(frozen=True)
class MeshingStrategy:
    """Backend-agnostic meshing knobs.

    Named by intent rather than by any backend's option names, so the loop can
    reason about "more optimization" without knowing what gmsh calls it. The
    backend maps these onto its own settings and records what it actually applied.
    """
    algorithm: str = "default"
    """Opaque backend algorithm selector. The loop treats it as a token to swap,
    never as something to interpret."""
    element_order: int = 1
    optimize_passes: int = 1
    smooth_passes: int = 1
    geometry_tolerance: float | None = None
    heal: bool = False
    heal_tolerance_frac: float = 1e-4
    """Healing tolerance as a FRACTION OF MODEL SCALE.

    gmsh's healShapes() defaults to an absolute 1e-8, which on a 76 mm part is
    500,000x smaller than the 0.005 mm sliver it is being asked to remove -- so
    enabling healing changed literally nothing and the loop reported
    'previous action had NO EFFECT'. Every other threshold in this project is
    scale-relative for exactly this reason."""
    imprint: bool = True
    """Imprint multi-body assemblies so touching solids share nodes. Default on:
    without it the assembly meshes as disconnected parts that look correct and
    transmit no load."""
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeshRequest:
    size_field: SizeFieldSpec
    strategy: MeshingStrategy = field(default_factory=MeshingStrategy)
    dims: tuple[int, ...] = (2,)
    """Surface meshing only for now. Volume meshing is explicitly out of scope."""

    def fingerprint(self) -> str:
        payload = json.dumps({"size": self.size_field.to_dict(),
                              "strategy": self.strategy.to_dict(),
                              "dims": list(self.dims)},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class MeshProvenance:
    """Which CAD entity produced which mesh element.

    tri_pid[i] is the persistent id of the face that owns triangle i. Stored as a
    parallel list rather than embedded in the mesh so SurfaceMesh stays a plain
    geometric container that quality/ can score without knowing about CAD.
    """
    tri_pid: list[str] = field(default_factory=list)
    requested_size: dict[str, float] = field(default_factory=dict)
    """Size actually asked for per face PID. Needed to score size adequation
    against what was REQUESTED rather than against a global nominal -- otherwise a
    correctly curvature-refined face reads as oversized failure."""
    face_tags: dict[str, int] = field(default_factory=dict)
    tri_field_size: list[float] = field(default_factory=list)
    """Per-triangle size the FIELD asked for at that location, ramps included.

    Distinct from requested_size, which is one nominal per face. With spatial
    ramps the two differ: an element sitting in a ramp band on a coarse face is
    correctly finer than that face's nominal. Scoring against the nominal marks
    every ramp element out-of-band -- block_hole's in_band fell 0.991 -> 0.774 for
    that reason alone -- and an adaptive loop optimizing in_band would then be
    fighting its own ramps."""
    interface_pids: tuple[str, ...] = ()
    """Faces bounding two or more volumes -- internal walls of an imprinted
    assembly. They are real mesh faces (volume meshing needs them) but they are NOT
    part of the exterior skin, so manifoldness and watertightness must be checked
    with them excluded."""

    def interface_triangles(self) -> set[int]:
        if not self.interface_pids:
            return set()
        iface = set(self.interface_pids)
        return {i for i, pid in enumerate(self.tri_pid) if pid in iface}

    def pids(self) -> list[str]:
        return sorted(set(self.tri_pid))

    def indices_for(self, pid: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.tri_pid) == pid)

    def size_field_callable(self):
        """A per-element target size function for quality.size_adequation.

        Prefers tri_field_size (ramp-aware) and falls back to the per-face nominal.
        Returns None when nothing was recorded, so the caller can fall back to a
        scalar target instead of silently scoring against zeros.
        """
        if self.tri_field_size and len(self.tri_field_size) == len(self.tri_pid):
            sizes = np.asarray(self.tri_field_size, float)

            def f_field(_centroids):
                return sizes
            return f_field
        if not self.requested_size or not self.tri_pid:
            return None
        sizes = np.array([self.requested_size.get(p, np.nan)
                          for p in self.tri_pid], float)

        def f(_centroids):
            return sizes
        return f


@dataclass
class MeshStats:
    n_triangles: int = 0
    n_vertices: int = 0
    n_faces_meshed: int = 0
    n_faces_empty: int = 0
    seconds: float = 0.0
    backend: str = ""
    algorithm_applied: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeshResult:
    mesh: SurfaceMesh
    provenance: MeshProvenance
    stats: MeshStats
    request_fingerprint: str = ""

    def to_report(self) -> dict:
        """Compact, serializable. The run record stores this, not the mesh."""
        return {"request": self.request_fingerprint,
                "stats": self.stats.to_dict(),
                "faces": len(self.provenance.pids()),
                "requested_size": dict(self.provenance.requested_size)}


class MeshingBackend(Protocol):
    """What the pipeline may ask of any mesher."""

    name: str

    def load(self, path: str, strategy: MeshingStrategy) -> None:
        """Import CAD, optionally heal and imprint, and register entity ids."""

    def face_info(self) -> dict[str, dict]:
        """PID -> {curvature_max, thickness, area, ...} for size-field resolution."""

    def face_adjacency(self) -> dict[str, set[str]]:
        """PID -> neighbouring face PIDs, for gradation smoothing."""

    def mesh(self, request: MeshRequest) -> MeshResult:
        """Generate a surface mesh. Must not mutate the request."""

    def surface_for(self, pid: str):
        """A Surface-protocol object for one face, for fidelity metrics."""

    # -- contract inputs: optional, but a backend without them gets quality
    # -- checks only and cannot detect lost geometry --
    def body_kind_inputs(self) -> dict:
        """n_volumes, n_faces, free_edge_length, interface_pids."""

    def removed_faces(self, model_scale: float | None = None):
        """CAD faces that disappeared, e.g. through healing."""


# ---------------------------------------------------------------------------
# Per-face attribution -- pure
# ---------------------------------------------------------------------------

@dataclass
class FaceScore:
    pid: str
    n_triangles: int
    shape_min: float
    shape_mean: float
    min_angle: float
    size_ratio_max: float = float("nan")
    chordal_relative: float = float("nan")
    normal_p99_deg: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def submesh(mesh: SurfaceMesh, tri_indices) -> SurfaceMesh:
    """Extract the triangles of one face as a standalone mesh.

    Vertices are re-indexed, so the result is scoreable by every quality function
    without special cases.
    """
    idx = np.asarray(tri_indices, dtype=np.int64)
    tris = mesh.triangles[idx]
    used = np.unique(tris)
    remap = {int(v): i for i, v in enumerate(used)}
    return SurfaceMesh(
        vertices=mesh.vertices[used],
        triangles=np.array([[remap[int(v)] for v in t] for t in tris],
                           dtype=np.int64).reshape(-1, 3),
        method=mesh.method, meta=dict(mesh.meta))


def attribute_by_face(mesh: SurfaceMesh, prov: MeshProvenance, *,
                      surface_for=None, max_fidelity_samples: int = 200,
                      n_worst: int = 10) -> list[FaceScore]:
    """Score every face separately, worst shape first.

    surface_for(pid) -> Surface enables the fidelity columns. It is optional
    because projecting against real CAD is expensive: one OCC call per sample, so a
    full model at full sampling is unusable. The per-face budget is small on
    purpose -- attribution needs to RANK faces, not to measure any one of them to
    high precision.
    """
    from quality import core as Q
    from quality import normals as N
    from quality import size as S

    out: list[FaceScore] = []
    for pid in prov.pids():
        idx = prov.indices_for(pid)
        if idx.size == 0:
            continue
        sm = submesh(mesh, idx)
        rep = Q.evaluate(sm)
        fs = FaceScore(pid=pid, n_triangles=sm.n_triangles,
                       shape_min=rep.shape_min, shape_mean=rep.shape_mean,
                       min_angle=rep.min_angle)

        h = prov.requested_size.get(pid)
        if h:
            fs.size_ratio_max = S.size_adequation(sm, h).ratio_max

        if surface_for is not None:
            surf = surface_for(pid)
            if surf is not None:
                ch = Q.chordal_deviation(sm, surf, target_size=h,
                                         max_triangles=max_fidelity_samples)
                fs.chordal_relative = ch.max_relative
                nd = N.normal_deviation(sm, surf,
                                        max_triangles=max_fidelity_samples)
                fs.normal_p99_deg = nd.p99_deg
        out.append(fs)

    out.sort(key=lambda f: f.shape_min)
    return out


def worst_faces(scores: list[FaceScore], metric: str, n: int = 5) -> list[str]:
    """PIDs of the n worst faces by one metric -- the policy's scoping input.

    Localizing an action to these instead of acting globally is the difference
    between adaptation and a parameter sweep.
    """
    higher_is_worse = metric in ("size_ratio_max", "chordal_relative",
                                "normal_p99_deg")
    vals = [(getattr(s, metric), s.pid) for s in scores
            if not np.isnan(getattr(s, metric))]
    if not vals:
        return []
    vals.sort(reverse=higher_is_worse)
    return [pid for _, pid in vals[:n]]
