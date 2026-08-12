"""
Stage 0c -- boundary discretization (1D).

Gmsh's pipeline is hierarchical: curves are discretized once, then every adjacent
face's 2D mesh conforms to that shared discretization. That sharing is what makes
a CAD-derived surface mesh watertight by construction. So the 1D mesh is not a
throwaway intermediate -- it is the skeleton the whole surface mesh hangs on.

IMPORTANT SCOPING DECISION
--------------------------
The discretization produced here is a *reference* discretization: fixed, uniform,
documented sizing, no agent involvement. Its purpose is diagnostic -- to expose
curves that cannot carry segments, and to give the agent a baseline element count
to reason about. Stage 1 discards it and re-discretizes with agent-chosen sizing
fields.

Keeping Stage 0 free of agent decisions is what makes it a reproducible
measurement rather than a search step.
"""

from __future__ import annotations

import gmsh
import numpy as np

from .config import Thresholds, DEFAULTS
from .report import Stage0Report, CurveDiscretization, Severity
from .topology import EdgeClass, classify_edge_in_assembly, edge_incidence

# Reference sizing lives in config.Thresholds.reference_size_frac. Deliberately
# coarse: this is an instrument, not a deliverable.


def _face_boundaries() -> dict[int, list[int]]:
    """face tag -> bounding curve tags, DUPLICATES PRESERVED.

    oriented=True is required: it returns a seam curve twice (once per
    orientation), which is what lets topology.classify_edge() tell a seam apart
    from a genuine free edge. Deduplicating here loses that distinction and makes
    every through hole look like an open shell.
    """
    out: dict[int, list[int]] = {}
    for _, ftag in gmsh.model.getEntities(2):
        bnd = gmsh.model.getBoundary([(2, ftag)], combined=False, oriented=True)
        out[ftag] = [abs(t) for _, t in bnd]
    return out


def discretize(report: Stage0Report, th: Thresholds = DEFAULTS) -> None:
    """Run generate(1) with reference sizing and measure the result."""
    size_frac = th.reference_size_frac
    h = size_frac * report.model_scale
    gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.25)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    # Curvature-driven sizing off on purpose -- we want a clean baseline, and
    # curvature adaption is an agent decision in Stage 1.
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

    gmsh.model.mesh.generate(1)

    incidences, adj = edge_incidence(_face_boundaries())
    total_segments = 0

    for _, tag in gmsh.model.getEntities(1):
        length = gmsh.model.occ.getMass(1, tag)
        _, coords, _ = gmsh.model.mesh.getNodes(1, tag, includeBoundary=True)
        etypes, etags, enodes = gmsh.model.mesh.getElements(1, tag)
        n_seg = sum(len(t) for t in etags)
        total_segments += n_seg

        seg_lengths = _segment_lengths(tag)
        n_inc = incidences.get(tag, 0)
        faces = adj.get(tag, [])
        # Cross-reference the imprinted interface list: a curve bounding a
        # conformal interface has 3+ incidences BY CONSTRUCTION (the interface
        # plus one face from each adjoining volume). Incidence counting alone
        # cannot tell that apart from a genuine non-manifold defect.
        edge_class = classify_edge_in_assembly(
            n_inc, len(faces), faces, report.shared_interface_faces)

        cd = CurveDiscretization(
            tag=tag,
            length=length,
            n_segments=n_seg,
            min_segment=float(seg_lengths.min()) if seg_lengths.size else 0.0,
            max_segment=float(seg_lengths.max()) if seg_lengths.size else 0.0,
            adjacent_face_tags=faces,
            is_degenerate=length <= 1e-9 * report.model_scale,
            n_incidences=n_inc,
            edge_class=edge_class.value,
        )
        report.curves.append(cd)

        if cd.is_degenerate:
            continue
        if n_seg < th.min_segments_per_curve:
            report.add("discretization.empty_curve", Severity.BLOCK,
                       f"curve produced {n_seg} segments at reference size "
                       f"{h:.4g}; 2D meshing cannot close a loop through it",
                       dim=1, tag=tag)
        if edge_class is EdgeClass.FREE:
            report.add("discretization.free_edge", Severity.BLOCK,
                       "curve has a single face incidence -- the shell is open "
                       "here, so the surface mesh cannot be watertight",
                       dim=1, tag=tag, data={"faces": faces, "incidences": n_inc})
        elif edge_class is EdgeClass.SEAM:
            # Two incidences on ONE face: a periodic surface closing on itself.
            # Manifold and watertight. Not a defect.
            report.add("discretization.seam_edge", Severity.INFO,
                       f"seam curve of periodic face {faces[0]}; manifold via "
                       "two incidences on one face",
                       dim=1, tag=tag, data={"faces": faces})
        elif edge_class is EdgeClass.INTERFACE:
            report.add("discretization.interface_edge", Severity.INFO,
                       f"{n_inc} face incidences across {len(faces)} faces, at a "
                       f"conformal assembly interface; expected topology",
                       dim=1, tag=tag,
                       data={"faces": faces, "incidences": n_inc,
                             "interface_faces": sorted(
                                 set(faces) & set(report.shared_interface_faces))})
        elif edge_class is EdgeClass.NONMANIFOLD:
            report.add("discretization.nonmanifold_edge", Severity.WARN,
                       f"{n_inc} face incidences across {len(faces)} face(s) "
                       "(non-manifold) and NOT at a known assembly interface -- "
                       "investigate",
                       dim=1, tag=tag, data={"faces": faces, "incidences": n_inc})

    report.add("discretization.reference", Severity.INFO,
               f"reference 1D mesh: {total_segments} segments at h={h:.4g} "
               f"({size_frac:.3g} of model scale)",
               data={"h": h, "total_segments": total_segments})


def _segment_lengths(curve_tag: int) -> np.ndarray:
    """Physical lengths of the line elements on one curve."""
    etypes, etags, enodes = gmsh.model.mesh.getElements(1, curve_tag)
    out: list[float] = []
    for et, nodes in zip(etypes, enodes):
        nnodes = gmsh.model.mesh.getElementProperties(et)[3]
        nodes = np.asarray(nodes).reshape(-1, nnodes)
        for row in nodes:
            # Endpoints are the first two nodes for line elements of any order.
            p0 = np.asarray(gmsh.model.mesh.getNode(row[0])[0])
            p1 = np.asarray(gmsh.model.mesh.getNode(row[1])[0])
            out.append(float(np.linalg.norm(p1 - p0)))
    return np.asarray(out, dtype=float)


def trace_curve_on_face(curve_tag: int, face_tag: int, n: int = 20) -> np.ndarray:
    """Map a curve's 1D parameter range into a face's (u,v) domain.

    This is the bridge from Stage 0c to Stage 0b: the (u,v) traces of a face's
    bounding curves define the 2D domain that a parametric-space algorithm will
    actually mesh.

    WARNING: ambiguous on seam curves of periodic faces (the same 3D curve has
    two distinct (u,v) traces). Check FaceParametrization.seam_curve_tags first.
    Setting Geometry.ReparamOnFaceRobust=1 makes gmsh's handling more tolerant.
    """
    lo, hi = gmsh.model.getParametrizationBounds(1, curve_tag)
    ts = np.linspace(lo[0], hi[0], n)
    uv = gmsh.model.reparametrizeOnSurface(1, curve_tag, list(ts), face_tag)
    return np.asarray(uv, dtype=float).reshape(-1, 2)
