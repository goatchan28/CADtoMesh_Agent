"""
Backend-independent mesh quality evaluation.

One evaluator scores every mesh in this project -- the gmsh production backend and
the three research meshers alike. That is deliberate and load-bearing: if the
production path were scored by gmsh's own metrics and the research meshers by ours,
normalization differences would be indistinguishable from real quality differences,
and the comparison that justified adopting gmsh could never be re-run as gmsh gets
tuned.

Nothing in this package imports gmsh. Anything needing surface geometry (chordal
deviation, normal deviation) takes an object satisfying the Surface protocol --
.project(), .normal(), .derivatives() -- so it works equally against an analytic
surface or a CAD face.

Four metric families, plus a verdict layer
------------------------------------------
    shape      core.shape_quality, radius_ratio, angles_deg, aspect_ratio
    topology   core.topology_checks -- watertight, manifold, orientation
    fidelity   core.chordal_deviation (distance), normals.normal_deviation (angle)
    size       size.size_adequation (vs target), size.gradation (vs neighbours)

    criteria   the VALIDITY GATE vs OBJECTIVE SCORE split. Read criteria.py before
               using this for anything automated: hard failures make a mesh
               unusable and must never be traded against a score.

Closed-form references, for validating the metrics rather than trusting them:
    core.sagitta(R, chord)              exact chordal deviation on a circle
    normals.facet_angle_bound(R, chord) exact facet-to-surface angle
"""

from . import core, criteria, normals, size

# ---- mesh container and geometry ----
from .core import (SurfaceMesh, edge_lengths, triangle_areas, triangle_normals)
# NB: corners() is a SurfaceMesh METHOD (mesh.corners()), not a module function.

# ---- shape metrics ----
from .core import (QualityReport, angles_deg, aspect_ratio, radius_ratio,
                   shape_quality)
from .core import evaluate as evaluate_shape

# ---- topology ----
from .core import topology_checks

# ---- fidelity ----
from .core import DEFAULT_BARYCENTRIC, ChordalReport, chordal_deviation, sagitta
from .normals import NormalReport, facet_angle_bound, normal_deviation

# ---- size ----
from .size import (GradationReport, SizeReport, element_size, gradation,
                   size_adequation, size_from_area)

# ---- verdict ----
from .criteria import Criteria, Failure, Kind, Objective, Verdict
from .criteria import evaluate as assess

__all__ = [
    "core", "criteria", "normals", "size",
    "SurfaceMesh", "edge_lengths", "triangle_areas", "triangle_normals",
    "QualityReport", "shape_quality", "radius_ratio", "angles_deg", "aspect_ratio",
    "evaluate_shape", "topology_checks",
    "ChordalReport", "chordal_deviation", "sagitta", "DEFAULT_BARYCENTRIC",
    "NormalReport", "normal_deviation", "facet_angle_bound",
    "SizeReport", "GradationReport", "size_adequation", "gradation",
    "element_size", "size_from_area",
    "Criteria", "Verdict", "Objective", "Failure", "Kind", "assess",
]
