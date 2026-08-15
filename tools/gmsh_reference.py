"""
Gmsh's built-in surface mesher as an EXTERNAL PRODUCTION BENCHMARK.

Black box, strictly
-------------------
Gmsh is used only through its public meshing API: build geometry, set a size, call
generate(2), read the resulting nodes and triangles. Its meshing algorithms are not
inspected, ported, or consulted, and nothing here informs the three implementations
in stage1/. The purpose is a yardstick -- "how far off production quality are we" --
not a source.

Gmsh generates its OWN 1D discretization
----------------------------------------
This is a deliberate asymmetry and it must be reported as one. The three stage1
meshers all receive one identical FaceBoundary, which is what makes THEIR mutual
comparison fair. Gmsh instead does what it would do in production: choose its own
boundary node placement from the size field.

So gmsh's element count and quality are not directly attributable to its 2D
algorithm alone -- part of the difference is boundary placement. That is the right
trade for a benchmark (it measures the real tool as shipped) and the wrong trade
for a peer comparison, hence the separate section in the report. The boundary node
count is printed for both so the size of the asymmetry is visible.

Geometry equivalence
--------------------
Each case is rebuilt with OCC primitives that are geometrically identical to the
corresponding analytic surface, so quality and chordal numbers are comparable:

    plane        unit square in z=0
    lshape       the same 6-vertex polygon in z=0
    cylinder     vertical line at radius 10 revolved 1.0 rad about z
    sphere_*     meridian arc of radius 10 revolved 1.2 rad about z

Because the geometry matches, chordal deviation is measured against the SAME
analytic surface object the other three are measured against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from stage1.quality import SurfaceMesh

R = 10.0

# Gmsh 2D algorithm ids, passed straight through as Mesh.Algorithm. Listed so the
# benchmark can be run against more than one production setting; their internals
# are irrelevant here.
ALGORITHMS = {1: "MeshAdapt", 5: "Delaunay", 6: "Frontal-Delaunay",
              11: "QuasiStructuredQuad"}


class GmshUnavailable(RuntimeError):
    pass


@dataclass
class GmshResult:
    mesh: SurfaceMesh
    n_boundary_nodes: int = 0
    algorithm: int = 6
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry builders. Each returns the surface tag(s) to be meshed.
# ---------------------------------------------------------------------------

def _build_plane(gmsh, surface, poly):
    """Planar polygon, vertices taken from the analytic surface."""
    p = [gmsh.model.occ.addPoint(*surface.point(uv)) for uv in poly]
    lines = [gmsh.model.occ.addLine(p[i], p[(i + 1) % len(p)])
             for i in range(len(p))]
    loop = gmsh.model.occ.addCurveLoop(lines)
    return [gmsh.model.occ.addPlaneSurface([loop])]


def _build_cylinder(gmsh, surface, poly):
    """Generator line at u=0, revolved about the cylinder axis.

    Seed points come from surface.point(), NOT from hardcoded coordinates. That
    matters: CylinderSurface builds its basis as e1 = unit(axis x seed), so for a
    z-axis cylinder u=0 lies along +y, not +x. Hardcoding (R,0,0) described a
    cylinder rotated 90 degrees from the one the other three meshers were given --
    still a valid benchmark surface, but not the SAME surface, so the areas and
    chordal numbers would not have been comparable.
    """
    u0 = poly[0][0]
    v_lo = min(v for _, v in poly)
    v_hi = max(v for _, v in poly)
    du = max(u for u, _ in poly) - min(u for u, _ in poly)

    a = gmsh.model.occ.addPoint(*surface.point((u0, v_lo)))
    b = gmsh.model.occ.addPoint(*surface.point((u0, v_hi)))
    line = gmsh.model.occ.addLine(a, b)
    ax = surface.axis
    o = surface.origin
    out = gmsh.model.occ.revolve([(1, line)], *o, *ax, du)
    return [t for d, t in out if d == 2]


def _build_sphere_patch(gmsh, surface, poly):
    """Meridian arc at u=0, revolved about the polar axis.

    The revolve direction may produce the mirror image of the analytic patch when
    the parametrization's u increases clockwise about the axis. That is harmless
    here: a mirrored patch is CONGRUENT, so area, element quality and chordal
    deviation are all identical. Only the surface identity matters, and that is
    guaranteed by seeding from surface.point().
    """
    v_lo = min(v for _, v in poly)
    v_hi = max(v for _, v in poly)
    du = max(u for u, _ in poly) - min(u for u, _ in poly)
    u0 = poly[0][0]

    c = gmsh.model.occ.addPoint(*surface.center)
    a = gmsh.model.occ.addPoint(*surface.point((u0, v_lo)))
    b = gmsh.model.occ.addPoint(*surface.point((u0, v_hi)))
    arc = gmsh.model.occ.addCircleArc(a, c, b)
    out = gmsh.model.occ.revolve([(1, arc)], *surface.center, 0, 0, 1, du)
    return [t for d, t in out if d == 2]


# case -> builder(gmsh, analytic_surface, uv_polygon). The polygon and surface come
# from tools.compare_meshers.CASES, so the reference geometry is derived from the
# very same definition the three stage1 meshers use.
BUILDERS = {
    "plane": _build_plane,
    "lshape": _build_plane,
    "cylinder": _build_cylinder,
    "sphere_band": _build_sphere_patch,
    "sphere_pole": _build_sphere_patch,
}


# ---------------------------------------------------------------------------

def mesh_case(case: str, target: float, surface, poly, *, algorithm: int = 6,
              curvature_adapt: int = 0, verbose: bool = False) -> GmshResult:
    """Mesh one case with gmsh's built-in surface mesher.

    Args:
        surface: the analytic surface the other three meshers used. Its point()
            seeds the OCC geometry so the benchmark runs on the SAME surface.
        poly: the uv polygon defining the patch extent.
        algorithm: Mesh.Algorithm, passed through unmodified.
        curvature_adapt: Mesh.MeshSizeFromCurvature. Default 0 -- UNIFORM sizing,
            matching what the three stage1 meshers do. Leaving gmsh's curvature
            adaptation on would let it spend elements where they help chordal
            accuracy, which is a genuinely better production strategy but would
            compare a size-adaptive mesher against three uniform ones. Set it
            nonzero to see how much that feature is worth.
    """
    try:
        import gmsh
    except ImportError as e:
        raise GmshUnavailable("gmsh is not installed in this environment") from e
    if case not in BUILDERS:
        raise KeyError(f"no gmsh geometry for case {case!r}")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add(f"ref_{case}")
        surfaces = BUILDERS[case](gmsh, surface, poly)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", target)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_adapt)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)

        # generate(2) runs gmsh's own 1D pass first -- that is the point.
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, float).reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(node_tags)}

        tris: list[list[int]] = []
        for stag in surfaces:
            etypes, _, enodes = gmsh.model.mesh.getElements(2, stag)
            for et, nodes in zip(etypes, enodes):
                if et != 2:          # 2 == 3-node triangle; ignore anything else
                    continue
                arr = np.asarray(nodes).reshape(-1, 3)
                tris.extend([[index[int(v)] for v in row] for row in arr])

        # How many nodes gmsh chose to put on the boundary curves. Printed so the
        # 1D asymmetry against the stage1 meshers is explicit rather than hidden.
        bnd_nodes: set[int] = set()
        for _, ctag in gmsh.model.getBoundary([(2, s) for s in surfaces],
                                              combined=True, oriented=False):
            ntags, _, _ = gmsh.model.mesh.getNodes(1, abs(ctag),
                                                   includeBoundary=True)
            bnd_nodes.update(int(t) for t in ntags)

        faces = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        used = sorted({int(v) for t in tris for v in t})
        remap = {v: i for i, v in enumerate(used)}
        mesh = SurfaceMesh(
            vertices=coords[used] if used else np.zeros((0, 3)),
            triangles=np.array([[remap[v] for v in t] for t in tris],
                               dtype=np.int64).reshape(-1, 3),
            method=f"gmsh_algo{algorithm}",
            meta={"case": case, "target_size": target,
                  "algorithm": algorithm,
                  "algorithm_name": ALGORITHMS.get(algorithm, str(algorithm)),
                  "own_1d_discretization": True,
                  "curvature_adapt": curvature_adapt})
        return GmshResult(mesh=mesh, n_boundary_nodes=len(bnd_nodes),
                          algorithm=algorithm, meta=dict(mesh.meta))
    finally:
        gmsh.finalize()


def available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False
