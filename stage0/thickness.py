"""
Stage 0d -- local thickness detector.

Purpose is narrow: refuse to be silently wrong. Your parts are mostly chunky, so
this is not a pipeline branch -- it is a guard. One tet through a wall meshes
fine, solves fine, and reports bending stiffness that is wrong by an order of
magnitude. Nothing downstream catches that, so it has to be caught here.

Method: cast a ray inward along the reversed surface normal from sample points on
a coarse surface mesh; local thickness = distance to first hit. A small cone of
rays guards against grazing incidence near fillets, taking the median.

Needs a discretization, so it runs a coarse generate(2). That mesh is thrown away
-- accuracy of a few percent is plenty for a threshold check.
"""

from __future__ import annotations

import gmsh
import numpy as np
import trimesh

from .config import Thresholds, DEFAULTS
from .geometry_audit import face_volume_owners
from .report import Stage0Report, ThicknessSample, Severity
from .topology import external_faces

CONE_HALF_ANGLE = 0.12      # radians; ~7 degrees
CONE_RAYS = 5
EPS_FRAC = 1e-5             # ray origin offset, fraction of model scale


def _coarse_surface_mesh(size_frac: float = 0.03, external_only: bool = True):
    """Coarse triangulated surface. Returns (vertices, faces, face_tag_per_tri,
    excluded_face_tags).

    external_only=True keeps ONLY faces bounding a single volume -- the outer skin.

    This matters after imprinting. fragment() turns a shared assembly interface
    into a real face sitting inside the material, and it lands in the surface mesh
    like any other. Inward rays then stop at that internal wall, so two bonded
    30 mm blocks report a 30 mm thickness for a 60 mm continuous region. It also
    breaks the watertightness check, because the interface face leaves the
    triangulation non-manifold along its edges.

    Excluding internal faces fixes both: the outer skin of a conformally bonded
    region is watertight, and rays traverse the full material.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
    h = size_frac * diag
    gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = {int(t): i for i, t in enumerate(node_tags)}

    all_faces = [t for _, t in gmsh.model.getEntities(2)]
    if external_only:
        keep = set(external_faces(face_volume_owners()))
        # No volumes (surface-only import) leaves keep empty -- fall back to all
        # faces rather than producing an empty mesh.
        if not keep:
            keep = set(all_faces)
    else:
        keep = set(all_faces)
    excluded = sorted(t for t in all_faces if t not in keep)

    tris: list[list[int]] = []
    tri_face: list[int] = []
    for ftag in all_faces:
        if ftag not in keep:
            continue
        etypes, etags, enodes = gmsh.model.mesh.getElements(2, ftag)
        for et, nodes in zip(etypes, enodes):
            if et != 2:          # 2 == 3-node triangle
                continue
            nodes = np.asarray(nodes).reshape(-1, 3)
            for row in nodes:
                tris.append([index[int(n)] for n in row])
                tri_face.append(ftag)
    return (coords, np.asarray(tris, dtype=np.int64),
            np.asarray(tri_face, dtype=np.int64), excluded)


def _cone_directions(normal: np.ndarray, n: int, half_angle: float) -> np.ndarray:
    """n unit directions clustered around `normal` within `half_angle`."""
    normal = normal / np.linalg.norm(normal)
    # Any vector not parallel to normal gives us a tangent basis.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, normal)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, seed)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    dirs = [normal]
    for k in range(n - 1):
        phi = 2.0 * np.pi * k / max(n - 1, 1)
        d = (np.cos(half_angle) * normal
             + np.sin(half_angle) * (np.cos(phi) * t1 + np.sin(phi) * t2))
        dirs.append(d / np.linalg.norm(d))
    return np.asarray(dirs)


def measure(report: Stage0Report, th: Thresholds = DEFAULTS) -> None:
    max_samples = th.thickness_max_samples
    verts, tris, tri_face, excluded = _coarse_surface_mesh(th.thickness_size_frac)
    if excluded:
        report.add("thickness.internal_faces_excluded", Severity.INFO,
                   f"excluded {len(excluded)} internal interface face(s) from ray "
                   f"casting: {excluded[:10]}. Thickness is measured through the "
                   "outer skin of the bonded region, not to internal walls.",
                   data={"excluded_faces": excluded})
    if tris.size == 0:
        report.add("thickness.no_surface", Severity.BLOCK,
                   "coarse surface mesh produced no triangles")
        return

    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    # trimesh normals point outward for a consistently-oriented closed mesh; if
    # the CAD solid was valid, gmsh gives us that orientation.
    if not mesh.is_watertight:
        report.add("thickness.not_watertight", Severity.WARN,
                   "coarse outer-skin mesh is not watertight; thickness values are "
                   "unreliable and the geometry audit findings take precedence",
                   data={"excluded_faces": excluded})

    centers = mesh.triangles_center
    normals = mesh.face_normals
    n_tri = len(centers)
    if n_tri > max_samples:
        sel = np.random.default_rng(0).choice(n_tri, max_samples, replace=False)
    else:
        sel = np.arange(n_tri)

    eps = EPS_FRAC * report.model_scale
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    owner: list[int] = []
    for i in sel:
        inward = -normals[i]
        for d in _cone_directions(inward, CONE_RAYS, CONE_HALF_ANGLE):
            origins.append(centers[i] + d * eps)
            directions.append(d)
            owner.append(i)

    origins = np.asarray(origins)
    directions = np.asarray(directions)
    owner = np.asarray(owner)

    # multiple_hits=False returns only the nearest hit per ray.
    locs, ray_idx, _ = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False)

    if len(ray_idx) == 0:
        report.add("thickness.no_hits", Severity.WARN,
                   "no inward ray hits; check surface normal orientation")
        return

    dist = np.linalg.norm(locs - origins[ray_idx], axis=1)

    per_tri: dict[int, list[float]] = {}
    for r, d in zip(ray_idx, dist):
        per_tri.setdefault(int(owner[r]), []).append(float(d))

    values: list[float] = []
    for tri_i, ds in per_tri.items():
        t = float(np.median(ds))
        values.append(t)
        report.thickness_samples.append(ThicknessSample(
            x=float(centers[tri_i][0]), y=float(centers[tri_i][1]),
            z=float(centers[tri_i][2]), thickness=t,
            face_tag=int(tri_face[tri_i]),
        ))

    arr = np.asarray(values)
    report.thickness_median = float(np.median(arr))
    report.thickness_p01 = float(np.percentile(arr, 1.0))

    report.add("thickness.measured", Severity.INFO,
               f"{len(arr)} samples; median {report.thickness_median:.4g}, "
               f"p01 {report.thickness_p01:.4g}, min {arr.min():.4g}",
               data={"median": report.thickness_median,
                     "p01": report.thickness_p01,
                     "min": float(arr.min())})


def check_against_element_size(report: Stage0Report, target_h: float,
                               th: Thresholds = DEFAULTS,
                               min_elements_through: int | None = None) -> None:
    """The actual guard. Call this once Stage 1 proposes a target element size."""
    if min_elements_through is None:
        min_elements_through = th.min_elements_through_thickness
    if report.thickness_p01 is None:
        return
    needed = target_h * min_elements_through
    if report.thickness_p01 < needed:
        thin = [s for s in report.thickness_samples if s.thickness < needed]
        faces = sorted({s.face_tag for s in thin})
        report.add("thickness.under_resolved", Severity.BLOCK,
                   f"p01 thickness {report.thickness_p01:.4g} < "
                   f"{min_elements_through} x h ({needed:.4g}). "
                   f"{len(thin)} sample(s) on face(s) {faces[:10]} would get fewer "
                   f"than {min_elements_through} elements through-thickness. "
                   "Refine locally or reconsider solid tets for these regions.",
                   threshold_name="min_elements_through_thickness",
                   threshold=float(needed),
                   measured=float(report.thickness_p01),
                   data={"needed": needed, "faces": faces})
