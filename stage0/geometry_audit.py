"""
Stage 0a -- topology / tolerance audit. No meshing.

Separates CAD defects from meshing defects before any mesher runs, so downstream
failures are attributable.

HEALING IS OPT-IN. Read the note on import_cad() before changing that.
"""

from __future__ import annotations

import math

import gmsh

from .config import Thresholds, DEFAULTS
from .report import Stage0Report, Severity
from .topology import interface_faces


def import_cad(path: str, *, tolerance: float | None = None,
               heal: bool = False) -> None:
    """Import STEP/IGES/BREP into the OCC kernel and synchronize.

    HEALING DEFAULTS TO OFF, and that default is load-bearing.

    An earlier version forced OCCFixDegenerated, OCCFixSmallEdges,
    OCCFixSmallFaces, OCCSewFaces and healShapes() on every import. That
    destroyed valid geometry:

      * A sphere's poles are DEGENERATE EDGES by construction. FixDegenerated
        strips them, the shell can no longer validate as closed, and a perfectly
        good solid imports as 0 volumes.
      * Two solids sharing a coincident face get sewn together by SewFaces into
        shells, dropping both solid wrappers.

    In both cases the audit then reported topology.no_volume -- a BLOCK caused
    entirely by the repair, on geometry that was never broken.

    The lesson generalizes: healing is a REMEDIAL ACTION, not an import setting.
    Import raw, audit, and heal only if the raw audit shows a defect worth
    repairing -- then re-audit and compare. That also keeps Stage 0 honest, since
    it reports what is in the file rather than what OCC wishes were in it.

    Caller owns gmsh.initialize()/finalize().
    """
    from backends.base import check_cad_format
    check_cad_format(path)      # named error before OCC sees an unreadable file

    if tolerance is not None:
        gmsh.option.setNumber("Geometry.Tolerance", tolerance)

    flag = 1 if heal else 0
    gmsh.option.setNumber("Geometry.OCCFixDegenerated", flag)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", flag)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", flag)
    gmsh.option.setNumber("Geometry.OCCSewFaces", flag)

    gmsh.model.occ.importShapes(path)
    if heal:
        gmsh.model.occ.healShapes()
    gmsh.model.occ.synchronize()


def circularity(area: float, perimeter: float) -> float:
    """Isoperimetric quotient 4*pi*A/P^2. Pure -- unit-testable.

    Scale-free shape measure: 1.0 for a circle, 0.785 for a square, approaching 0
    for a sliver. For a rectangle of aspect ratio r it equals pi*r/(1+r)^2.

    This is the right sliver test. Area alone is not: the 50 x 0.005 rail face on
    sliver_block.step has area 0.25 (4e-5 of scale^2, nowhere near "tiny") but
    circularity 3e-4. It is thin, not small, and only circularity sees that.
    """
    if perimeter <= 0.0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def aspect_from_circularity(c: float) -> float:
    """Invert circularity = pi*r/(1+r)^2 for a rectangle. Pure, for messages."""
    if c <= 0:
        return math.inf
    if c >= math.pi / 4:
        return 1.0
    b = 2.0 - math.pi / c          # r^2 + b r + 1 = 0
    disc = b * b - 4.0
    if disc <= 0:
        return 1.0
    return (-b + math.sqrt(disc)) / 2.0


def _face_perimeter(face_tag: int) -> float:
    """Total length of a face's DISTINCT bounding curves.

    Deduplicated on purpose: a seam curve appears twice in a periodic face's
    boundary, and double-counting it would deflate circularity and manufacture a
    false sliver on every cylindrical face.
    """
    bnd = gmsh.model.getBoundary([(2, face_tag)], combined=False, oriented=False)
    return sum(gmsh.model.occ.getMass(1, t) for t in {abs(t) for _, t in bnd})


def enforce_conformal_assembly(report: Stage0Report) -> None:
    """Imprint multi-body assemblies so contacting parts share topology.

    The single most dangerous omission in a CAD->FEM pipeline. Without fragment(),
    two touching solids mesh independently, share no nodes, and the assembled
    model transmits no load across the interface -- while looking normal.
    """
    volumes = gmsh.model.getEntities(3)
    if len(volumes) < 2:
        return

    faces_before = len(gmsh.model.getEntities(2))
    gmsh.model.occ.fragment(volumes, [])
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    faces_after = len(gmsh.model.getEntities(2))

    shared = interface_faces(face_volume_owners())
    # Recorded on the report so the 1D edge check and the thickness pass can both
    # reason about interfaces instead of rediscovering them.
    report.shared_interface_faces = shared
    report.add(
        "assembly.imprinted",
        Severity.INFO if shared else Severity.WARN,
        f"{len(volumes)} volumes imprinted; faces {faces_before} -> {faces_after}; "
        f"{len(shared)} shared interface face(s)"
        + ("" if shared else " -- volumes may not actually touch, verify intent"),
        data={"shared_faces": shared},
    )


def face_volume_owners() -> dict[int, set[int]]:
    """face tag -> set of volume tags bounding it.

    len == 1 is the outer skin; len >= 2 is a conformal interface created by
    imprinting. Both callers need this, so it is computed once here.
    """
    owners: dict[int, set[int]] = {}
    for _, vtag in gmsh.model.getEntities(3):
        for _, ftag in gmsh.model.getBoundary([(3, vtag)], combined=False,
                                              oriented=False):
            owners.setdefault(abs(ftag), set()).add(vtag)
    return owners


def audit(report: Stage0Report, th: Thresholds = DEFAULTS) -> None:
    """Populate topology counts, model scale, and defect findings."""
    ents = {d: gmsh.model.getEntities(d) for d in (0, 1, 2, 3)}
    report.n_points = len(ents[0])
    report.n_curves = len(ents[1])
    report.n_faces = len(ents[2])
    report.n_volumes = len(ents[3])

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    report.bbox = (xmin, ymin, zmin, xmax, ymax, zmax)
    diag = math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))
    report.model_scale = diag

    if report.n_volumes == 0:
        if report.healed:
            hint = ("This run ALREADY healed. Healing can itself destroy valid "
                    "solids -- it strips a sphere's degenerate pole edges and sews "
                    "coincident faces of touching solids. Retry WITHOUT --heal "
                    "before concluding the file is broken.")
        else:
            hint = ("Raw import. If the file really does hold only surfaces, "
                    "retry with --heal to let OCC attempt to close the shell.")
        report.add("topology.no_volume", Severity.BLOCK,
                   f"No 3D volume after import ({report.n_faces} faces present). "
                   + hint)
        return

    report.total_volume = sum(gmsh.model.occ.getMass(3, t) for _, t in ents[3])
    if report.total_volume <= 0:
        report.add("topology.nonpositive_volume", Severity.BLOCK,
                   f"Total volume is {report.total_volume:.4g}; orientation or "
                   "closure is broken.",
                   measured=report.total_volume)

    # --- open shells: every face of a valid solid must bound a volume ---
    face_owners: dict[int, int] = {}
    for _, vtag in ents[3]:
        for _, ftag in gmsh.model.getBoundary([(3, vtag)], combined=False,
                                              oriented=False):
            face_owners[abs(ftag)] = face_owners.get(abs(ftag), 0) + 1
    orphans = [t for _, t in ents[2] if abs(t) not in face_owners]
    if orphans:
        report.add("topology.orphan_faces", Severity.BLOCK,
                   f"{len(orphans)} face(s) bound no volume: {orphans[:10]}",
                   data={"tags": orphans})

    # --- sub-tolerance curves ---
    for _, tag in ents[1]:
        L = gmsh.model.occ.getMass(1, tag)
        frac = L / diag if diag else 0.0
        if frac <= th.degenerate_curve_frac:
            # Expected at cone apexes and sphere poles: OCC represents these as
            # zero-length edges. Not a defect -- and note that healing DELETES
            # them, which is exactly what broke sphere.step.
            report.add("curve.degenerate", Severity.INFO,
                       f"zero-length edge (length {L:.3g}) -- pole/apex, expected",
                       dim=1, tag=tag,
                       threshold_name="degenerate_curve_frac",
                       threshold=th.degenerate_curve_frac, measured=frac)
        elif frac < th.tiny_curve_frac:
            report.add("curve.tiny", Severity.WARN,
                       f"curve length {L:.3g} is {frac:.2e} of model scale; will "
                       "force local over-refinement or fail to discretize",
                       dim=1, tag=tag,
                       threshold_name="tiny_curve_frac",
                       threshold=th.tiny_curve_frac, measured=frac)

    # --- faces: THIN and SMALL are different defects, checked separately ---
    for _, tag in ents[2]:
        A = gmsh.model.occ.getMass(2, tag)
        P = _face_perimeter(tag)
        c = circularity(A, P)
        area_frac = A / (diag * diag) if diag else 0.0

        if c < th.sliver_circularity:
            aspect = aspect_from_circularity(c)
            report.add("face.sliver", Severity.WARN,
                       f"circularity {c:.3e} (about {aspect:.0f}:1 aspect ratio), "
                       f"area {A:.4g}, perimeter {P:.4g}; thin face forces "
                       "over-refinement along its short direction",
                       dim=2, tag=tag,
                       threshold_name="sliver_circularity",
                       threshold=th.sliver_circularity, measured=c,
                       data={"area": A, "perimeter": P, "aspect_ratio": aspect})

        if area_frac < th.tiny_face_area_frac:
            report.add("face.tiny", Severity.WARN,
                       f"area {A:.3g} is {area_frac:.2e} of scale^2; candidate "
                       "for defeaturing",
                       dim=2, tag=tag,
                       threshold_name="tiny_face_area_frac",
                       threshold=th.tiny_face_area_frac, measured=area_frac)
