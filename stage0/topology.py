"""
Edge incidence classification. Pure -- no gmsh, so it is unit-testable.

The rule that matters:

    Count curve->face incidences WITH MULTIPLICITY, not distinct faces.

A seam curve of a periodic surface (full cylinder, sphere, torus) appears twice
in that one face's boundary loop, once per orientation. The surface wraps around
and closes on itself along it, so the curve has material on both sides -- it is
manifold and watertight. But it touches only ONE distinct face.

Counting distinct faces therefore reports every seam curve as a free edge, which
is a false BLOCK on any part containing a through hole. Counting incidences
distinguishes the two cases cleanly:

    incidences  distinct  meaning
    ----------  --------  -------------------------------------------------
        1          1      genuine free edge -- open shell, cannot be watertight
        2          2      ordinary manifold edge between two faces
        2          1      SEAM of a periodic face -- manifold, fine
       >2         any     non-manifold: expected at imprinted assembly
                          interfaces, suspicious otherwise
"""

from __future__ import annotations

from enum import Enum


class EdgeClass(str, Enum):
    FREE = "free"               # BLOCK: shell is open here
    MANIFOLD = "manifold"       # ordinary shared edge
    SEAM = "seam"               # periodic face closing on itself
    NONMANIFOLD = "nonmanifold" # 3+ incidences
    INTERFACE = "interface"     # 3+ incidences AT a conformal assembly interface
    ORPHAN = "orphan"           # bounds nothing at all


def edge_incidence(face_boundaries: dict[int, list[int]]):
    """Build per-curve incidence counts and distinct-face lists.

    Args:
        face_boundaries: face tag -> list of bounding curve tags, WITH duplicates
            preserved (a seam curve must appear twice for its face).

    Returns:
        (incidences, faces) where incidences maps curve tag -> int count and
        faces maps curve tag -> sorted list of distinct face tags.
    """
    incidences: dict[int, int] = {}
    faces: dict[int, list[int]] = {}
    for ftag, curves in face_boundaries.items():
        for ctag in curves:
            t = abs(ctag)
            incidences[t] = incidences.get(t, 0) + 1
            faces.setdefault(t, []).append(ftag)
    return incidences, {k: sorted(set(v)) for k, v in faces.items()}


def classify_edge(n_incidences: int, n_distinct_faces: int) -> EdgeClass:
    """Classify one curve from its incidence count and distinct face count."""
    if n_incidences == 0:
        return EdgeClass.ORPHAN
    if n_incidences == 1:
        return EdgeClass.FREE
    if n_incidences == 2:
        return EdgeClass.SEAM if n_distinct_faces == 1 else EdgeClass.MANIFOLD
    return EdgeClass.NONMANIFOLD


# --------------------------------------------------------------------------
# Volume ownership. Separates the outer skin of a solid region from internal
# interfaces created by imprinting.
# --------------------------------------------------------------------------

def external_faces(owners: dict[int, set[int]]) -> list[int]:
    """Faces bounding exactly ONE volume -- the outer skin.

    After fragment() imprints an assembly, the shared interface faces are real
    mesh faces sitting INSIDE the material. Anything that treats the full face
    set as a boundary (ray casting, watertightness) must exclude them or it will
    measure to an internal wall and call it a surface.
    """
    return sorted(t for t, v in owners.items() if len(v) == 1)


def interface_faces(owners: dict[int, set[int]]) -> list[int]:
    """Faces bounding two or more volumes -- conformal assembly interfaces."""
    return sorted(t for t, v in owners.items() if len(v) >= 2)


def is_interface_edge(incident_faces, interface_face_tags) -> bool:
    """True if this curve touches a known conformal interface face.

    A curve bounding an imprinted interface has 3+ face incidences by
    construction: the interface itself plus one face from each adjoining volume.
    That is expected topology, not a defect -- but raw incidence counting cannot
    tell it apart from a genuine non-manifold edge. Cross-referencing the
    interface list can.
    """
    iface = set(interface_face_tags)
    return bool(iface) and any(f in iface for f in incident_faces)


def classify_edge_in_assembly(n_incidences: int, n_distinct_faces: int,
                              incident_faces, interface_face_tags) -> EdgeClass:
    """classify_edge(), plus INTERFACE for expected assembly non-manifoldness."""
    base = classify_edge(n_incidences, n_distinct_faces)
    if base is EdgeClass.NONMANIFOLD and is_interface_edge(incident_faces,
                                                           interface_face_tags):
        return EdgeClass.INTERFACE
    return base
