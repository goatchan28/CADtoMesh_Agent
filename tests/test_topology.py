"""
Edge classification tests. No gmsh needed.

Regression: block_hole.step reported curve 27 as a BLOCK free edge. Curve 27 is
the seam of the cylindrical hole face 11 -- manifold, not a defect. The cause was
counting DISTINCT faces instead of incidences.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage0.topology import (EdgeClass, classify_edge, classify_edge_in_assembly,
                             edge_incidence, external_faces, interface_faces,
                             is_interface_edge)


def test_ordinary_manifold_edge():
    """Curve 5 shared between faces 1 and 2."""
    inc, faces = edge_incidence({1: [5, 6], 2: [5, 7]})
    assert inc[5] == 2 and faces[5] == [1, 2]
    assert classify_edge(inc[5], len(faces[5])) is EdgeClass.MANIFOLD


def test_seam_edge_is_not_free():
    """THE REGRESSION. Face 11 is a full cylinder; curve 27 is its seam and
    appears twice in its own boundary loop. Two incidences, one distinct face."""
    inc, faces = edge_incidence({11: [25, 26, 27, 27]})
    assert inc[27] == 2, inc
    assert faces[27] == [11], faces
    assert classify_edge(inc[27], len(faces[27])) is EdgeClass.SEAM
    # The old distinct-face rule would have said FREE here.
    assert classify_edge(inc[27], len(faces[27])) is not EdgeClass.FREE


def test_genuine_free_edge_still_blocks():
    """An open shell must still be caught: one incidence, one face."""
    inc, faces = edge_incidence({1: [5, 6], 2: [6, 7]})
    assert inc[5] == 1 and faces[5] == [1]
    assert classify_edge(inc[5], len(faces[5])) is EdgeClass.FREE


def test_nonmanifold_edge():
    """Three faces meeting at one curve -- imprinted assembly interface."""
    inc, faces = edge_incidence({1: [9], 2: [9], 3: [9]})
    assert inc[9] == 3
    assert classify_edge(inc[9], len(faces[9])) is EdgeClass.NONMANIFOLD


def test_block_hole_topology_end_to_end():
    """Full block_hole.step topology: 11 faces, 27 curves.

    6 planar box faces + 4 partial-cylinder fillets (90 degree sweep, no seam)
    + 1 full cylindrical hole (has a seam). Exactly one SEAM, zero FREE.
    """
    fb: dict[int, list[int]] = {}
    # Top face (1) and bottom face (2): outer ring of 4 lines + 4 arcs, PLUS the
    # hole circle punched through it -- the hole is a second boundary loop.
    fb[1] = list(range(1, 9)) + [25]
    fb[2] = list(range(9, 17)) + [26]
    # 8 vertical tangency lines, curves 17-24. Four side faces and four fillet
    # faces alternate around the part, each consuming two of them.
    verticals = list(range(17, 25))
    for i in range(8):
        fb[3 + i] = [verticals[i], verticals[(i + 1) % 8], 1 + i, 9 + i]
    # Hole face 11: two circles (25, 26) and the seam (27) twice.
    fb[11] = [25, 26, 27, 27]

    inc, faces = edge_incidence(fb)
    classes = {c: classify_edge(inc[c], len(faces[c])) for c in inc}

    n_seam = sum(1 for v in classes.values() if v is EdgeClass.SEAM)
    n_free = sum(1 for v in classes.values() if v is EdgeClass.FREE)
    assert n_seam == 1, f"expected 1 seam, got {n_seam}"
    assert n_free == 0, (
        f"expected 0 free edges, got {n_free}: "
        f"{[c for c, v in classes.items() if v is EdgeClass.FREE]}")
    assert classes[27] is EdgeClass.SEAM
    assert len(fb) == 11, "block_hole has 11 faces"
    assert len(inc) == 27, f"block_hole has 27 curves, counted {len(inc)}"


def test_orphan_curve():
    inc, faces = edge_incidence({1: [2]})
    assert classify_edge(inc.get(99, 0), 0) is EdgeClass.ORPHAN


# ---------------------------------------------------------------------------
# Assembly awareness.
# Regression: two_blocks.step produced four nonmanifold WARNs on the interface
# face's edges. Each bounds 3 faces -- the interface plus one side face from each
# block -- which is expected topology after imprinting, not a defect.
# ---------------------------------------------------------------------------

def _two_blocks_owners():
    """face tag -> volumes, for two imprinted 30x30x20 blocks (11 faces).

    Face 5 is the shared interface at x=30; the other ten are outer skin.
    """
    owners = {t: {1} for t in (1, 2, 3, 4, 6)}          # block 1 outer
    owners.update({t: {2} for t in (7, 8, 9, 10, 11)})  # block 2 outer
    owners[5] = {1, 2}                                   # shared interface
    return owners


def test_interface_and_external_face_split():
    owners = _two_blocks_owners()
    assert interface_faces(owners) == [5]
    assert external_faces(owners) == [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    assert len(external_faces(owners)) == 10, "11 faces, 1 internal"


def test_interface_edge_downgraded_from_nonmanifold():
    """THE REGRESSION. Curve 5 bounds interface face 5 plus one face per block."""
    iface = interface_faces(_two_blocks_owners())
    incident = [5, 2, 8]
    assert classify_edge(3, 3) is EdgeClass.NONMANIFOLD, "raw count says nonmanifold"
    assert is_interface_edge(incident, iface)
    assert classify_edge_in_assembly(3, 3, incident, iface) is EdgeClass.INTERFACE


def test_genuine_nonmanifold_still_warns():
    """3 incidences with no interface face involved must stay NONMANIFOLD."""
    iface = interface_faces(_two_blocks_owners())
    incident = [2, 3, 4]          # no face 5
    assert not is_interface_edge(incident, iface)
    assert classify_edge_in_assembly(3, 3, incident, iface) is EdgeClass.NONMANIFOLD


def test_single_part_has_no_interfaces():
    """A one-volume part: every face is skin, so nothing gets downgraded."""
    owners = {t: {1} for t in range(1, 12)}
    assert interface_faces(owners) == []
    assert len(external_faces(owners)) == 11
    assert not is_interface_edge([1, 2, 3], [])
    assert classify_edge_in_assembly(3, 3, [1, 2, 3], []) is EdgeClass.NONMANIFOLD


def test_seam_and_free_unaffected_by_interface_list():
    """Downgrading must apply ONLY to nonmanifold; a free edge stays a BLOCK."""
    iface = [5]
    assert classify_edge_in_assembly(1, 1, [5], iface) is EdgeClass.FREE
    assert classify_edge_in_assembly(2, 1, [5], iface) is EdgeClass.SEAM
    assert classify_edge_in_assembly(2, 2, [5, 6], iface) is EdgeClass.MANIFOLD


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
