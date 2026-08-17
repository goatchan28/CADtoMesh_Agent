"""
Entity identity tests. PURE -- no gmsh.

What has to be true for the pipeline to work at all:

  * identical geometry, ANY tag order  -> identical ids (else two runs disagree)
  * small healing perturbation         -> ids preserved
  * imprint merging two faces          -> MERGE recorded, both parents traceable
  * imprint splitting one face         -> SPLIT recorded, children traceable
  * a healed-away sliver               -> DELETED, not silently reused
  * a genuinely different face         -> different id (no false identity)

The last one matters most. A false SAME is worse than a false CREATED: it silently
attaches evidence from one face to a different face, and every localized action
downstream then targets the wrong geometry.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pipeline.entities import (DEFAULT_DIGITS, EntityDescriptor, EntityRegistry,
                               ModelFrame, Relation, canonical_order, fingerprint,
                               match_snapshots)

FRAME = ModelFrame.from_bbox(0, 0, 0, 60, 40, 25)     # block_hole's bbox


def face(tag, mass, centroid, bbox, nb=4, curv=()):
    return EntityDescriptor(dim=2, tag=tag, mass=mass, centroid=centroid,
                            bbox=bbox, n_boundary=nb, curvature_sig=curv)


def box_faces(x0=0.0, x1=30.0, tags=(1, 2, 3, 4, 5, 6)):
    """Six faces of an axis-aligned box spanning [x0,x1] x [0,30] x [0,20]."""
    w = x1 - x0
    cx = 0.5 * (x0 + x1)
    return [
        face(tags[0], w * 30, (cx, 15, 0), (x0, 0, 0, x1, 30, 0)),
        face(tags[1], w * 30, (cx, 15, 20), (x0, 0, 20, x1, 30, 20)),
        face(tags[2], w * 20, (cx, 0, 10), (x0, 0, 0, x1, 0, 20)),
        face(tags[3], w * 20, (cx, 30, 10), (x0, 30, 0, x1, 30, 20)),
        face(tags[4], 30 * 20, (x0, 15, 10), (x0, 0, 0, x0, 30, 20)),
        face(tags[5], 30 * 20, (x1, 15, 10), (x1, 0, 0, x1, 30, 20)),
    ]


# --------------------------------------------------------------------------
# Fingerprint invariance
# --------------------------------------------------------------------------

def test_identical_geometry_gives_identical_fingerprint():
    a = face(1, 2400.0, (30, 20, 0), (0, 0, 0, 60, 40, 0))
    b = face(999, 2400.0, (30, 20, 0), (0, 0, 0, 60, 40, 0))   # different TAG
    assert fingerprint(a, FRAME) == fingerprint(b, FRAME), \
        "the backend tag must not participate in identity"


def test_fingerprint_separates_genuinely_different_faces():
    """A false SAME is the worst failure mode: evidence attaches to the wrong face."""
    base = face(1, 2400.0, (30, 20, 0), (0, 0, 0, 60, 40, 0))
    variants = [
        face(1, 2400.0, (30, 20, 25), (0, 0, 25, 60, 40, 25)),   # moved
        face(1, 1200.0, (30, 20, 0), (0, 0, 0, 60, 20, 0)),      # half area
        face(1, 2400.0, (30, 20, 0), (0, 0, 0, 60, 40, 0), nb=5),  # extra loop
        face(1, 2400.0, (30, 20, 0), (0, 0, 0, 60, 40, 0), curv=(0.1,)),  # curved
    ]
    fps = {fingerprint(v, FRAME) for v in variants}
    assert fingerprint(base, FRAME) not in fps
    assert len(fps) == len(variants), "each difference must produce a distinct id"


def test_fingerprint_survives_sub_tolerance_perturbation():
    """Healing nudges geometry. Quantization must absorb nudges below its floor."""
    eps = 1e-6 * FRAME.scale
    a = face(1, 2400.0, (30, 20, 0), (0, 0, 0, 60, 40, 0))
    b = face(1, 2400.0 + eps, (30 + eps, 20, 0), (0, 0, 0, 60, 40, 0))
    assert fingerprint(a, FRAME) == fingerprint(b, FRAME)


def test_registration_is_independent_of_tag_order():
    """Two runs on identical geometry must produce identical ids."""
    faces = box_faces()
    r1, r2 = EntityRegistry(), EntityRegistry()
    r1.register(list(faces), FRAME, "a")
    r2.register(list(reversed(faces)), FRAME, "a")
    assert sorted(r1.records) == sorted(r2.records)


def test_canonical_order_is_deterministic():
    faces = box_faces()
    assert ([d.tag for d in canonical_order(faces)]
            == [d.tag for d in canonical_order(list(reversed(faces)))])


# --------------------------------------------------------------------------
# Frame stability
# --------------------------------------------------------------------------

def test_first_frame_is_reused_across_snapshots():
    """A healing pass that trims a sliver changes the model bbox.

    Recomputing the frame per snapshot would shift the normalization and change
    EVERY fingerprint, even for geometry that never moved.
    """
    reg = EntityRegistry()
    reg.register(box_faces(), FRAME, "raw")
    shrunk = ModelFrame.from_bbox(0, 0, 0, 59.99, 40, 25)
    reg.register(box_faces(), shrunk, "healed")
    assert reg.frame == FRAME, "frame must not be replaced"
    assert reg.stability()["by_relation"].get("same") == 6


# --------------------------------------------------------------------------
# Provenance: the non-bijective cases
# --------------------------------------------------------------------------

def test_reimport_preserves_all_ids():
    reg = EntityRegistry()
    reg.register(box_faces(), FRAME, "import1")
    before = {r.pid for r in reg.live()}
    reg.register(box_faces(tags=(100, 101, 102, 103, 104, 105)), FRAME, "import2")
    assert {r.pid for r in reg.live()} == before
    assert reg.stability()["preserved_frac"] == 1.0


def test_imprint_merge_of_two_coincident_faces():
    """two_blocks.step: fragment() turns two coincident faces into one interface.

    Both parents must be traceable to the merged child, or a finding recorded
    against either one is lost.
    """
    left = box_faces(0, 30, tags=(1, 2, 3, 4, 5, 6))
    right = box_faces(30, 60, tags=(7, 8, 9, 10, 11, 12))
    reg = EntityRegistry()
    reg.register(left + right, FRAME, "raw")
    pid_left_far = reg.pid_for_tag(2, 6)      # x=30 face of the left box
    pid_right_near = reg.pid_for_tag(2, 11)   # x=30 face of the right box
    assert pid_left_far and pid_right_near and pid_left_far != pid_right_near

    # After imprint: 11 faces, the two x=30 faces replaced by one interface.
    merged = face(50, 30 * 20, (30, 15, 10), (30, 0, 0, 30, 30, 20))
    after = ([f for f in left if f.tag != 6] + [f for f in right if f.tag != 11]
             + [merged])
    reg.register(after, FRAME, "imprinted")

    child = reg.pid_for_tag(2, 50)
    assert child is not None
    rec = reg.records[child]
    assert rec.relation is Relation.MERGE, rec.relation
    assert set(rec.parents) == {pid_left_far, pid_right_near}
    for p in (pid_left_far, pid_right_near):
        assert reg.descendants(p) == [child], "both parents must reach the child"
        assert not reg.records[p].alive
    assert len(reg.live(dim=2)) == 11


def test_imprint_split_of_one_face():
    """A partially overlapping solid splits a face. Children must trace to it."""
    faces = box_faces()
    reg = EntityRegistry()
    reg.register(faces, FRAME, "raw")
    parent = reg.pid_for_tag(2, 2)            # top face, z=20, area 900

    a = face(40, 300.0, (5, 15, 20), (0, 0, 20, 10, 30, 20))
    b = face(41, 600.0, (20, 15, 20), (10, 0, 20, 30, 30, 20))
    after = [f for f in faces if f.tag != 2] + [a, b]
    reg.register(after, FRAME, "imprinted")

    kids = reg.descendants(parent)
    assert len(kids) == 2, kids
    for k in kids:
        assert reg.records[k].relation is Relation.SPLIT
        assert parent in reg.records[k].parents
        assert reg.ancestors(k) == [parent]
    assert not reg.records[parent].alive


def test_healed_away_sliver_is_deleted_not_reused():
    """A removed entity must not have its id silently reassigned."""
    faces = box_faces()
    sliver = face(99, 1e-6, (15, 0.001, 20), (0, 0, 20, 30, 0.002, 20))
    reg = EntityRegistry()
    reg.register(faces + [sliver], FRAME, "raw")
    pid = reg.pid_for_tag(2, 99)
    reg.register(faces, FRAME, "healed")
    assert not reg.records[pid].alive
    assert reg.tag_for_pid(pid) is None
    assert reg.records[pid].children == []
    assert len(reg.live(dim=2)) == 6


def test_created_entity_has_no_false_parent():
    faces = box_faces()
    reg = EntityRegistry()
    reg.register(faces, FRAME, "raw")
    # A face somewhere else entirely: must not be attributed to any existing face.
    newcomer = face(77, 100.0, (200, 200, 200), (195, 195, 195, 205, 205, 205))
    reg.register(faces + [newcomer], FRAME, "added")
    pid = reg.pid_for_tag(2, 77)
    assert reg.records[pid].relation is Relation.CREATED
    assert reg.records[pid].parents == []


def test_match_snapshots_classifies_all_relations():
    faces = box_faces()
    reg = EntityRegistry()
    reg.register(faces, FRAME, "raw")
    kept = [f for f in faces if f.tag not in (1, 2)]
    split_a = face(40, 300.0, (5, 15, 20), (0, 0, 20, 10, 30, 20))
    split_b = face(41, 600.0, (20, 15, 20), (10, 0, 20, 30, 30, 20))
    created = face(77, 100.0, (200, 200, 200), (195, 195, 195, 205, 205, 205))
    ms = match_snapshots(list(reg.records.values()),
                         kept + [split_a, split_b, created], FRAME)
    kinds = {m.relation for m in ms}
    assert Relation.SAME in kinds and Relation.SPLIT in kinds
    assert Relation.CREATED in kinds and Relation.DELETED in kinds
    split = next(m for m in ms if m.relation is Relation.SPLIT)
    assert abs(split.mass_ratio - 1.0) < 1e-9, "split must conserve area"


def test_mass_ratio_flags_geometry_loss():
    """A split that loses area is a defeaturing event, not a clean imprint."""
    faces = box_faces()
    reg = EntityRegistry()
    reg.register(faces, FRAME, "raw")
    lossy = face(40, 450.0, (10, 15, 20), (0, 0, 20, 20, 30, 20))   # half of 900
    ms = match_snapshots(list(reg.records.values()),
                         [f for f in faces if f.tag != 2] + [lossy], FRAME)
    m = next(mm for mm in ms if 40 in mm.new_tags)
    assert abs(m.mass_ratio - 0.5) < 1e-9, m.mass_ratio


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def test_registry_serializes_with_provenance():
    """A future agent reads this; it must carry the graph, not just the ids."""
    import json
    left = box_faces(0, 30, tags=(1, 2, 3, 4, 5, 6))
    right = box_faces(30, 60, tags=(7, 8, 9, 10, 11, 12))
    reg = EntityRegistry()
    reg.register(left + right, FRAME, "raw")
    merged = face(50, 600.0, (30, 15, 10), (30, 0, 0, 30, 30, 20))
    reg.register([f for f in left if f.tag != 6]
                 + [f for f in right if f.tag != 11] + [merged], FRAME, "imprinted")
    d = json.loads(reg.to_json())
    assert d["snapshots"] == ["raw", "imprinted"]
    assert d["frame"]["scale"] == FRAME.scale
    child = next(e for e in d["entities"] if e["tag"] == 50)
    assert child["relation"] == "merge" and len(child["parents"]) == 2
    dead = [e for e in d["entities"] if not e["alive"]]
    assert len(dead) == 2 and all(e["children"] == [child["pid"]] for e in dead)


def test_stability_summary_reports_preservation():
    reg = EntityRegistry()
    reg.register(box_faces(), FRAME, "raw")
    reg.register(box_faces(tags=(20, 21, 22, 23, 24, 25)), FRAME, "reimport")
    s = reg.stability(dim=2)
    assert s["n_live"] == 6
    assert s["preserved_frac"] == 1.0
    assert s["snapshots"] == ["raw", "reimport"]


# --------------------------------------------------------------------------
# Coincident faces with OPPOSITE normals -- the real-CAD case
# --------------------------------------------------------------------------

def _oriented(tag, mass, centroid, bbox, normal):
    return EntityDescriptor(dim=2, tag=tag, mass=mass, centroid=centroid,
                            bbox=bbox, n_boundary=4, normal_sig=normal)


IFACE_BBOX = (30, 0, 0, 30, 30, 20)


def _assembly_with_normals():
    """Left/right box faces plus the two coincident x=30 faces, normals opposed."""
    others = [d for d in box_faces(0, 30, tags=(1, 2, 3, 4, 5, 99)) if d.tag != 99]
    left = _oriented(6, 600.0, (30, 15, 10), IFACE_BBOX, (1.0, 0.0, 0.0))
    right = _oriented(11, 600.0, (30, 15, 10), IFACE_BBOX, (-1.0, 0.0, 0.0))
    return others, left, right


def test_normal_signature_separates_coincident_faces():
    """The discriminator works -- which is precisely what creates the trap below."""
    _, left, right = _assembly_with_normals()
    assert fingerprint(left, FRAME) != fingerprint(right, FRAME)


def test_merge_detected_when_normals_differ():
    """REGRESSION, found on real two_blocks.step geometry.

    Coincident faces of touching solids have OPPOSITE outward normals, so
    normal_sig makes their fingerprints differ. After fragment() the single
    interface face then matches ONE parent exactly -- an apparently unambiguous 1:1
    hit. Fingerprint matching accepted it as SAME and orphaned the other parent as
    DELETED: the real run reported same 11, dead 1, merge 0 where a merge was
    correct.

    The discriminator made the matcher CONFIDENT exactly where it should have been
    uncertain, so coincidence must be resolved BEFORE fingerprints are consulted.
    """
    others, left, right = _assembly_with_normals()
    reg = EntityRegistry()
    reg.register(others + [left, right], FRAME, "raw")
    p_left, p_right = reg.pid_for_tag(2, 6), reg.pid_for_tag(2, 11)
    assert p_left != p_right

    iface = _oriented(50, 600.0, (30, 15, 10), IFACE_BBOX, (1.0, 0.0, 0.0))
    reg.register(others + [iface], FRAME, "imprinted")

    child = reg.pid_for_tag(2, 50)
    rec = reg.records[child]
    assert rec.relation is Relation.MERGE, rec.relation
    assert set(rec.parents) == {p_left, p_right}
    for parent in (p_left, p_right):
        assert reg.descendants(parent) == [child]
        assert not reg.records[parent].alive
    assert len(reg.live(dim=2)) == 6


def test_coincident_pair_survives_plain_reimport():
    """The guard on the fix. Contracting must require the cluster to have SHRUNK.

    A re-import of the same assembly has 2 old and 2 new coincident faces. Reading
    that as a merge would destroy identity on every re-import, which A_reimport
    must hold at 1.000.
    """
    others, left, right = _assembly_with_normals()
    reg = EntityRegistry()
    reg.register(others + [left, right], FRAME, "raw")
    before = {r.pid for r in reg.live(dim=2)}

    again = [_oriented(60, 600.0, (30, 15, 10), IFACE_BBOX, (1.0, 0.0, 0.0)),
             _oriented(61, 600.0, (30, 15, 10), IFACE_BBOX, (-1.0, 0.0, 0.0))]
    reg.register(others + again, FRAME, "reimport")

    assert {r.pid for r in reg.live(dim=2)} == before
    assert reg.stability(dim=2)["by_relation"] == {"same": 7}
    assert reg.stability(dim=2)["preserved_frac"] == 1.0


def test_merge_mass_ratio_reports_area_accounting():
    """One 600 face replacing two 600 faces gives 0.5 -- correct and informative."""
    others, left, right = _assembly_with_normals()
    reg = EntityRegistry()
    reg.register(others + [left, right], FRAME, "raw")
    iface = _oriented(50, 600.0, (30, 15, 10), IFACE_BBOX, (1.0, 0.0, 0.0))
    ms = match_snapshots(list(reg.records.values()),
                         canonical_order(others + [iface]), FRAME)
    m = next(x for x in ms if x.relation is Relation.MERGE)
    assert abs(m.mass_ratio - 0.5) < 1e-9, m.mass_ratio


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
