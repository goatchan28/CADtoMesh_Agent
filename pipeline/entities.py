"""
Persistent CAD entity identity. PURE -- no gmsh. Backend-neutral by construction.

The problem
-----------
OCC/gmsh entity tags are NOT stable. They change across healShapes(), fragment(),
re-import, and tolerance changes. We already hit this while building the synthetic
test parts, where fillet edges had to be selected by geometry because "tags are not
stable across boolean operations."

That makes tags useless as report keys. A finding that says "face 47 has poor
chordal error" is meaningless after the next re-import, and a future agent cannot
act on evidence gathered across iterations. Every downstream report, every localized
size-field rule, and every contract check needs an identifier that survives the
operations the pipeline performs on the geometry.

Two mechanisms, because one is not enough
-----------------------------------------
1. FINGERPRINT -- a hash of scale-normalized geometric invariants. Handles the easy
   and common case: the entity is unchanged, so it hashes the same.

2. PROVENANCE MATCHING -- geometric correspondence when the fingerprint changes.
   This is mandatory, not a fallback for robustness, because identity is genuinely
   NOT BIJECTIVE under the operations we run:

       fragment() on two solids sharing a coincident face MERGES two faces into one
       fragment() on a partially-overlapping face SPLITS one face into several
       healShapes() perturbs mass and centroid by up to the healing tolerance

   A pure hash cannot express "this face became those three faces". The provenance
   graph can, and downstream code needs exactly that to translate a finding about a
   parent onto its children.

Normalization frame
-------------------
All invariants are normalized by a ModelFrame (origin + scale) captured ONCE, from
the first snapshot, and reused for every later snapshot. Recomputing the frame per
snapshot would be self-defeating: a healing pass that trims a sliver changes the
bounding box, which would shift the normalization and change every fingerprint in
the model even though the geometry did not move.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np

# Quantization of normalized invariants, in decimal places. 4 means entities are
# distinguished down to 1e-4 of model scale (7.6 um on a 76 mm part). Deliberately
# COARSER than the OCC healing tolerance so that small healing perturbations do not
# change the hash; anything larger than that is handled by provenance matching.
DEFAULT_DIGITS = 4


@dataclass(frozen=True)
class ModelFrame:
    """Scale/origin normalization. Captured once, reused for every snapshot."""
    origin: tuple[float, float, float]
    scale: float

    @staticmethod
    def from_bbox(xmin, ymin, zmin, xmax, ymax, zmax) -> "ModelFrame":
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        return ModelFrame(origin=(float(xmin), float(ymin), float(zmin)),
                          scale=max(diag, 1e-12))

    def norm_point(self, p) -> np.ndarray:
        return (np.asarray(p, float) - np.asarray(self.origin, float)) / self.scale

    def norm_mass(self, mass: float, dim: int) -> float:
        return float(mass) / (self.scale ** max(dim, 1))


@dataclass(frozen=True)
class EntityDescriptor:
    """Backend-neutral description of one CAD entity.

    Carries only geometry-derived quantities, so an OCC face, an analytic surface,
    and a hand-built test fixture all describe the same way. `tag` is the backend's
    current handle and is explicitly NOT part of the identity.
    """
    dim: int
    tag: int
    mass: float                                  # length / area / volume
    centroid: tuple[float, float, float]
    bbox: tuple[float, float, float, float, float, float]
    n_boundary: int = 0                          # topological arity
    curvature_sig: tuple[float, ...] = ()        # optional shape signature
    perimeter: float = 0.0
    """Total length of the entity's distinct bounding curves.

    DELIBERATELY NOT part of the fingerprint. Adding a field to the hash payload
    changes every PID in every model, which would invalidate recorded removal
    approvals and any evidence keyed to an id. Like `tag`, this is descriptive data
    that identity does not depend on."""
    normal_sig: tuple[float, ...] = ()
    """Optional area-weighted OUTWARD unit normal.

    Included because two solids sharing a coincident face produce two faces with
    identical mass, centroid and bbox -- geometrically indistinguishable. Their
    outward normals are opposite, so this separates them when the backend can
    supply it. When it cannot, the ambiguity is handled by deferral in
    match_snapshots rather than by a wrong guess."""

    def bbox_min(self) -> np.ndarray:
        return np.asarray(self.bbox[:3], float)

    def bbox_max(self) -> np.ndarray:
        return np.asarray(self.bbox[3:], float)

    def bbox_contains(self, other: "EntityDescriptor", tol: float) -> bool:
        return bool(np.all(other.bbox_min() >= self.bbox_min() - tol) and
                    np.all(other.bbox_max() <= self.bbox_max() + tol))


def fingerprint(desc: EntityDescriptor, frame: ModelFrame,
                digits: int = DEFAULT_DIGITS) -> str:
    """Stable hash of scale-normalized geometric invariants.

    Deliberately excludes the backend tag. Includes topological arity (n_boundary)
    because it cheaply separates entities that are geometrically similar but
    structurally different -- a planar face with a hole versus one without.
    """
    c = np.round(frame.norm_point(desc.centroid), digits)
    lo = np.round(frame.norm_point(desc.bbox[:3]), digits)
    hi = np.round(frame.norm_point(desc.bbox[3:]), digits)
    m = round(frame.norm_mass(desc.mass, desc.dim), digits)
    curv = tuple(round(float(x) * frame.scale, digits) for x in desc.curvature_sig)
    nrm = tuple(round(float(x), digits) for x in desc.normal_sig)

    payload = json.dumps({
        "dim": desc.dim, "m": m, "c": c.tolist(),
        "lo": lo.tolist(), "hi": hi.tolist(),
        "nb": int(desc.n_boundary), "k": curv, "n": nrm,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Relation(str, Enum):
    """How a new entity relates to the previous snapshot."""
    SAME = "same"          # identity preserved
    SPLIT = "split"        # one old entity became several new ones
    MERGE = "merge"        # several old entities became one new one
    CREATED = "created"    # no predecessor (e.g. an imprinted interface face)
    DELETED = "deleted"    # no successor (e.g. a healed-away sliver)


@dataclass
class Match:
    relation: Relation
    new_tags: list[int] = field(default_factory=list)
    old_pids: list[str] = field(default_factory=list)
    dim: int = 0
    mass_ratio: float = 1.0
    """new mass / old mass. Near 1.0 means nothing was lost; a large deviation is a
    signal that the operation changed geometry rather than just topology."""


@dataclass
class EntityRecord:
    pid: str
    dim: int
    tag: int | None                    # current backend tag, None if gone
    descriptor: EntityDescriptor | None
    relation: Relation = Relation.CREATED
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    alive: bool = True
    first_seen: str = ""
    last_seen: str = ""


def _new_pid(desc: EntityDescriptor, frame: ModelFrame, taken: set[str],
             digits: int) -> str:
    """Fingerprint, with deterministic ordinal disambiguation on collision.

    Coarse quantization can make two genuinely distinct entities hash alike. The
    suffix keeps ids unique without ever depending on iteration order, because the
    caller sorts descriptors canonically before registering.
    """
    base = fingerprint(desc, frame, digits)
    if base not in taken:
        return base
    for k in range(1, 1000):
        cand = f"{base}~{k}"
        if cand not in taken:
            return cand
    raise RuntimeError("fingerprint collision space exhausted")


def canonical_order(descs: list[EntityDescriptor]) -> list[EntityDescriptor]:
    """Deterministic ordering, independent of backend tag order.

    Registration must not depend on the order gmsh happens to return entities in,
    or ids would differ between runs on identical geometry.
    """
    return sorted(descs, key=lambda d: (d.dim, round(d.centroid[0], 9),
                                        round(d.centroid[1], 9),
                                        round(d.centroid[2], 9),
                                        round(d.mass, 9), d.tag))


def match_snapshots(old: list[EntityRecord], new: list[EntityDescriptor],
                    frame: ModelFrame, *, digits: int = DEFAULT_DIGITS,
                    rel_tol: float = 0.02) -> list[Match]:
    """Correspond a new snapshot to the previous one, per dimension.

    Three passes, cheapest and most certain first:

      1. exact fingerprint  -> SAME
      2. bbox containment   -> SPLIT (children inside one parent, masses sum) or
                               SAME (single child, mass preserved: a healing nudge)
      3. reverse containment-> MERGE (one new entity covering several old ones)

    Leftovers are CREATED / DELETED. rel_tol is the fractional mass tolerance for
    accepting a split or merge as mass-preserving.
    """
    matches: list[Match] = []
    live = [r for r in old if r.alive and r.descriptor is not None]

    for dim in sorted({d.dim for d in new} | {r.dim for r in live}):
        olds = [r for r in live if r.dim == dim]
        news = [d for d in new if d.dim == dim]
        old_by_fp = {}
        for r in olds:
            old_by_fp.setdefault(fingerprint(r.descriptor, frame, digits), []).append(r)

        used_old: set[str] = set()
        used_new: set[int] = set()
        tol0 = rel_tol * frame.scale

        def _coincident(a, b) -> bool:
            """Same region of space AND same size -- not merely overlapping."""
            if not (a.bbox_contains(b, tol0) and b.bbox_contains(a, tol0)):
                return False
            big = max(a.mass, b.mass)
            return big <= 0 or abs(a.mass - b.mass) / big <= rel_tol

        # --- pass 0: contract coincident clusters ---
        #
        # Runs BEFORE fingerprint matching, and this ordering is the whole point.
        #
        # Two solids meeting at a shared face have coincident faces with OPPOSITE
        # outward normals. normal_sig therefore makes their fingerprints differ, so
        # after fragment() the single interface face matches one parent exactly --
        # an apparently unambiguous 1:1 hit. Pass 1 accepts it as SAME and orphans
        # the other parent as DELETED, losing half the provenance. Measured on
        # two_blocks.step: same 11, dead 1, merge 0, where a merge was correct.
        #
        # The discriminator made pass 1 CONFIDENT exactly where it should have been
        # uncertain, so no amount of tuning inside pass 1 fixes it. Coincidence is a
        # structural fact and has to be resolved first.
        #
        # Contract only when the cluster genuinely shrank. A plain re-import of the
        # same assembly has 2 old and 2 new coincident faces; calling that a merge
        # would destroy identity on every re-import, which A_reimport must keep at
        # 1.000.
        clusters: list[list] = []
        assigned: set[str] = set()
        for r in olds:
            if r.pid in assigned:
                continue
            group = [r]
            assigned.add(r.pid)
            for other in olds:
                if other.pid not in assigned and _coincident(r.descriptor,
                                                             other.descriptor):
                    group.append(other)
                    assigned.add(other.pid)
            if len(group) > 1:
                clusters.append(group)

        for group in clusters:
            here = [d for d in news
                    if d.tag not in used_new
                    and any(_coincident(d, r.descriptor) for r in group)]
            if not here or len(here) >= len(group):
                continue                     # unchanged (or grown): leave to pass 1
            total_old = sum(r.descriptor.mass for r in group)
            total_new = sum(d.mass for d in here)
            ratio = total_new / total_old if total_old else 0.0
            for d in here:
                matches.append(Match(Relation.MERGE, [d.tag],
                                     [r.pid for r in group], dim, ratio))
                used_new.add(d.tag)
            used_old.update(r.pid for r in group)

        # --- pass 1: exact fingerprint, but ONLY when unambiguous ---
        #
        # Matching greedily on fingerprint is wrong when several entities share one.
        # Two solids meeting at a coincident face produce two faces with identical
        # mass, centroid and bbox; after fragment() they become ONE interface face.
        # A greedy match claims that face as SAME for whichever parent it saw first
        # and orphans the other, so the merge is never recorded and half the
        # provenance is lost.
        #
        # Rule: accept a fingerprint match only when the number of old and new
        # entities carrying it agree. Anything else is deferred to the geometric
        # passes, which can express one-to-many and many-to-one.
        new_by_fp: dict[str, list[EntityDescriptor]] = {}
        for d in news:
            if d.tag in used_new:
                continue
            new_by_fp.setdefault(fingerprint(d, frame, digits), []).append(d)

        for fp, group in new_by_fp.items():
            bucket = [r for r in old_by_fp.get(fp, []) if r.pid not in used_old]
            if not bucket or len(bucket) != len(group):
                continue                     # ambiguous -> let the geometry decide
            for d, r in zip(sorted(group, key=lambda x: x.tag),
                            sorted(bucket, key=lambda x: x.pid)):
                used_old.add(r.pid)
                used_new.add(d.tag)
                matches.append(Match(Relation.SAME, [d.tag], [r.pid], dim, 1.0))

        rem_new = [d for d in news if d.tag not in used_new]
        rem_old = [r for r in olds if r.pid not in used_old]
        if not rem_new and not rem_old:
            continue

        tol = rel_tol * frame.scale

        def coincident(a, b) -> bool:
            """Mutually containing: the same region of space, not a parent/child."""
            return a.bbox_contains(b, tol) and b.bbox_contains(a, tol)

        # --- pass 2a: COINCIDENCE, checked before parent/child containment ---
        #
        # Order matters. A merged interface face is contained in each of the two
        # coincident parents it replaced, so a containment test alone reads it as a
        # single child of one parent -- SAME -- and orphans the other. Testing
        # mutual containment first separates "occupies the same space as several
        # old entities" (a merge) from "sits inside a larger old entity" (a split).
        for d in sorted(rem_new, key=lambda x: x.tag):
            same_space = [r for r in rem_old
                          if r.pid not in used_old and coincident(d, r.descriptor)]
            if not same_space:
                continue
            total = sum(r.descriptor.mass for r in same_space)
            ratio = d.mass / total if total else 0.0
            rel = Relation.SAME if len(same_space) == 1 else Relation.MERGE
            matches.append(Match(rel, [d.tag], [r.pid for r in same_space], dim,
                                 ratio if rel is Relation.MERGE else 1.0))
            used_new.add(d.tag)
            used_old.update(r.pid for r in same_space)

        rem_new = [d for d in news if d.tag not in used_new]
        rem_old = [r for r in olds if r.pid not in used_old]

        # --- pass 2b: each remaining new entity sits inside one remaining old one ---
        parent_of: dict[int, str] = {}
        for d in rem_new:
            cands = [r for r in rem_old if r.descriptor.bbox_contains(d, tol)]
            if cands:
                # Tightest enclosing parent: smallest mass that still contains it.
                cands.sort(key=lambda r: r.descriptor.mass)
                parent_of[d.tag] = cands[0].pid

        by_parent: dict[str, list[EntityDescriptor]] = {}
        for d in rem_new:
            pid = parent_of.get(d.tag)
            if pid is not None:
                by_parent.setdefault(pid, []).append(d)

        for pid, children in by_parent.items():
            parent = next(r for r in rem_old if r.pid == pid)
            total = sum(c.mass for c in children)
            ratio = total / parent.descriptor.mass if parent.descriptor.mass else 0.0
            rel = Relation.SAME if len(children) == 1 else Relation.SPLIT
            matches.append(Match(rel, [c.tag for c in children], [pid], dim, ratio))
            used_old.add(pid)
            used_new.update(c.tag for c in children)

        rem_new = [d for d in news if d.tag not in used_new]
        rem_old = [r for r in olds if r.pid not in used_old]

        # --- pass 3: one new entity covering several old ones ---
        for d in rem_new:
            covered = [r for r in rem_old
                       if r.pid not in used_old and d.bbox_contains(r.descriptor, tol)]
            if len(covered) >= 2:
                total = sum(r.descriptor.mass for r in covered)
                ratio = d.mass / total if total else 0.0
                matches.append(Match(Relation.MERGE, [d.tag],
                                     [r.pid for r in covered], dim, ratio))
                used_new.add(d.tag)
                used_old.update(r.pid for r in covered)

        for d in news:
            if d.tag not in used_new:
                matches.append(Match(Relation.CREATED, [d.tag], [], dim, 1.0))
        for r in olds:
            if r.pid not in used_old:
                matches.append(Match(Relation.DELETED, [], [r.pid], dim, 0.0))
    return matches


class EntityRegistry:
    """Assigns persistent ids and records how geometry evolved between snapshots.

    Usage is snapshot-based: register the raw import, then register again after each
    geometry-changing operation. The registry diffs against the previous snapshot
    and maintains the provenance graph.
    """

    def __init__(self, digits: int = DEFAULT_DIGITS, rel_tol: float = 0.02):
        self.digits = digits
        self.rel_tol = rel_tol
        self.frame: ModelFrame | None = None
        self.records: dict[str, EntityRecord] = {}
        self.snapshots: list[str] = []
        self._by_tag: dict[tuple[int, int], str] = {}     # (dim, tag) -> pid

    # -- registration --------------------------------------------------------
    def register(self, descriptors: list[EntityDescriptor], frame: ModelFrame,
                 label: str) -> dict[tuple[int, int], str]:
        """Register a snapshot. The FIRST frame is kept for all later snapshots.

        Reusing the initial frame is deliberate: a healing pass that trims a sliver
        changes the model bounding box, and recomputing the frame would shift the
        normalization and change every fingerprint even where geometry did not move.
        """
        descriptors = canonical_order(descriptors)
        if self.frame is None:
            self.frame = frame
        first = not self.snapshots
        self.snapshots.append(label)
        self._by_tag = {}

        if first:
            taken: set[str] = set()
            for d in descriptors:
                pid = _new_pid(d, self.frame, taken, self.digits)
                taken.add(pid)
                self.records[pid] = EntityRecord(
                    pid=pid, dim=d.dim, tag=d.tag, descriptor=d,
                    relation=Relation.CREATED, first_seen=label, last_seen=label)
                self._by_tag[(d.dim, d.tag)] = pid
            return dict(self._by_tag)

        matches = match_snapshots(list(self.records.values()), descriptors,
                                  self.frame, digits=self.digits,
                                  rel_tol=self.rel_tol)
        by_tag = {(d.dim, d.tag): d for d in descriptors}
        taken = set(self.records)

        for m in matches:
            if m.relation is Relation.DELETED:
                for pid in m.old_pids:
                    r = self.records[pid]
                    r.alive = False
                    r.tag = None
                continue

            if m.relation is Relation.SAME:
                pid = m.old_pids[0]
                d = by_tag[(m.dim, m.new_tags[0])]
                r = self.records[pid]
                r.tag, r.descriptor, r.last_seen = d.tag, d, label
                r.relation = Relation.SAME      # else stability() sees only births
                r.alive = True
                self._by_tag[(d.dim, d.tag)] = pid
                continue

            # SPLIT / MERGE / CREATED all mint new ids and link provenance. A split
            # child is genuinely a NEW entity -- it has different geometry -- but it
            # must remain traceable to its parent so a finding recorded against the
            # parent can be translated onto it.
            for tag in m.new_tags:
                d = by_tag[(m.dim, tag)]
                pid = _new_pid(d, self.frame, taken, self.digits)
                taken.add(pid)
                self.records[pid] = EntityRecord(
                    pid=pid, dim=d.dim, tag=d.tag, descriptor=d,
                    relation=m.relation, parents=list(m.old_pids),
                    first_seen=label, last_seen=label)
                self._by_tag[(d.dim, d.tag)] = pid
                for p in m.old_pids:
                    parent = self.records[p]
                    parent.children.append(pid)
                    parent.alive = False
                    parent.tag = None
        return dict(self._by_tag)

    # -- queries -------------------------------------------------------------
    def pid_for_tag(self, dim: int, tag: int) -> str | None:
        return self._by_tag.get((dim, tag))

    def tag_for_pid(self, pid: str) -> int | None:
        r = self.records.get(pid)
        return r.tag if r and r.alive else None

    def live(self, dim: int | None = None) -> list[EntityRecord]:
        return [r for r in self.records.values()
                if r.alive and (dim is None or r.dim == dim)]

    def descendants(self, pid: str) -> list[str]:
        """All live ids derived from pid, following the provenance graph.

        This is what lets a finding recorded against a pre-imprint face be applied
        to the faces it became.
        """
        out, stack, seen = [], [pid], {pid}
        while stack:
            cur = stack.pop()
            r = self.records.get(cur)
            if r is None:
                continue
            if r.alive:
                out.append(cur)
            for c in r.children:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return sorted(out)

    def ancestors(self, pid: str) -> list[str]:
        out, stack, seen = [], [pid], {pid}
        while stack:
            cur = stack.pop()
            r = self.records.get(cur)
            if r is None:
                continue
            for p in r.parents:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
                    stack.append(p)
        return sorted(out)

    def stability(self, dim: int | None = None) -> dict:
        """Summary of how well identity survived. The number to watch."""
        rec = [r for r in self.records.values() if dim is None or r.dim == dim]
        live = [r for r in rec if r.alive]
        by_rel: dict[str, int] = {}
        for r in live:
            by_rel[r.relation.value] = by_rel.get(r.relation.value, 0) + 1
        n_same = by_rel.get("same", 0) + by_rel.get("created", 0) if len(
            self.snapshots) == 1 else by_rel.get("same", 0)
        return {"snapshots": list(self.snapshots), "n_records": len(rec),
                "n_live": len(live), "by_relation": by_rel,
                "preserved": n_same,
                "preserved_frac": (n_same / len(live)) if live else 0.0}

    def to_dict(self) -> dict:
        return {
            "frame": asdict(self.frame) if self.frame else None,
            "digits": self.digits,
            "snapshots": list(self.snapshots),
            "entities": [
                {"pid": r.pid, "dim": r.dim, "tag": r.tag, "alive": r.alive,
                 "relation": r.relation.value, "parents": r.parents,
                 "children": r.children, "first_seen": r.first_seen,
                 "last_seen": r.last_seen,
                 "mass": r.descriptor.mass if r.descriptor else None,
                 "centroid": list(r.descriptor.centroid) if r.descriptor else None}
                for r in sorted(self.records.values(), key=lambda x: (x.dim, x.pid))
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
