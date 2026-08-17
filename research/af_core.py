"""
Shared advancing-front infrastructure. PURE -- no gmsh, no surface knowledge.

Both AF methods run the identical loop:

    1. pick the shortest active front edge
    2. propose an ideal apex ahead of it
    3. gather nearby existing front nodes as alternative apexes
    4. take the first candidate that yields a VALID triangle
    5. commit: emit the triangle and update the front
    6. repeat until the front is empty

Everything above lives here. What differs between the two methods is only the
geometry of steps 2 and 4:

    parametric AF  -- works in metric-normalized xi coordinates, so steps 2 and 4
                      are ordinary 2D operations
    direct 3D AF   -- proposes the apex in R^3 and projects it onto the surface,
                      then validates in a tangent plane at the front edge

Sharing the loop is what makes the comparison meaningful. If each method had its
own front management, differences in edge ordering or closure heuristics would be
indistinguishable from differences in the actual meshing strategy.

The front update rule (step 5) is the classic one and is easy to get subtly wrong:
advancing edge (i,j) with apex k removes (i,j), then for each of (j,k) and (k,i),
CANCELS the opposite directed edge if it is already on the front, otherwise adds
it. Cancellation is how the front closes on itself; without it the front never
empties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Spatial hash
# ---------------------------------------------------------------------------

class SpatialHash:
    """Uniform grid over node coordinates, for radius queries.

    Advancing front asks "which existing nodes are near this point?" once per
    candidate. Linear scans make that O(n) and the whole mesher O(n^2), which is
    the same trap the CDT's point location fell into.
    """

    def __init__(self, cell: float, dim: int = 2):
        self.cell = max(float(cell), 1e-12)
        self.dim = dim
        self._grid: dict[tuple, list[int]] = {}

    def _key(self, p) -> tuple:
        return tuple(int(math.floor(p[d] / self.cell)) for d in range(self.dim))

    def add(self, idx: int, p) -> None:
        self._grid.setdefault(self._key(p), []).append(idx)

    def remove(self, idx: int, p) -> None:
        bucket = self._grid.get(self._key(p))
        if bucket and idx in bucket:
            bucket.remove(idx)

    def near(self, p, radius: float) -> list[int]:
        r = max(1, int(math.ceil(radius / self.cell)))
        base = self._key(p)
        out: list[int] = []
        if self.dim == 2:
            for du in range(-r, r + 1):
                for dv in range(-r, r + 1):
                    out.extend(self._grid.get((base[0] + du, base[1] + dv), ()))
        else:
            for du in range(-r, r + 1):
                for dv in range(-r, r + 1):
                    for dw in range(-r, r + 1):
                        out.extend(self._grid.get(
                            (base[0] + du, base[1] + dv, base[2] + dw), ()))
        return out


# ---------------------------------------------------------------------------
# Front
# ---------------------------------------------------------------------------

@dataclass
class AFStats:
    n_triangles: int = 0
    n_new_nodes: int = 0
    n_existing_node_reuse: int = 0
    n_edge_retries: int = 0
    n_relaxed_accepts: int = 0
    stalled_edges: int = 0
    front_remaining: int = 0
    completed: bool = False
    surface_evals: int = 0

    def summary(self) -> str:
        state = "closed" if self.completed else f"STALLED ({self.front_remaining} edges)"
        return (f"{self.n_triangles} tris, {self.n_new_nodes} new nodes, "
                f"{self.n_existing_node_reuse} reuses, "
                f"{self.n_relaxed_accepts} relaxed, front {state}")


class Front:
    """Directed active edges plus the node store they index into.

    Orientation convention: for a directed front edge (i, j), the un-meshed
    material lies to the LEFT. FaceBoundary.orient_loops guarantees this by making
    the outer loop CCW and inner loops CW, so traversing each loop in its own
    direction keeps material on the left everywhere.
    """

    def __init__(self, coords, cell: float, dim: int = 2):
        self.pts: list[np.ndarray] = [np.asarray(p, float) for p in coords]
        self.dim = dim
        self.edges: set[tuple[int, int]] = set()
        self.triangles: list[tuple[int, int, int]] = []
        self.hash = SpatialHash(cell, dim)
        # Only FRONT nodes go in the hash. A node fully surrounded by triangles is
        # no longer a valid apex, and keeping it would waste every proximity query.
        self._on_front: dict[int, int] = {}
        for i, p in enumerate(self.pts):
            self._on_front[i] = 0

    # -- nodes ---------------------------------------------------------------
    def add_node(self, p) -> int:
        self.pts.append(np.asarray(p, float))
        idx = len(self.pts) - 1
        self._on_front[idx] = 0
        return idx

    def front_nodes_near(self, p, radius: float) -> list[int]:
        return [i for i in set(self.hash.near(p, radius))
                if self._on_front.get(i, 0) > 0]

    # -- edges ---------------------------------------------------------------
    def add_edge(self, i: int, j: int) -> None:
        if (i, j) in self.edges:
            return
        self.edges.add((i, j))
        for v in (i, j):
            if self._on_front[v] == 0:
                self.hash.add(v, self.pts[v])
            self._on_front[v] += 1

    def remove_edge(self, i: int, j: int) -> None:
        if (i, j) not in self.edges:
            return
        self.edges.discard((i, j))
        for v in (i, j):
            self._on_front[v] -= 1
            if self._on_front[v] == 0:
                self.hash.remove(v, self.pts[v])

    def seed_from_loops(self, loops_global_edges) -> None:
        for i, j in loops_global_edges:
            self.add_edge(i, j)

    def edge_length(self, e) -> float:
        i, j = e
        return float(np.linalg.norm(self.pts[j] - self.pts[i]))

    def shortest_edges(self):
        """Front edges, shortest first.

        Shortest-first is the standard ordering and it matters: advancing the
        smallest gap keeps the front from leaving narrow slivers that no valid
        triangle can fill later.
        """
        return sorted(self.edges, key=self.edge_length)

    # -- commit --------------------------------------------------------------
    def commit(self, i: int, j: int, k: int) -> None:
        """Emit triangle (i,j,k) and update the front.

        Cancellation is the crux: if the reverse of a new edge is already active,
        the two annihilate and the front closes there. Only genuinely new edges are
        added. Skip this and the front never empties.
        """
        self.triangles.append((i, j, k))
        self.remove_edge(i, j)
        # The new front edges are (k,j) and (i,k) -- the REVERSES of the triangle's
        # CCW traversal (j,k) and (k,i).
        #
        # Reason: front edges carry unmeshed material on their left. Triangle
        # (i,j,k) is CCW, so the TRIANGLE lies to the left of j->k and k->i. The
        # material still to be meshed is on the opposite side, hence the reversal.
        #
        # Using the un-reversed edges is a silent catastrophe rather than an error.
        # On a unit square the boundary already contains the closing edge (15,0);
        # adding (15,0) again is a no-op, so the front never shrinks and the mesher
        # emitted 200000 triangles over 14 nodes before hitting its cap. With the
        # reversal, (0,15) cancels against the active (15,0) and the front closes.
        for a, b in ((k, j), (i, k)):
            if (b, a) in self.edges:
                self.remove_edge(b, a)
            else:
                self.add_edge(a, b)


# ---------------------------------------------------------------------------
# 2D geometry used by both methods
# ---------------------------------------------------------------------------

def left_normal_2d(d: np.ndarray) -> np.ndarray:
    """Unit normal 90 degrees counter-clockwise from d, i.e. pointing left."""
    n = np.array([-d[1], d[0]], float)
    L = float(np.linalg.norm(n))
    return n / L if L > 0 else n


def apex_height(base_length: float, target: float) -> float:
    """Height of the apex above the base, for legs of about `target`.

    Shared by BOTH methods on purpose. This single number sets how fast the front
    marches, and letting the two methods compute it differently would make their
    element counts incomparable -- a fairness leak that would be invisible in the
    results.
    """
    half = 0.5 * base_length
    if target > half:
        h = math.sqrt(max(target * target - half * half, 0.0))
    else:
        h = half * math.sqrt(3.0)
    return max(h, 0.35 * base_length)     # never propose a degenerate flat triangle


def ideal_apex_2d(pa: np.ndarray, pb: np.ndarray, target: float) -> np.ndarray:
    """Apex of an isoceles triangle on base (pa,pb), opening to the LEFT.

    Height is chosen so the two new legs are about `target` long. When the base is
    already longer than 2*target an equilateral height is used instead, since
    demanding shorter legs than half the base is impossible.
    """
    d = pb - pa
    L = float(np.linalg.norm(d))
    if L <= 0:
        return pa.copy()
    return pa + 0.5 * d + left_normal_2d(d) * apex_height(L, target)


def shape_quality_2d(pa, pb, pc) -> float:
    """4*sqrt(3)*A / sum(l^2). Same formula as quality.shape_quality, in 2D.

    Deliberately identical so a candidate's predicted quality is on the same scale
    as the number the final mesh will be scored with.
    """
    ab, bc, ca = pb - pa, pc - pb, pa - pc
    area2 = ab[0] * (-ca[1]) - ab[1] * (-ca[0])
    A = 0.5 * area2
    s = float(ab @ ab + bc @ bc + ca @ ca)
    if s <= 0:
        return 0.0
    return float(4.0 * math.sqrt(3.0) * A / s)


def triangle_is_valid_2d(front: Front, i: int, j: int, k: int, pk: np.ndarray,
                         proj, nearby, *, min_quality: float,
                         min_node_dist: float) -> bool:
    """Shared validation, in whatever 2D frame `proj` maps nodes into.

    proj(idx) -> 2D position of an existing node; pk is the candidate apex already
    in that frame. The direct-3D method passes a tangent-plane projection here, the
    parametric method passes identity. That is the whole reason validation is
    shared rather than duplicated.

    `nearby` is supplied by the caller rather than queried here, because the two
    methods search in different spaces: the parametric method queries its 2D hash
    directly, while the 3D method queries a 3D hash and then projects.

    Four checks, each guarding a distinct failure:
      * orientation/quality -- rejects inverted and sliver triangles
      * apex not too close to an unrelated front node -- prevents near-duplicate
        nodes that later make un-fillable slivers
      * new legs must not cross the front -- prevents overlapping triangles
      * no front node strictly inside -- prevents swallowing part of the front
    """
    from .predicates import orient2d, point_in_triangle, segments_properly_intersect

    pi, pj = proj(i), proj(j)
    if orient2d(tuple(pi), tuple(pj), tuple(pk)) <= 0:
        return False
    if shape_quality_2d(pi, pj, pk) < min_quality:
        return False

    for n in nearby:
        if n in (i, j, k):
            continue
        if float(np.linalg.norm(proj(n) - pk)) < min_node_dist:
            return False

    # New legs must not properly cross any active front edge.
    for (a, b) in front.edges:
        if (a, b) == (i, j):
            continue
        pa, pb = proj(a), proj(b)
        for (u, v, pu, pv) in ((j, k, pj, pk), (k, i, pk, pi)):
            if len({a, b, u, v}) < 4:
                continue
            if segments_properly_intersect(tuple(pu), tuple(pv),
                                           tuple(pa), tuple(pb)):
                return False

    for n in nearby:
        if n in (i, j, k):
            continue
        if point_in_triangle(tuple(proj(n)), tuple(pi), tuple(pj), tuple(pk)):
            return False
    return True


@dataclass
class AFConfig:
    """Tunables shared by both methods, so neither gets an unfair advantage."""
    min_quality_strict: float = 0.30
    min_quality_relaxed: float = 0.05
    reuse_radius_factor: float = 1.6
    """Search radius for existing front nodes, in units of target size."""
    min_node_dist_factor: float = 0.45
    """Reject an apex closer than this * target to an unrelated front node."""
    max_triangles: int = 200000
    candidate_limit: int = 12
    """How many existing nodes to try per edge, best-scoring first."""
    max_edge_factor: float = 2.2
    """Reject a candidate whose new legs exceed this * target. Without it the
    front happily spans the whole domain by reusing a far boundary node."""
    force_reuse_factor: float = 0.55
    """An existing node within this * target of the ideal apex MUST be reused: a
    new node that close would be a near-duplicate and would leave an un-fillable
    sliver between them."""


def order_candidates(pi, pj, ideal, existing, proj, target: float, cfg: "AFConfig"):
    """Ordered apex candidates as (node_id_or_None, position).

    node_id None means "create a new node here". Order matters more than any other
    heuristic in the method:

      1. FORCED REUSE -- existing nodes within force_reuse_factor*target of the
         ideal apex, nearest first. A new node that close would be a near-duplicate
         and the sliver between the two would be un-fillable.
      2. THE IDEAL NEW NODE -- the default. Growing inward with fresh, well-placed
         nodes is what produces near-equilateral interior elements.
      3. REMAINING EXISTING NODES, best resulting shape quality first. This is the
         closure mechanism, used when the ideal node is invalid.

    Getting this order wrong is not subtle in its effects but is silent in its
    cause. Ranking existing nodes by distance and trying them BEFORE the ideal node
    meant a far boundary node always beat creating a new one, so a unit square at
    target 0.25 came out as 14 triangles with a full-diagonal edge of length 1.0
    instead of ~37 well-shaped ones.
    """
    forced, rest = [], []
    for n in existing:
        d = float(np.linalg.norm(proj(n) - ideal))
        if d < cfg.force_reuse_factor * target:
            forced.append((d, n))
        else:
            rest.append(n)
    forced.sort()

    scored = []
    for n in rest:
        pn = proj(n)
        q = shape_quality_2d(pi, pj, pn)
        longest = max(float(np.linalg.norm(pn - pi)),
                      float(np.linalg.norm(pn - pj)))
        if longest > cfg.max_edge_factor * target:
            continue                       # would span too far
        scored.append((-q, n))
    scored.sort()

    out: list[tuple[int | None, np.ndarray]] = [(n, proj(n)) for _, n in forced]
    out.append((None, ideal))
    out.extend((n, proj(n)) for _, n in scored[:cfg.candidate_limit])
    return out
