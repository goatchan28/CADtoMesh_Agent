"""
Robust geometric predicates. Pure -- no gmsh, no numpy in the exact path.

Delaunay triangulation is decided entirely by the SIGN of two determinants.
Floating point gets those signs wrong near degeneracy, and a single wrong sign
does not produce a slightly-wrong mesh -- it produces an inverted triangle, a
non-terminating cavity walk, or an infinite flip loop. This is the classic reason
hand-rolled Delaunay implementations hang on real geometry.

Strategy: a floating-point filter with an exact rational fallback. Compute in
float; if the result is not provably clear of zero, recompute with
fractions.Fraction, which represents every float exactly (a Python float IS a
binary rational) and therefore gets the sign right always.

This is a simplified adaptive filter, not Shewchuk's error-bounded version. It is
more conservative -- it falls back to exact arithmetic more often than strictly
necessary -- which costs speed and never costs correctness.
"""

from __future__ import annotations

from fractions import Fraction

FILTER_EPS = 1e-12


def _sign(x) -> int:
    """Sign as a plain Python int.

    int() coercion is required, not cosmetic: coordinates arrive as np.float64
    (any `tuple(row)` of a numpy array yields them), so `x > 0` is np.bool_ and
    `np.bool_ - np.bool_` raises TypeError. Works for float, np.float64 and
    Fraction alike.
    """
    return int(x > 0) - int(x < 0)


def orient2d(a, b, c) -> int:
    """Sign of the signed area of triangle (a, b, c).

    +1 counter-clockwise, -1 clockwise, 0 exactly collinear.
    """
    acx, acy = a[0] - c[0], a[1] - c[1]
    bcx, bcy = b[0] - c[0], b[1] - c[1]
    left, right = acx * bcy, acy * bcx
    det = left - right
    if abs(det) > FILTER_EPS * (abs(left) + abs(right)):
        return _sign(det)
    return _orient2d_exact(a, b, c)


def _orient2d_exact(a, b, c) -> int:
    ax, ay = Fraction(a[0]), Fraction(a[1])
    bx, by = Fraction(b[0]), Fraction(b[1])
    cx, cy = Fraction(c[0]), Fraction(c[1])
    return _sign((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))


def incircle(a, b, c, d) -> int:
    """Is d inside the circumcircle of the CCW triangle (a, b, c)?

    +1 strictly inside, -1 strictly outside, 0 exactly cocircular.
    Caller must pass (a, b, c) counter-clockwise; orientation flips the sign.
    """
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]

    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy

    bdxcdy, cdxbdy = bdx * cdy, cdx * bdy
    cdxady, adxcdy = cdx * ady, adx * cdy
    adxbdy, bdxady = adx * bdy, bdx * ady

    det = (alift * (bdxcdy - cdxbdy)
           + blift * (cdxady - adxcdy)
           + clift * (adxbdy - bdxady))
    permanent = (alift * (abs(bdxcdy) + abs(cdxbdy))
                 + blift * (abs(cdxady) + abs(adxcdy))
                 + clift * (abs(adxbdy) + abs(bdxady)))
    if abs(det) > FILTER_EPS * permanent:
        return _sign(det)
    return _incircle_exact(a, b, c, d)


def _incircle_exact(a, b, c, d) -> int:
    adx, ady = Fraction(a[0]) - Fraction(d[0]), Fraction(a[1]) - Fraction(d[1])
    bdx, bdy = Fraction(b[0]) - Fraction(d[0]), Fraction(b[1]) - Fraction(d[1])
    cdx, cdy = Fraction(c[0]) - Fraction(d[0]), Fraction(c[1]) - Fraction(d[1])
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    det = (alift * (bdx * cdy - cdx * bdy)
           + blift * (cdx * ady - adx * cdy)
           + clift * (adx * bdy - bdx * ady))
    return _sign(det)


def segments_properly_intersect(p1, p2, q1, q2) -> bool:
    """True if open segments p1p2 and q1q2 cross.

    Shared endpoints do NOT count -- advancing-front triangles legitimately share
    edges and vertices with the front, and treating that as an intersection would
    reject every valid candidate.
    """
    if (p1 in (q1, q2)) or (p2 in (q1, q2)):
        return False
    d1 = orient2d(p1, p2, q1)
    d2 = orient2d(p1, p2, q2)
    d3 = orient2d(q1, q2, p1)
    d4 = orient2d(q1, q2, p2)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    # Collinear overlap: treat any touching as intersecting, since a degenerate
    # sliver is as fatal to the front as a crossing.
    if d1 == d2 == d3 == d4 == 0:
        return _collinear_overlap(p1, p2, q1, q2)
    return False


def _collinear_overlap(p1, p2, q1, q2) -> bool:
    def on(seg_a, seg_b, pt):
        return (min(seg_a[0], seg_b[0]) <= pt[0] <= max(seg_a[0], seg_b[0]) and
                min(seg_a[1], seg_b[1]) <= pt[1] <= max(seg_a[1], seg_b[1]))
    return (on(p1, p2, q1) or on(p1, p2, q2)
            or on(q1, q2, p1) or on(q1, q2, p2))


def point_in_triangle(p, a, b, c) -> bool:
    """Inclusive containment for a CCW triangle."""
    d1 = orient2d(a, b, p)
    d2 = orient2d(b, c, p)
    d3 = orient2d(c, a, p)
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)


def circumcenter(a, b, c):
    """Circumcenter of a non-degenerate triangle, in the same 2D coordinates."""
    ax, ay = a
    bx, by = b[0] - ax, b[1] - ay
    cx, cy = c[0] - ax, c[1] - ay
    d = 2.0 * (bx * cy - by * cx)
    if d == 0.0:
        raise ValueError("degenerate triangle has no circumcenter")
    bl = bx * bx + by * by
    cl = cx * cx + cy * cy
    return (ax + (bl * cy - cl * by) / d, ay + (cl * bx - bl * cx) / d)
