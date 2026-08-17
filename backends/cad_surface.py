"""
Surface protocol over a live gmsh/OCC face, so quality/ can score real CAD.

quality.chordal_deviation and quality.normal_deviation need .project() and
.normal(). Against analytic surfaces those are closed form; against OCC each call
is a kernel query, and that cost is the reason this file exists as more than a
one-line adapter.

Cost, concretely: fidelity sampling is 7 barycentric points per triangle. A
100k-triangle model is 700k getClosestPoint calls. That is not usable inside an
adaptive loop that remeshes several times, so callers MUST pass max_triangles to
subsample. Attribution needs to RANK faces, not measure any one of them precisely.

Caching is keyed on rounded coordinates because the sampler revisits nearly the
same points across iterations of the loop.
"""

from __future__ import annotations

import numpy as np


class CadFaceSurface:
    """One OCC face, exposed as a Surface.

    Note the normal convention: normal() returns unit(dX/du x dX/dv), NOT gmsh's
    getNormal. The two can be opposite, and the research phase established that
    orientation must follow the (u,v) parametrization -- a mismatch there sent an
    advancing front off a sphere patch entirely. quality.normal_deviation measures
    unsigned angles, so this choice does not bias the metric; it just keeps one
    convention across the codebase.
    """

    def __init__(self, face_tag: int, cache_digits: int = 9):
        import gmsh
        self._gmsh = gmsh
        self.tag = int(face_tag)
        self._digits = cache_digits
        self._proj: dict[tuple, tuple] = {}
        self._norm: dict[tuple, np.ndarray] = {}
        self.n_project = 0
        self.n_normal = 0

    def point(self, uv) -> np.ndarray:
        p = self._gmsh.model.getValue(2, self.tag, [float(uv[0]), float(uv[1])])
        return np.asarray(p, float)

    def derivatives(self, uv):
        d = self._gmsh.model.getDerivative(
            2, self.tag, [float(uv[0]), float(uv[1])])
        assert len(d) == 6, f"unexpected getDerivative layout: {len(d)}"
        d = np.asarray(d, float)
        return d[0:3], d[3:6]

    def normal(self, uv) -> np.ndarray:
        key = (round(float(uv[0]), self._digits), round(float(uv[1]), self._digits))
        n = self._norm.get(key)
        if n is None:
            du, dv = self.derivatives(uv)
            n = np.cross(du, dv)
            L = float(np.linalg.norm(n))
            n = n / L if L > 0 else n
            self._norm[key] = n
            self.n_normal += 1
        return n

    def project(self, xyz):
        """Closest point on the face. Returns (xyz, uv), matching the protocol."""
        key = tuple(round(float(x), self._digits) for x in xyz)
        hit = self._proj.get(key)
        if hit is None:
            res = self._gmsh.model.getClosestPoint(2, self.tag,
                                                   [float(v) for v in xyz])
            hit = (np.asarray(res[0][:3], float), np.asarray(res[1][:2], float))
            self._proj[key] = hit
            self.n_project += 1
        return hit

    def is_inside(self, uv) -> bool:
        return bool(self._gmsh.model.isInside(
            2, self.tag, [float(uv[0]), float(uv[1])], parametric=True))

    @property
    def uv_bounds(self):
        lo, hi = self._gmsh.model.getParametrizationBounds(2, self.tag)
        return (lo[0], lo[1], hi[0], hi[1])

    def cost(self) -> dict:
        return {"projections": self.n_project, "normals": self.n_normal,
                "cached_projections": len(self._proj)}
