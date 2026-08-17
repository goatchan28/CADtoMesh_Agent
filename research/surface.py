"""
Surface abstraction: the projection operator a direct-3D advancing front needs.

Why this exists as a protocol rather than direct gmsh calls: it makes the
advancing front testable WITHOUT gmsh. The analytic surfaces below (plane,
sphere, cylinder) have exact projections and exact normals, so af3d.py can be
validated end to end -- including against known answers like "a mesh on a plane
should be near-equilateral everywhere" -- before gmsh is involved at all.

That matters because the direct-3D front is the one piece of Stage 1 with no
reference implementation to lean on.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Surface(Protocol):
    """What the meshers need from a surface.

    project/normal/point are the direct-3D front's needs. derivatives/is_inside/
    uv_bounds are what the two PARAMETRIC methods add: they work in (u,v) and so
    need the first fundamental form (from the derivatives) to know what a
    parametric step costs in real space, plus a trim test to know where the domain
    actually is.
    """

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Closest point on the surface, and its (u,v). Returns (xyz, uv)."""
        ...

    def normal(self, uv: np.ndarray) -> np.ndarray:
        """Unit surface normal at (u,v)."""
        ...

    def point(self, uv: np.ndarray) -> np.ndarray:
        """Position at (u,v)."""
        ...

    def derivatives(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(dX/du, dX/dv) at (u,v). Feeds the metric tensor."""
        ...

    def is_inside(self, uv: np.ndarray) -> bool:
        """Trim test: is (u,v) in the face's actual domain?"""
        ...

    @property
    def uv_bounds(self) -> tuple[float, float, float, float]:
        """(umin, vmin, umax, vmax) of the parametric rectangle."""
        ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# Analytic surfaces -- exact, for testing
# ---------------------------------------------------------------------------

class PlaneSurface:
    """Plane through `origin` spanned by orthonormal e1, e2. Normal = e1 x e2."""

    def __init__(self, origin=(0, 0, 0), e1=(1, 0, 0), e2=(0, 1, 0)):
        self.origin = np.asarray(origin, float)
        self.e1 = _unit(np.asarray(e1, float))
        e2 = np.asarray(e2, float)
        self.e2 = _unit(e2 - np.dot(e2, self.e1) * self.e1)
        self._n = _unit(np.cross(self.e1, self.e2))

    def project(self, xyz):
        d = np.asarray(xyz, float) - self.origin
        u, v = float(np.dot(d, self.e1)), float(np.dot(d, self.e2))
        return self.point((u, v)), np.array([u, v])

    def normal(self, uv):
        return self._n.copy()

    def point(self, uv):
        u, v = float(uv[0]), float(uv[1])
        return self.origin + u * self.e1 + v * self.e2

    def derivatives(self, uv):
        # Constant, orthonormal: the metric is the identity, so parametric space
        # IS real space. This is the reference case for testing a parametric
        # mesher -- any anisotropy in the output is the algorithm's fault, not
        # the parametrization's.
        return self.e1.copy(), self.e2.copy()

    def is_inside(self, uv):
        return True

    @property
    def uv_bounds(self):
        return (-1e9, -1e9, 1e9, 1e9)


class SphereSurface:
    """Sphere. (u,v) = (azimuth, polar) with polar in (0, pi)."""

    def __init__(self, center=(0, 0, 0), radius=1.0):
        self.center = np.asarray(center, float)
        self.radius = float(radius)

    def project(self, xyz):
        d = np.asarray(xyz, float) - self.center
        n = np.linalg.norm(d)
        if n == 0:
            d = np.array([0.0, 0.0, self.radius])
            n = self.radius
        p = self.center + d * (self.radius / n)
        dn = d / n
        v = float(np.arccos(np.clip(dn[2], -1.0, 1.0)))
        u = float(np.arctan2(dn[1], dn[0]))
        return p, np.array([u, v])

    def normal(self, uv):
        p = self.point(uv)
        return _unit(p - self.center)

    def point(self, uv):
        u, v = float(uv[0]), float(uv[1])
        return self.center + self.radius * np.array(
            [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])

    def derivatives(self, uv):
        u, v = float(uv[0]), float(uv[1])
        R = self.radius
        # |dX/du| = R sin(v) -> 0 at the poles. That collapse is the whole reason
        # a sphere is in the test set: parametric methods degrade there and the
        # direct-3D front should not.
        du = R * np.array([-np.sin(u) * np.sin(v), np.cos(u) * np.sin(v), 0.0])
        dv = R * np.array([np.cos(u) * np.cos(v), np.sin(u) * np.cos(v), -np.sin(v)])
        return du, dv

    def is_inside(self, uv):
        return 0.0 <= float(uv[1]) <= np.pi

    @property
    def uv_bounds(self):
        return (-np.pi, 0.0, np.pi, np.pi)


class CylinderSurface:
    """Infinite cylinder about `axis` through `origin`. (u,v) = (angle, height)."""

    def __init__(self, origin=(0, 0, 0), axis=(0, 0, 1), radius=1.0):
        self.origin = np.asarray(origin, float)
        self.axis = _unit(np.asarray(axis, float))
        self.radius = float(radius)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(seed, self.axis)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        self.e1 = _unit(np.cross(self.axis, seed))
        self.e2 = _unit(np.cross(self.axis, self.e1))

    def project(self, xyz):
        d = np.asarray(xyz, float) - self.origin
        h = float(np.dot(d, self.axis))
        radial = d - h * self.axis
        r = np.linalg.norm(radial)
        if r == 0:
            radial, r = self.e1.copy(), 1.0
        u = float(np.arctan2(np.dot(radial, self.e2), np.dot(radial, self.e1)))
        return self.point((u, h)), np.array([u, h])

    def normal(self, uv):
        u = float(uv[0])
        return _unit(np.cos(u) * self.e1 + np.sin(u) * self.e2)

    def point(self, uv):
        u, h = float(uv[0]), float(uv[1])
        return (self.origin + h * self.axis
                + self.radius * (np.cos(u) * self.e1 + np.sin(u) * self.e2))

    def derivatives(self, uv):
        u = float(uv[0])
        # |dX/du| = R, |dX/dv| = 1: uniformly ANISOTROPIC when R != 1. A mesher
        # that ignores the metric produces stretched triangles here, which is what
        # makes the cylinder the discriminating test for metric awareness.
        du = self.radius * (-np.sin(u) * self.e1 + np.cos(u) * self.e2)
        return du, self.axis.copy()

    def is_inside(self, uv):
        return True

    @property
    def uv_bounds(self):
        return (-np.pi, -1e9, np.pi, 1e9)


# ---------------------------------------------------------------------------
# gmsh-backed surface
# ---------------------------------------------------------------------------

class GmshSurface:
    """Wraps one CAD face. Requires an initialized gmsh with the model loaded.

    VERIFY ON FIRST RUN:
      * getClosestPoint returns (coord, parametricCoord); we assume 3 + 2 values
        for a surface. The assert makes a layout change loud.
      * getNormal expects parametric coords and returns 3 values.
      * Normal ORIENTATION is the CAD face's own, which for a valid solid points
        out of the material. af3d relies on that to orient the front, and checks
        it explicitly via `outward_consistent`.
    """

    def __init__(self, face_tag: int):
        import gmsh
        self._gmsh = gmsh
        self.face_tag = int(face_tag)
        self._bounds = None

    def project(self, xyz):
        res = self._gmsh.model.getClosestPoint(
            2, self.face_tag, [float(x) for x in xyz])
        coord, param = res[0], res[1]
        assert len(coord) == 3 and len(param) == 2, (
            f"unexpected getClosestPoint layout: {len(coord)}, {len(param)}")
        return np.asarray(coord, float), np.asarray(param, float)

    def normal(self, uv):
        n = self._gmsh.model.getNormal(
            self.face_tag, [float(uv[0]), float(uv[1])])
        assert len(n) == 3, f"unexpected getNormal layout: {len(n)}"
        return _unit(np.asarray(n, float))

    def point(self, uv):
        p = self._gmsh.model.getValue(
            2, self.face_tag, [float(uv[0]), float(uv[1])])
        return np.asarray(p, float)

    def derivatives(self, uv):
        d = self._gmsh.model.getDerivative(
            2, self.face_tag, [float(uv[0]), float(uv[1])])
        # Layout [dXdu, dXdv] verified on a known cylinder by
        # tools/verify_gmsh_api.py. The assert keeps a silent change loud.
        assert len(d) == 6, f"unexpected getDerivative layout: {len(d)}"
        d = np.asarray(d, float)
        return d[0:3], d[3:6]

    def is_inside(self, uv):
        return bool(self._gmsh.model.isInside(
            2, self.face_tag, [float(uv[0]), float(uv[1])], parametric=True))

    @property
    def uv_bounds(self):
        if self._bounds is None:
            lo, hi = self._gmsh.model.getParametrizationBounds(2, self.face_tag)
            self._bounds = (lo[0], lo[1], hi[0], hi[1])
        return self._bounds
