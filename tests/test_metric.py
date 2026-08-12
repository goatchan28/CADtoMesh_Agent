"""Analytic checks on the first-fundamental-form math (no gmsh needed)."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage0.metrics import metric_from_derivatives, target_size_in_parametric_space


def test_identity_plane():
    """X(u,v) = (u, v, 0): area scale 1, anisotropy 1."""
    du = np.array([[1.0, 0.0, 0.0]])
    dv = np.array([[0.0, 1.0, 0.0]])
    a, k = metric_from_derivatives(du, dv)
    assert np.allclose(a, 1.0), a
    assert np.allclose(k, 1.0), k


def test_uniform_scaling():
    """X = (3u, 3v, 0): area scale 9, still isotropic."""
    du = np.array([[3.0, 0.0, 0.0]])
    dv = np.array([[0.0, 3.0, 0.0]])
    a, k = metric_from_derivatives(du, dv)
    assert np.allclose(a, 9.0), a
    assert np.allclose(k, 1.0), k


def test_anisotropic_stretch():
    """X = (10u, v, 0): area scale 10, anisotropy 10."""
    du = np.array([[10.0, 0.0, 0.0]])
    dv = np.array([[0.0, 1.0, 0.0]])
    a, k = metric_from_derivatives(du, dv)
    assert np.allclose(a, 10.0), a
    assert np.allclose(k, 10.0), k


def test_sphere_pole_degeneracy():
    """Unit sphere X(u,v) = (cos u sin v, sin u sin v, cos v).

    dXdu = (-sin u sin v, cos u sin v, 0)  -> vanishes as v -> 0 (the pole).
    Area scale = sin v, so it collapses at the pole. This is the signature the
    DEGENERATE classification keys on.
    """
    us = np.array([0.3, 0.3, 0.3, 0.3])
    vs = np.array([np.pi / 2, 0.1, 0.01, 1e-9])
    du = np.stack([-np.sin(us) * np.sin(vs), np.cos(us) * np.sin(vs),
                   np.zeros_like(us)], axis=1)
    dv = np.stack([np.cos(us) * np.cos(vs), np.sin(us) * np.cos(vs),
                   -np.sin(vs)], axis=1)
    a, k = metric_from_derivatives(du, dv)
    assert np.allclose(a, np.sin(vs), atol=1e-12), a
    # Equator isotropic, pole unboundedly anisotropic.
    assert abs(k[0] - 1.0) < 1e-9, k
    assert k[1] > 5.0 and k[2] > 50.0, k
    assert a[-1] < 1e-8, a


def test_sheared_basis():
    """Non-orthogonal (u,v): area scale must use sqrt(EG-F^2), not sqrt(EG)."""
    du = np.array([[1.0, 0.0, 0.0]])
    dv = np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0]])
    a, k = metric_from_derivatives(du, dv)
    # Parallelogram area = |du||dv| sin(angle) = 1 * 1 * sin(30 deg) = 0.5
    assert np.allclose(a, 0.5), a
    assert k[0] > 1.0, k


def test_degenerate_is_not_negative():
    """fp noise must not produce NaN area scale."""
    du = np.array([[1.0, 0.0, 0.0]])
    dv = np.array([[1.0, 0.0, 0.0]])   # exactly parallel -> det I == 0
    a, k = metric_from_derivatives(du, dv)
    assert np.isfinite(a).all() and a[0] == 0.0, a
    assert np.isinf(k[0]), k


def test_parametric_target_size():
    """Anisotropic map: a 0.1 physical target needs different du and dv steps."""
    du = np.array([[10.0, 0.0, 0.0]])   # |Xu| = 10
    dv = np.array([[0.0, 2.0, 0.0]])    # |Xv| = 2
    h_u, h_v = target_size_in_parametric_space(du, dv, 0.1)
    assert np.allclose(h_u, 0.01), h_u
    assert np.allclose(h_v, 0.05), h_v


def test_parametric_target_size_degenerate():
    du = np.array([[0.0, 0.0, 0.0]])
    dv = np.array([[0.0, 1.0, 0.0]])
    h_u, h_v = target_size_in_parametric_space(du, dv, 0.1)
    assert np.isinf(h_u[0])
    assert np.allclose(h_v, 0.1)


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
