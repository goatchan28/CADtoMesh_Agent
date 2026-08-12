"""
Tests for the sliver metric and threshold config. No gmsh needed.

Regression: sliver_block.step produced four curve.tiny warnings but ZERO
face.sliver, even though it has two 50 x 0.005 rail faces. Cause: face.sliver
tested AREA. The rail faces have area 0.25, which is 4e-5 of scale^2 -- far above
the 1e-7 "tiny" cut. They are thin, not small. Area cannot detect a sliver at any
threshold; circularity can.
"""

import math
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage0.config import Thresholds, DEFAULTS
from stage0.report import Finding, Severity


def _circ(a, p):
    """Local copy of the formula so this test does not import gmsh."""
    return 0.0 if p <= 0 else 4.0 * math.pi * a / (p * p)


def test_circularity_reference_shapes():
    assert abs(_circ(math.pi, 2 * math.pi) - 1.0) < 1e-12, "circle must be 1.0"
    assert abs(_circ(1.0, 4.0) - math.pi / 4) < 1e-12, "square must be pi/4"


def test_circularity_rectangle_formula():
    """For a x b, circularity == pi*r/(1+r)^2 with r = a/b."""
    for r in (1.0, 2.0, 10.0, 100.0, 1000.0):
        a, b = r, 1.0
        assert abs(_circ(a * b, 2 * (a + b)) - math.pi * r / (1 + r) ** 2) < 1e-12, r


def test_rail_sliver_is_caught_but_area_is_not():
    """THE REGRESSION, with the real numbers from sliver_block.step."""
    diag = 76.81
    A, P = 50 * 0.005, 2 * (50 + 0.005)

    area_frac = A / diag**2
    assert area_frac > DEFAULTS.tiny_face_area_frac, (
        f"area fraction {area_frac:.3e} is ABOVE the tiny-face cut "
        f"{DEFAULTS.tiny_face_area_frac:.3e} -- this is why the area test missed it")

    c = _circ(A, P)
    assert c < DEFAULTS.sliver_circularity, (
        f"circularity {c:.3e} must be below the cut "
        f"{DEFAULTS.sliver_circularity:.3e}")


def test_legitimate_faces_not_flagged():
    """Faces that must NOT trip the sliver cut."""
    cut = DEFAULTS.sliver_circularity
    assert _circ(1.0, 4.0) > cut, "a square is not a sliver"
    # block_hole top face: filleted 60x40 outline with an r=8 hole punched out.
    outer_A = 60 * 40 - 4 * (16 - math.pi * 16 / 4)
    outer_P = 2 * (60 + 40) - 8 * 4 + 4 * (2 * math.pi * 4 / 4)
    c = _circ(outer_A - math.pi * 64, outer_P + 2 * math.pi * 8)
    assert c > cut, f"block_hole top face {c:.4f} must clear the cut {cut}"
    assert _circ(10.0, 22.0) > cut, "a 10:1 face is not a sliver"


def test_default_cut_is_about_60_to_1():
    """Document what the default actually means in aspect-ratio terms."""
    def aspect(c):
        b = 2.0 - math.pi / c
        return (-b + math.sqrt(b * b - 4.0)) / 2.0
    r = aspect(DEFAULTS.sliver_circularity)
    assert 40 < r < 90, f"default cut implies {r:.0f}:1, expected roughly 60:1"


def test_config_from_toml_roundtrip():
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write("[thresholds]\nsliver_circularity = 0.02\ntiny_curve_frac = 5e-5\n")
        path = fh.name
    try:
        th = Thresholds.from_toml(path)
        assert th.sliver_circularity == 0.02
        assert th.tiny_curve_frac == 5e-5
        assert th.anisotropy_warn == DEFAULTS.anisotropy_warn, "unset keys keep defaults"
    finally:
        os.unlink(path)


def test_config_rejects_unknown_keys():
    """A typo in the TOML must fail loudly, not silently do nothing."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write("[thresholds]\nsliver_circularityy = 0.02\n")
        path = fh.name
    try:
        try:
            Thresholds.from_toml(path)
        except ValueError as e:
            assert "sliver_circularityy" in str(e)
        else:
            raise AssertionError("expected ValueError on unknown key")
    finally:
        os.unlink(path)


def test_thresholds_serialize_for_report_provenance():
    d = DEFAULTS.to_dict()
    assert "sliver_circularity" in d and "tiny_curve_frac" in d
    assert all(isinstance(v, (int, float)) for v in d.values())


# ---------------------------------------------------------------------------
# Borderline detection.
# thin_plate.step's sliver faces measured 4.574e-2 against a 5e-2 cut: ratio
# 0.91. A slightly different default would have reported nothing. Findings that
# close to their threshold must be flagged, because they are the only ones that
# tell you anything about the threshold.
# ---------------------------------------------------------------------------

def _f(threshold, measured):
    return Finding(code="face.sliver", severity=Severity.WARN, message="",
                   threshold_name="sliver_circularity",
                   threshold=threshold, measured=measured)


def test_thin_plate_sliver_is_borderline():
    f = _f(5e-2, 4.574e-2)
    assert abs(f.margin - 0.9148) < 1e-3, f.margin
    assert f.borderline, "ratio 0.91 must flag as borderline"


def test_sliver_block_rail_is_not_borderline():
    """3.141e-4 against a 5e-2 cut: ratio 0.0063, two orders clear."""
    f = _f(5e-2, 3.141e-4)
    assert f.margin < 0.01
    assert not f.borderline


def test_borderline_window_is_symmetric_in_ratio():
    factor = DEFAULTS.borderline_factor
    assert _f(1.0, 1.0).borderline
    assert _f(1.0, factor * 0.99).borderline
    assert not _f(1.0, factor * 1.01).borderline
    assert _f(1.0, 1.01 / factor).borderline
    assert not _f(1.0, 0.99 / factor).borderline


def test_borderline_handles_missing_provenance():
    """Findings with no threshold must not crash or claim borderline."""
    bare = Finding(code="topology.no_volume", severity=Severity.BLOCK, message="")
    assert bare.margin is None
    assert not bare.borderline
    assert not _f(0.0, 1.0).borderline, "zero threshold must not divide"


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
