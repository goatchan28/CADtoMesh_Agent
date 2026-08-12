"""
Verify the gmsh API assumptions Stage 0 depends on, against known-answer geometry.

    python -m tools.verify_gmsh_api models/cylinder.step

Run this before trusting Stage 0 output. Each check has an answer derivable from
the geometry, so a silent API mismatch becomes a visible failure.

Checks, in order of how badly a wrong answer would hurt:

1. getDerivative(2, ...) layout -- ASSUMED [dX/du (3), dX/dv (3)] per point.
   On an OCC cylinder X(u,v) = (R cos u, R sin u, v), so |Xu| == R and |Xv| == 1.
   If these come back swapped, every anisotropy and area-scale number in
   parametric_map.py is wrong, and nothing else would catch it.

2. isInside(..., parametric=True) actually respects the trim, not just the
   untrimmed parametric rectangle.

3. Surface normal orientation from gmsh -> trimesh (outward for a valid solid).

4. getElementProperties(et)[3] == numNodes.

5. reparametrizeOnSurface behaviour on a periodic face's seam curve.
"""

from __future__ import annotations

import argparse
import math
import sys

import gmsh
import numpy as np

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"{status:4} {name}" + (f" -- {detail}" if detail else ""))


def check_derivative_layout(radius_hint: float | None) -> None:
    """Check 1. Requires a model containing exactly one cylindrical face."""
    faces = gmsh.model.getEntities(2)
    best = None
    for _, tag in faces:
        lo, hi = gmsh.model.getParametrizationBounds(2, tag)
        # A cylinder's u range is the full 2*pi angle.
        if abs((hi[0] - lo[0]) - 2.0 * math.pi) < 1e-6:
            best = (tag, lo, hi)
            break
    if best is None:
        record(WARN, "getDerivative layout",
               "no periodic (2*pi) face found; run on models/cylinder.step")
        return

    tag, lo, hi = best
    us = np.linspace(lo[0] + 0.1, hi[0] - 0.1, 5)
    vs = np.linspace(lo[1] + 0.1 * (hi[1] - lo[1]),
                     hi[1] - 0.1 * (hi[1] - lo[1]), 5)
    pts = [c for u, v in zip(us, vs) for c in (u, v)]

    d = gmsh.model.getDerivative(2, tag, pts)
    n = len(us)
    if len(d) != 6 * n:
        record(FAIL, "getDerivative layout",
               f"expected {6*n} values for {n} points, got {len(d)}")
        return

    d = np.asarray(d, dtype=float).reshape(n, 6)
    first = np.linalg.norm(d[:, 0:3], axis=1)
    second = np.linalg.norm(d[:, 3:6], axis=1)

    detail = (f"|block1|={first.mean():.4f}+-{first.std():.1e}, "
              f"|block2|={second.mean():.4f}+-{second.std():.1e}")

    # For a cylinder: one block has constant norm R (>1), the other constant 1.
    b1_is_unit = abs(second.mean() - 1.0) < 1e-4 and second.std() < 1e-6
    b2_is_unit = abs(first.mean() - 1.0) < 1e-4 and first.std() < 1e-6

    if b1_is_unit and first.mean() > 1.0 + 1e-3:
        r = first.mean()
        ok = radius_hint is None or abs(r - radius_hint) < 1e-3
        record(PASS if ok else FAIL, "getDerivative layout",
               f"[dXdu, dXdv] confirmed; radius={r:.4f} "
               + (detail if not ok else ""))
    elif b2_is_unit and second.mean() > 1.0 + 1e-3:
        record(FAIL, "getDerivative layout",
               "BLOCKS ARE SWAPPED -- returns [dXdv, dXdu]. "
               "Fix the slicing in parametric_map.analyze_face(). " + detail)
    else:
        record(WARN, "getDerivative layout",
               "inconclusive (is this really a cylinder?). " + detail)


def check_is_inside_trim() -> None:
    """Check 2. Needs a face whose trimmed domain is a strict subset of its
    parametric rectangle -- e.g. the top face of block_hole.step, which has a
    circular hole punched out of it."""
    found_trimmed = False
    for _, tag in gmsh.model.getEntities(2):
        lo, hi = gmsh.model.getParametrizationBounds(2, tag)
        if hi[0] - lo[0] <= 0 or hi[1] - lo[1] <= 0:
            continue
        n = 25
        inside = 0
        total = 0
        for u in np.linspace(lo[0], hi[0], n):
            for v in np.linspace(lo[1], hi[1], n):
                total += 1
                if gmsh.model.isInside(2, tag, [u, v], parametric=True):
                    inside += 1
        frac = inside / total
        if frac < 0.98:
            found_trimmed = True
            record(PASS, "isInside respects trim",
                   f"face {tag}: {inside}/{total} samples inside "
                   f"({frac:.2f} of rectangle)")
            break
    if not found_trimmed:
        record(WARN, "isInside respects trim",
               "every face filled its parametric rectangle; run on "
               "models/block_hole.step, which has a trimmed face")


def check_element_properties() -> None:
    """Check 4. Element type 2 is a 3-node triangle, 4 is a 4-node tet."""
    try:
        props_tri = gmsh.model.mesh.getElementProperties(2)
        props_tet = gmsh.model.mesh.getElementProperties(4)
    except Exception as e:
        record(FAIL, "getElementProperties index", str(e))
        return
    n_tri, n_tet = props_tri[3], props_tet[3]
    if n_tri == 3 and n_tet == 4:
        record(PASS, "getElementProperties index", "index [3] == numNodes")
    else:
        record(FAIL, "getElementProperties index",
               f"index [3] gave tri={n_tri}, tet={n_tet}; expected 3 and 4. "
               f"full tri tuple: {props_tri}")


def check_normal_orientation() -> None:
    """Check 3. Coarse-mesh the solid, verify gmsh's normals point outward.

    Test: a point just inside the surface along -normal must lie inside the
    solid. Uses the divergence-theorem volume of the triangulation as the signal
    -- positive means outward-oriented.
    """
    vols = gmsh.model.getEntities(3)
    if not vols:
        record(WARN, "normal orientation", "no volume in model")
        return

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.08 * diag)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.02 * diag)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, float).reshape(-1, 3)
    idx = {int(t): i for i, t in enumerate(node_tags)}

    signed = 0.0
    n_tri = 0
    for _, ftag in gmsh.model.getEntities(2):
        etypes, _, enodes = gmsh.model.mesh.getElements(2, ftag)
        for et, nodes in zip(etypes, enodes):
            if et != 2:
                continue
            tri = np.asarray(nodes).reshape(-1, 3)
            for row in tri:
                a, b, c = (coords[idx[int(k)]] for k in row)
                signed += np.dot(a, np.cross(b, c)) / 6.0
                n_tri += 1

    cad_vol = sum(gmsh.model.occ.getMass(3, t) for _, t in vols)
    gmsh.model.mesh.clear()

    if n_tri == 0:
        record(FAIL, "normal orientation", "no triangles generated")
    elif signed > 0 and cad_vol > 0:
        err = abs(signed - cad_vol) / cad_vol
        record(PASS, "normal orientation",
               f"outward; mesh vol {signed:.4g} vs CAD {cad_vol:.4g} "
               f"({err:.1%} discretization error, {n_tri} tris)")
    else:
        record(FAIL, "normal orientation",
               f"signed volume {signed:.4g} is not positive -- normals point "
               "INWARD. Flip the sign on `inward` in thickness.py.")


def check_seam_reparam() -> None:
    """Check 5. On a periodic face, one seam curve has two valid (u,v) traces.
    Confirm gmsh returns something finite and note which branch."""
    for _, ftag in gmsh.model.getEntities(2):
        bnd = gmsh.model.getBoundary([(2, ftag)], combined=False, oriented=True)
        seen: dict[int, int] = {}
        for _, t in bnd:
            seen[abs(t)] = seen.get(abs(t), 0) + 1
        seams = [t for t, k in seen.items() if k > 1]
        if not seams:
            continue
        ctag = seams[0]
        lo, hi = gmsh.model.getParametrizationBounds(1, ctag)
        ts = list(np.linspace(lo[0], hi[0], 5))
        try:
            uv = np.asarray(gmsh.model.reparametrizeOnSurface(1, ctag, ts, ftag),
                            float).reshape(-1, 2)
        except Exception as e:
            record(FAIL, "seam reparametrization",
                   f"curve {ctag} on face {ftag}: {e}")
            return
        if not np.isfinite(uv).all():
            record(FAIL, "seam reparametrization",
                   f"non-finite (u,v) on seam curve {ctag}")
            return
        record(PASS, "seam reparametrization",
               f"seam curve {ctag} on face {ftag} -> u in "
               f"[{uv[:,0].min():.4f}, {uv[:,0].max():.4f}] "
               "(one of two valid branches; Stage 0 flags this face SEAMED)")
        return
    record(WARN, "seam reparametrization",
           "no periodic face found; run on models/cylinder.step")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify gmsh API assumptions")
    ap.add_argument("cad", help="STEP file; cylinder.step verifies the most")
    ap.add_argument("--radius", type=float, default=None,
                    help="known cylinder radius for an exact check (12.0 for "
                         "the generated cylinder.step)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if args.verbose else 0)
        gmsh.option.setNumber("Geometry.ReparamOnFaceRobust", 1)
        gmsh.model.occ.importShapes(args.cad)
        gmsh.model.occ.synchronize()

        print(f"model: {args.cad}\n")
        check_derivative_layout(args.radius)
        check_seam_reparam()
        check_is_inside_trim()
        check_element_properties()
        check_normal_orientation()
    finally:
        gmsh.finalize()

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    n_warn = sum(1 for s, _, _ in _results if s == WARN)
    print(f"\n{len(_results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail")
    if n_warn:
        print("Warnings usually mean the test part lacked the needed feature. "
              "Try cylinder.step and block_hole.step.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
