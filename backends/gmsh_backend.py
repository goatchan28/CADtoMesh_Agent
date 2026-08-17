"""
Gmsh production meshing backend. Black box.

Only gmsh's public API is used: build/import geometry, set per-face size fields,
generate(2), read the mesh. No inspection or reproduction of its algorithms.

Two design choices worth stating
--------------------------------
1. SIZE IS RESOLVED PER FACE, THEN COMPILED TO SPATIAL RAMPS. The size field is
   resolved to one number per face (pipeline.sizefield), then applied as a
   Distance + Threshold pair per fine face over a coarse MathEval background,
   combined with Min.

   The obvious implementation -- one Constant field per face -- was tried and is
   wrong. A constant cannot ramp INSIDE a face, so elements of two different sizes
   meet at every shared edge and the measured gradation equals the raw face-size
   ratio: 2.44 on block_hole against a 1.5 limit. Per-face smoothing only trades
   that against element count, one for one. Threshold ramps make the gradation
   limit achievable by the field itself, which is what lets the adaptive loop treat
   gradation and size adequation as independent objectives.

   Per-face resolution is still what makes the result auditable: we know exactly
   what was requested where, so quality.size_adequation can score the mesh against
   the REQUEST. A curvature-refined face would otherwise read as an
   oversized-element failure against a global nominal.

2. PROVENANCE IS RECORDED AT EXTRACTION. Every triangle carries the PID of the face
   that produced it. Without it the pipeline can only make global changes, and the
   loop degenerates into a parameter sweep.

Volume meshing is deliberately absent. This backend generates surface meshes only.
"""

from __future__ import annotations

import math
import time

import numpy as np

from backends.base import (MeshProvenance, MeshRequest, MeshResult, MeshStats,
                           MeshingStrategy)
from pipeline.entities import EntityRegistry
from quality.core import SurfaceMesh

# Gmsh 2D algorithm tokens. Opaque selectors -- the loop swaps them, never
# interprets them.
ALGORITHMS = {"meshadapt": 1, "delaunay": 5, "frontal": 6, "default": 6,
              "bamg": 7, "frontal_quad": 8, "packing": 9}


class GmshUnavailable(RuntimeError):
    pass


class GmshBackend:
    """MeshingBackend over gmsh/OCC."""

    name = "gmsh"

    def __init__(self, verbose: bool = False):
        try:
            import gmsh
        except ImportError as e:
            raise GmshUnavailable("gmsh is not installed") from e
        self._gmsh = gmsh
        self._verbose = verbose
        self._initialized = False
        self.registry = EntityRegistry()
        self.frame = None
        self._pid_of_tag: dict[int, str] = {}
        self._tag_of_pid: dict[str, int] = {}
        self._thickness: dict[str, float] = {}
        self._interface_pids: tuple[str, ...] = ()
        self._strategy = None
        self._ramps: list[dict] = []
        self._h_base: float = 0.0

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self):
        self._gmsh.initialize()
        self._gmsh.option.setNumber("General.Terminal", 1 if self._verbose else 0)
        self._initialized = True
        return self

    def __exit__(self, *exc):
        if self._initialized:
            self._gmsh.finalize()
            self._initialized = False
        return False

    # -- load ----------------------------------------------------------------
    def load(self, path: str, strategy: MeshingStrategy = MeshingStrategy()) -> dict:
        """Import CAD, optionally heal/imprint, and register persistent ids.

        Healing defaults OFF in MeshingStrategy for the reason Stage 0 established:
        forcing repair on import destroys valid geometry (it strips a sphere's
        degenerate pole edges and sews coincident faces of touching solids). It is a
        remedial action the loop elects after evidence, not an import setting.
        """
        from backends.base import check_cad_format
        check_cad_format(path)
        self._strategy = strategy

        g = self._gmsh
        from pipeline import gmsh_entities as GE

        g.clear()
        if strategy.geometry_tolerance is not None:
            g.option.setNumber("Geometry.Tolerance", strategy.geometry_tolerance)
        flag = 1 if strategy.heal else 0
        for opt in ("OCCFixDegenerated", "OCCFixSmallEdges", "OCCFixSmallFaces",
                    "OCCSewFaces"):
            g.option.setNumber(f"Geometry.{opt}", flag)
        g.option.setNumber("Geometry.ReparamOnFaceRobust", 1)

        g.model.occ.importShapes(path)
        heal_info = None
        if strategy.heal:
            g.model.occ.synchronize()
            before = len(g.model.getEntities(2))
            scale = self.model_scale()
            tol = float(strategy.heal_tolerance_frac) * scale
            try:
                g.model.occ.healShapes(tolerance=tol)
            except TypeError:
                # Older signatures take positional arguments only.
                g.model.occ.healShapes([], tol)
            g.model.occ.synchronize()
            after = len(g.model.getEntities(2))
            heal_info = {"tolerance": tol, "faces_before": before,
                         "faces_after": after, "removed": before - after}
        g.model.occ.synchronize()

        if self.frame is None:
            self.frame = GE.model_frame()
        self.registry.register(GE.snapshot(dims=(2,)), self.frame, "import")

        report = {"imported_faces": len(g.model.getEntities(2)),
                  "volumes": len(g.model.getEntities(3)), "imprinted": False}
        if heal_info is not None:
            # Reported so an ineffective heal is visible rather than inferred.
            report["heal"] = heal_info

        if strategy.imprint:
            vols = g.model.getEntities(3)
            if len(vols) >= 2:
                g.model.occ.fragment(vols, [])
                g.model.occ.removeAllDuplicates()
                g.model.occ.synchronize()
                self.registry.register(GE.snapshot(dims=(2,)), self.frame,
                                       "imprint")
                report["imprinted"] = True
                report["faces_after_imprint"] = len(g.model.getEntities(2))

        self._refresh_maps()
        self._interface_pids = self._detect_interface_faces()
        report["interface_faces"] = len(self._interface_pids)
        report["entities"] = self.registry.stability(dim=2)
        return report

    def _detect_interface_faces(self) -> tuple[str, ...]:
        """Faces bounding two or more volumes: internal walls after imprinting.

        They belong in the mesh (volume meshing needs them) but not in the exterior
        skin, so topology checks must exclude them or a correct assembly fails the
        validity gate.
        """
        g = self._gmsh
        owners: dict[int, set[int]] = {}
        for _, vtag in g.model.getEntities(3):
            for _, ftag in g.model.getBoundary([(3, vtag)], combined=False,
                                               oriented=False):
                owners.setdefault(abs(ftag), set()).add(vtag)
        return tuple(sorted(self._pid_of_tag[t] for t, v in owners.items()
                            if len(v) >= 2 and t in self._pid_of_tag))

    @property
    def interface_pids(self) -> tuple[str, ...]:
        return self._interface_pids

    def measure_thickness(self, *, size_frac: float = 0.03,
                          max_samples_per_face: int = 60,
                          percentile: float = 5.0) -> dict[str, float]:
        """Local wall thickness per face, by inward ray casting. Populates the
        thickness sizing rule.

        Without this the thickness rule is inert -- faces[pid]["thickness"] is None
        and the rule silently does nothing. Measured on thin_plate.step: the 100 x
        1.5 side faces were sized at the 7.07 base, giving 12.7 degree minimum
        angles, when 1.5/3 = 0.5 was wanted.

        Rays are cast against the EXTERIOR SKIN only. On an imprinted assembly the
        internal interface would otherwise stop every ray at the internal wall, so
        two bonded 30mm blocks would report 30mm through a 60mm continuous region --
        the same artifact Stage 0's thickness pass had to fix.
        """
        import trimesh
        g = self._gmsh

        g.model.mesh.clear()
        scale = self.model_scale()
        h = size_frac * scale
        for opt, val in (("Mesh.MeshSizeMin", h * 0.2), ("Mesh.MeshSizeMax", h),
                         ("Mesh.MeshSizeFromCurvature", 12),
                         ("Mesh.MeshSizeFromPoints", 0),
                         ("Mesh.MeshSizeExtendFromBoundary", 0)):
            g.option.setNumber(opt, val)
        # Honour the CURRENT algorithm. This pass meshes the whole model, so on a
        # part with a degenerate parametrization it raises exactly the error the
        # policy just switched algorithm to avoid -- and without this line it would
        # keep using the default and defeat the recovery it is running inside.
        algo = getattr(self._strategy, "algorithm", "default")
        g.option.setNumber("Mesh.Algorithm",
                           ALGORITHMS.get(algo, ALGORITHMS["default"]))
        g.model.mesh.generate(2)

        node_tags, coords, _ = g.model.mesh.getNodes()
        coords = np.asarray(coords, float).reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(node_tags)}

        iface = set(self._interface_pids)
        tris: list[list[int]] = []
        tri_pid: list[str] = []
        for tag, pid in sorted(self._pid_of_tag.items()):
            if pid in iface:
                continue                       # skin only
            etypes, _, enodes = g.model.mesh.getElements(2, tag)
            for et, nodes in zip(etypes, enodes):
                if et != 2:
                    continue
                for row in np.asarray(nodes).reshape(-1, 3):
                    tris.append([index[int(v)] for v in row])
                    tri_pid.append(pid)
        g.model.mesh.clear()
        if not tris:
            return {}

        mesh = trimesh.Trimesh(vertices=coords, faces=np.asarray(tris, np.int64),
                               process=False)
        centers, normals = mesh.triangles_center, mesh.face_normals
        eps = 1e-5 * scale

        by_pid: dict[str, list[int]] = {}
        for i, pid in enumerate(tri_pid):
            by_pid.setdefault(pid, []).append(i)

        rng = np.random.default_rng(0)
        out: dict[str, float] = {}
        for pid, idx in by_pid.items():
            sel = (rng.choice(idx, max_samples_per_face, replace=False)
                   if len(idx) > max_samples_per_face else np.asarray(idx))
            origins = centers[sel] - normals[sel] * eps
            dirs = -normals[sel]
            locs, ray_idx, _ = mesh.ray.intersects_location(
                origins, dirs, multiple_hits=False)
            if len(ray_idx) == 0:
                continue
            d = np.linalg.norm(locs - origins[ray_idx], axis=1)
            if d.size:
                out[pid] = float(np.percentile(d, percentile))
        self._thickness = out
        return out

    def _refresh_maps(self) -> None:
        self._pid_of_tag, self._tag_of_pid = {}, {}
        for r in self.registry.live(dim=2):
            if r.tag is not None:
                self._pid_of_tag[r.tag] = r.pid
                self._tag_of_pid[r.pid] = r.tag

    def set_thickness(self, thickness_by_pid: dict[str, float]) -> None:
        """Supply Stage 0 wall-thickness data for the thickness sizing rule."""
        self._thickness = dict(thickness_by_pid)

    # -- geometry queries ----------------------------------------------------
    def model_scale(self) -> float:
        xmin, ymin, zmin, xmax, ymax, zmax = self._gmsh.model.getBoundingBox(-1, -1)
        return float(math.dist((xmin, ymin, zmin), (xmax, ymax, zmax)))

    def face_info(self, n_samples: int = 4) -> dict[str, dict]:
        """PID -> sizing inputs. curvature_max drives the curvature rule.

        MAX absolute principal curvature, not the mean used in the entity
        fingerprint. Identity wants a stable average; sizing must respond to the
        tightest radius anywhere on the face, or a large flat face with one small
        fillet region gets sized for the flat part.
        """
        g = self._gmsh
        out: dict[str, dict] = {}
        for tag, pid in self._pid_of_tag.items():
            info = {"tag": tag, "curvature_max": 0.0, "thickness":
                    self._thickness.get(pid)}
            try:
                info["area"] = float(g.model.occ.getMass(2, tag))
            except Exception:
                info["area"] = 0.0
            try:
                lo, hi = g.model.getParametrizationBounds(2, tag)
                pts: list[float] = []
                for u in np.linspace(lo[0], hi[0], n_samples + 2)[1:-1]:
                    for v in np.linspace(lo[1], hi[1], n_samples + 2)[1:-1]:
                        if g.model.isInside(2, tag, [float(u), float(v)],
                                            parametric=True):
                            pts.extend([float(u), float(v)])
                if pts:
                    cmax, cmin, _, _ = g.model.getPrincipalCurvatures(tag, pts)
                    k = max(float(np.abs(np.asarray(cmax, float)).max()),
                            float(np.abs(np.asarray(cmin, float)).max()))
                    info["curvature_max"] = k
            except Exception:
                pass
            out[pid] = info
        return out

    def face_adjacency(self) -> dict[str, set[str]]:
        """PID -> neighbouring face PIDs, via shared curves."""
        g = self._gmsh
        by_curve: dict[int, list[str]] = {}
        for tag, pid in self._pid_of_tag.items():
            for _, ctag in g.model.getBoundary([(2, tag)], combined=False,
                                               oriented=False):
                by_curve.setdefault(abs(ctag), []).append(pid)
        adj: dict[str, set[str]] = {pid: set() for pid in self._pid_of_tag.values()}
        for pids in by_curve.values():
            for a in pids:
                for b in pids:
                    if a != b:
                        adj[a].add(b)
        return adj

    # -- meshing -------------------------------------------------------------
    def mesh(self, request: MeshRequest) -> MeshResult:
        g = self._gmsh
        t0 = time.time()
        warnings: list[str] = []

        # smooth=False: gradation is enforced SPATIALLY by the Threshold ramps
        # below, so pre-smoothing per face would only coarsen the fine faces.
        info = self.face_info()
        sizes = request.size_field.resolve(info, smooth=False)

        g.model.mesh.clear()
        self._clear_fields()
        n_ramps = self._compile_size_fields(sizes, request.size_field)

        st = request.strategy
        finite = [h for h in sizes.values() if math.isfinite(h) and h > 0]
        g.option.setNumber("Mesh.MeshSizeMin", min(finite) if finite else 0.0)
        g.option.setNumber("Mesh.MeshSizeMax", max(finite) if finite else 1e22)
        # Background field owns sizing; disable the competing sources so the
        # requested size is what is actually applied.
        g.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        g.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        g.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        g.option.setNumber("Mesh.Algorithm",
                           ALGORITHMS.get(st.algorithm, ALGORITHMS["default"]))
        g.option.setNumber("Mesh.ElementOrder", st.element_order)
        g.option.setNumber("Mesh.Optimize", 1 if st.optimize_passes else 0)
        g.option.setNumber("Mesh.Smoothing", int(st.smooth_passes))
        g.option.setNumber("Mesh.RandomSeed", float(st.seed))

        g.model.mesh.generate(2)

        mesh, prov, n_empty = self._extract(sizes)
        if n_empty:
            warnings.append(f"{n_empty} face(s) produced no triangles")

        stats = MeshStats(n_triangles=mesh.n_triangles, n_vertices=mesh.n_vertices,
                          n_faces_meshed=len(prov.pids()), n_faces_empty=n_empty,
                          seconds=time.time() - t0, backend=self.name,
                          algorithm_applied=st.algorithm, warnings=warnings)
        stats.warnings.append(f"{n_ramps} Distance+Threshold ramp(s) compiled")
        return MeshResult(mesh=mesh, provenance=prov, stats=stats,
                          request_fingerprint=request.fingerprint())

    # -- size field compilation ---------------------------------------------
    def _clear_fields(self) -> None:
        g = self._gmsh
        try:
            for fid in list(g.model.mesh.field.list()):
                g.model.mesh.field.remove(int(fid))
        except Exception:
            pass          # older gmsh without field.list(); a fresh clear() suffices

    def _set_in_field(self, fid: int, src: int) -> None:
        """Threshold's source-field option was renamed IField -> InField."""
        g = self._gmsh
        for name in ("InField", "IField"):
            try:
                g.model.mesh.field.setNumber(fid, name, src)
                return
            except Exception:
                continue
        raise RuntimeError("cannot set the Threshold source field on this gmsh")

    def _compile_size_fields(self, sizes: dict[str, float], spec) -> int:
        """Distance + Threshold ramps, combined with Min.

        Replaces one Constant field per face. Constant fields cannot ramp INSIDE a
        face, so elements of two different sizes meet at every shared edge and the
        observed gradation equals the raw face-size ratio -- 2.44 on block_hole
        against a 1.5 limit, and 2.24 on sliver_block. No per-face smoothing fixes
        that without either coarsening the fine faces or refining the whole part;
        it just trades one objective for another.

        A Threshold ramps from SizeMin at the fine face to SizeMax over
        ramp_distance(), which is exactly the distance geometric growth needs. So
        the field itself now satisfies the gradation limit, and the loop can
        optimize gradation and size adequation independently.
        """
        from pipeline.sizefield import RuleKind, ramp_distance
        g = self._gmsh

        finite = [h for h in sizes.values() if math.isfinite(h) and h > 0]
        if not finite:
            return 0
        h_base = max(finite)
        grad = spec.get(RuleKind.GRADATION)
        growth = float(grad.params["max_growth"]) if grad else 1.4
        scale = self.model_scale()

        self._ramps = []
        self._h_base = float(h_base)
        field_ids: list[int] = []
        bg = g.model.mesh.field.add("MathEval")
        g.model.mesh.field.setString(bg, "F", repr(float(h_base)))
        field_ids.append(bg)

        n_ramps = 0
        for pid, h in sorted(sizes.items()):
            tag = self._tag_of_pid.get(pid)
            if tag is None or not math.isfinite(h) or h <= 0:
                continue
            if h >= h_base * 0.999:
                continue                      # already at the coarse background

            # PIN the face itself with a Constant, then ramp OUTWARD from its
            # boundary curves. Relying on Distance-with-SurfacesList to cover the
            # face interior is not safe: if that seeding does not take effect the
            # distance is measured from the boundary alone, so the MIDDLE of a
            # large fine face sits half its width away and ramps up to the coarse
            # size. Measured on thin_plate.step: 41,560 elements against a
            # requested 185,129, with in_band 0.134 -- a mesh 4.5x COARSER than
            # asked for, which is the opposite of the intended failure mode.
            #
            # Constant + boundary-Distance uses only well-supported features and
            # is exact: the face gets h everywhere, neighbours ramp away from it.
            const = g.model.mesh.field.add("Constant")
            g.model.mesh.field.setNumbers(const, "SurfacesList", [tag])
            g.model.mesh.field.setNumber(const, "VIn", float(h))
            g.model.mesh.field.setNumber(const, "VOut", 1e22)   # Min ignores it
            field_ids.append(const)

            curves = [abs(t) for _, t in g.model.getBoundary(
                [(2, tag)], combined=False, oriented=False)]
            if not curves:
                n_ramps += 1
                continue

            dist = g.model.mesh.field.add("Distance")
            g.model.mesh.field.setNumbers(dist, "CurvesList", curves)
            g.model.mesh.field.setNumber(dist, "Sampling", 40)

            ramp = ramp_distance(h, h_base, growth)
            ramp = min(max(ramp, h), 0.5 * scale)   # never zero, never model-wide

            thr = g.model.mesh.field.add("Threshold")
            self._set_in_field(thr, dist)
            g.model.mesh.field.setNumber(thr, "SizeMin", float(h))
            g.model.mesh.field.setNumber(thr, "SizeMax", float(h_base))
            g.model.mesh.field.setNumber(thr, "DistMin", 0.0)
            g.model.mesh.field.setNumber(thr, "DistMax", float(ramp))
            field_ids.append(thr)
            self._ramps.append({"pid": pid, "tag": tag, "h": float(h),
                                "ramp": float(ramp)})
            n_ramps += 1

        mid = g.model.mesh.field.add("Min")
        g.model.mesh.field.setNumbers(mid, "FieldsList", field_ids)
        g.model.mesh.field.setAsBackgroundMesh(mid)
        return n_ramps

    def _extract(self, sizes: dict[str, float]):
        """Read the mesh back, tagging each triangle with its face PID."""
        g = self._gmsh
        node_tags, coords, _ = g.model.mesh.getNodes()
        coords = np.asarray(coords, float).reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(node_tags)}

        tris: list[list[int]] = []
        tri_pid: list[str] = []
        n_empty = 0
        for tag, pid in sorted(self._pid_of_tag.items()):
            etypes, _, enodes = g.model.mesh.getElements(2, tag)
            before = len(tris)
            for et, nodes in zip(etypes, enodes):
                if et != 2:            # 2 == 3-node triangle
                    continue
                arr = np.asarray(nodes).reshape(-1, 3)
                for row in arr:
                    tris.append([index[int(v)] for v in row])
                    tri_pid.append(pid)
            if len(tris) == before:
                n_empty += 1

        if not tris:
            empty = SurfaceMesh(vertices=np.zeros((0, 3)),
                                triangles=np.zeros((0, 3), dtype=np.int64),
                                method="gmsh")
            return (empty,
                    MeshProvenance([], dict(sizes), dict(self._tag_of_pid),
                                   self._interface_pids),
                    n_empty)

        used = sorted({int(v) for t in tris for v in t})
        remap = {v: i for i, v in enumerate(used)}
        mesh = SurfaceMesh(
            vertices=coords[used],
            triangles=np.array([[remap[v] for v in t] for t in tris],
                               dtype=np.int64).reshape(-1, 3),
            method="gmsh",
            meta={"backend": self.name, "n_faces": len(self._pid_of_tag)})
        prov = MeshProvenance(tri_pid=tri_pid, requested_size=dict(sizes),
                              face_tags=dict(self._tag_of_pid),
                              interface_pids=self._interface_pids,
                              tri_field_size=self._field_size_at(mesh, tri_pid,
                                                                 sizes))
        return mesh, prov, n_empty

    def _field_size_at(self, mesh, tri_pid, sizes) -> list[float]:
        """Reproduce the compiled size field at each triangle centroid.

        Mirrors exactly what gmsh evaluates: the Min of a coarse background, each
        fine face's Constant, and each fine face's Threshold ramping from its
        boundary. Distances come from a KD-tree over sampled boundary points, which
        is the same discretization gmsh's Distance field uses.

        This exists so size adequation is scored against the field we ASKED FOR
        rather than each face's nominal. Without it every ramp-band element reads as
        out-of-band and the loop optimizes against its own ramps.
        """
        if not self._ramps or mesh.n_triangles == 0:
            return [float(sizes.get(p, self._h_base)) for p in tri_pid]
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return [float(sizes.get(p, self._h_base)) for p in tri_pid]

        g = self._gmsh
        a, b, c = mesh.corners()
        cen = (a + b + c) / 3.0
        out = np.full(len(cen), self._h_base, float)

        # A face's own Constant pins its elements regardless of distance.
        on_face = np.array([sizes.get(p, self._h_base) for p in tri_pid], float)
        out = np.minimum(out, on_face)

        for r in self._ramps:
            pts: list[list[float]] = []
            for _, ctag in g.model.getBoundary([(2, r["tag"])], combined=False,
                                               oriented=False):
                try:
                    lo, hi = g.model.getParametrizationBounds(1, abs(ctag))
                    ts = np.linspace(lo[0], hi[0], 24)
                    vals = g.model.getValue(1, abs(ctag), list(ts))
                    pts.extend(np.asarray(vals, float).reshape(-1, 3).tolist())
                except Exception:
                    continue
            if not pts:
                continue
            d, _ = cKDTree(np.asarray(pts, float)).query(cen)
            h, ramp = r["h"], max(r["ramp"], 1e-12)
            frac = np.clip(d / ramp, 0.0, 1.0)
            out = np.minimum(out, h + (self._h_base - h) * frac)
        return [float(x) for x in out]

    # -- contract inputs ------------------------------------------------------
    def free_edge_length(self) -> float:
        """Total length of CAD curves bounding exactly one face.

        This is a sheet body's expected mesh boundary. Length rather than an edge
        COUNT, because the count depends on element size while the length does not.
        Zero for a closed solid, which is itself the check.
        """
        g = self._gmsh
        owners: dict[int, int] = {}
        for _, ftag in g.model.getEntities(2):
            for _, ctag in g.model.getBoundary([(2, ftag)], combined=False,
                                               oriented=True):
                owners[abs(ctag)] = owners.get(abs(ctag), 0) + 1
        total = 0.0
        for ctag, n in owners.items():
            if n == 1:                      # bounded once == free edge
                try:
                    total += float(g.model.occ.getMass(1, ctag))
                except Exception:
                    pass
        return total

    def removed_faces(self, model_scale: float | None = None):
        """Faces the registry saw disappear -- healing's side effects, made visible.

        Healing removed two 0.005 mm rails from sliver_block and the run reported
        'accepted' with the removal buried in a load report. Surfacing it here is
        what lets the feature contract demand approval.
        """
        from pipeline.contracts import Removal
        scale = model_scale or self.model_scale()
        out = []
        for r in self.registry.records.values():
            if r.dim != 2 or r.alive or r.children or r.descriptor is None:
                continue                    # children => split/merge, not removal
            area = float(r.descriptor.mass)
            out.append(Removal(pid=r.pid, area=area,
                               area_frac=area / (scale * scale) if scale else 0.0,
                               perimeter=float(r.descriptor.perimeter),
                               reason=f"removed between '{r.first_seen}' and "
                                      f"'{r.last_seen}' snapshots"))
        return tuple(sorted(out, key=lambda x: -x.area))

    def body_kind_inputs(self) -> dict:
        g = self._gmsh
        return {"n_volumes": len(g.model.getEntities(3)),
                "n_faces": len(g.model.getEntities(2)),
                "free_edge_length": self.free_edge_length(),
                "interface_pids": tuple(self._interface_pids)}

    # -- fidelity ------------------------------------------------------------
    def surface_for(self, pid: str):
        from backends.cad_surface import CadFaceSurface
        tag = self._tag_of_pid.get(pid)
        return CadFaceSurface(tag) if tag is not None else None
