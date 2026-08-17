"""
Mesh export. PURE -- no gmsh, no external writers.

The pipeline's stated deliverable is "a validated surface mesh". Until now it
produced a verdict, a contract report and a run record, and then discarded the
geometry -- nothing downstream could consume the result.

Two formats, for two different consumers:

  .msh (gmsh 2.2 ASCII)   the handoff to volume meshing. Carries a PHYSICAL GROUP
                          per CAD face, so face identity survives into the volume
                          mesh and boundary conditions can later be applied per
                          face rather than by picking triangles.

  .stl                    inspection only. Deliberately not the primary output: STL
                          has no face tags, no provenance, and no way to express
                          which triangles came from which CAD entity.

MSH 2.2 rather than 4.1 on purpose: 2.2 is a much smaller format to write
correctly, every tool reads it, and nothing here needs 4.1's entity section. The
PID -> physical-tag map goes in the manifest, because .msh can only carry integers
and a raw integer is not traceable back to CAD.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ExportManifest:
    """Everything needed to interpret the exported mesh later.

    A .msh with physical tag 7 is meaningless six months on. This records which CAD
    entity each tag refers to, what criteria the mesh was accepted against, and what
    was removed to get there.
    """
    model: str = ""
    mesh_file: str = ""
    n_triangles: int = 0
    n_vertices: int = 0
    physical_tags: dict = field(default_factory=dict)     # str(tag) -> PID
    requested_size: dict = field(default_factory=dict)    # PID -> size
    interface_pids: tuple = ()
    accepted: bool = False
    score: float = 0.0
    criteria: dict = field(default_factory=dict)
    contract: dict | None = None
    objectives: list = field(default_factory=list)
    iteration: int = -1

    def to_json(self, indent: int = 2) -> str:
        from dataclasses import asdict
        return json.dumps(asdict(self), indent=indent, default=str)


def write_stl(mesh, path: Path, name: str = "mesh") -> Path:
    """ASCII STL. Inspection only -- no face tags, no provenance."""
    a, b, c = mesh.corners()
    n = np.cross(b - a, c - a)
    L = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, L, out=np.zeros_like(n), where=L > 0)

    out = [f"solid {name}"]
    for i in range(len(a)):
        out.append(f"  facet normal {n[i,0]:.6e} {n[i,1]:.6e} {n[i,2]:.6e}")
        out.append("    outer loop")
        for p in (a[i], b[i], c[i]):
            out.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        out.append("    endloop")
        out.append("  endfacet")
    out.append(f"endsolid {name}")
    path.write_text("\n".join(out))
    return path


def write_msh(mesh, path: Path, tri_pid=None, name: str = "surface"):
    """gmsh 2.2 ASCII, one physical group per CAD face.

    Returns (path, {physical_tag: pid}). Physical tags are assigned in sorted PID
    order so two runs on the same part produce the same tags -- an exported mesh
    whose tags shuffle between runs cannot be diffed or scripted against.

    Node indices are 1-based, as the format requires. Element line layout is
    `id type n_tags physical elementary n1 n2 n3` with type 2 = 3-node triangle.
    """
    v = np.asarray(mesh.vertices, float)
    t = np.asarray(mesh.triangles, np.int64)

    if tri_pid and len(tri_pid) == len(t):
        pids = sorted(set(tri_pid))
        tag_of = {pid: i + 1 for i, pid in enumerate(pids)}
        elem_tag = [tag_of[p] for p in tri_pid]
    else:
        tag_of = {name: 1}
        elem_tag = [1] * len(t)

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat"]
    lines.append("$PhysicalNames")
    lines.append(str(len(tag_of)))
    for pid, tag in sorted(tag_of.items(), key=lambda kv: kv[1]):
        lines.append(f'2 {tag} "{pid}"')
    lines.append("$EndPhysicalNames")

    lines.append("$Nodes")
    lines.append(str(len(v)))
    for i, p in enumerate(v, start=1):
        lines.append(f"{i} {p[0]:.10g} {p[1]:.10g} {p[2]:.10g}")
    lines.append("$EndNodes")

    lines.append("$Elements")
    lines.append(str(len(t)))
    for i, (tri, tag) in enumerate(zip(t, elem_tag), start=1):
        lines.append(f"{i} 2 2 {tag} {tag} "
                     f"{tri[0] + 1} {tri[1] + 1} {tri[2] + 1}")
    lines.append("$EndElements")

    path.write_text("\n".join(lines) + "\n")
    return path, {v_: k for k, v_ in tag_of.items()}


def export_result(result, record, outdir, stem: str, *, criteria=None,
                  contract=None, iteration: int = -1, write_stl_too: bool = True):
    """Write mesh + manifest for an accepted (or best-valid) result.

    `result` is a backends.base.MeshResult. Returns the list of paths written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mesh, prov = result.mesh, result.provenance

    msh_path, tags = write_msh(mesh, outdir / f"{stem}.msh",
                               tri_pid=prov.tri_pid, name=stem)
    written = [msh_path]
    if write_stl_too:
        written.append(write_stl(mesh, outdir / f"{stem}.stl", stem))

    man = ExportManifest(
        model=getattr(record, "model", ""), mesh_file=msh_path.name,
        n_triangles=mesh.n_triangles, n_vertices=mesh.n_vertices,
        physical_tags={str(k): v for k, v in tags.items()},
        requested_size=dict(prov.requested_size),
        interface_pids=tuple(prov.interface_pids),
        accepted=bool(getattr(record, "outcome", None)
                      and record.outcome.succeeded),
        score=float(getattr(record.best, "score", 0.0) if record.best else 0.0),
        criteria=criteria or {}, contract=contract, iteration=iteration,
        objectives=(record.best.objectives if record.best else []))
    man_path = outdir / f"{stem}.manifest.json"
    man_path.write_text(man.to_json())
    written.append(man_path)

    rec_path = outdir / f"{stem}.run.json"
    rec_path.write_text(record.to_json())
    written.append(rec_path)
    return written
