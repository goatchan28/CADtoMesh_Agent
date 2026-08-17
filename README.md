# CADtoMesh_Agent

**Automated CAD → simulation-ready surface mesh.** Takes difficult CAD geometry and
produces a validated surface mesh, iterating on meshing parameters until explicitly
defined simulation-readiness criteria are met — or stopping and explaining why.

> ### 📄 [**RESEARCH.md**](RESEARCH.md) — methodology, experiments, results, failure analysis, and limitations
> Start there for the research narrative: the three surface meshers we built and
> measured, why we pivoted to Gmsh, what the ABC-dataset evaluation showed, and
> what does not yet work. This README is the code tour.

---

## What it does, and what it does not

**Goal.** Produce the best simulation-ready surface mesh possible from difficult
CAD, using mature meshing tools where appropriate.

**Not the goal.** Novel meshing algorithms, or replacing Gmsh. Gmsh is used as a
black box; the contribution is the intelligence around it.

**Scope.** Surface meshing only. Volume meshing, boundary conditions and solver
integration are out of scope for this stage.

**No LLM anywhere.** Geometry, meshing, metrics and acceptance are all
deterministic. `pipeline/policy.LLMSelector` reserves a seam for a future agent and
is intentionally unimplemented — an agent would choose *among* validated candidate
actions, never decide whether a mesh is acceptable.

---

## Quick start

Everything runs through one script:

```bash
chmod +x run.sh
./run.sh setup            # .venv + dependencies (uses uv when installed)
source .venv/bin/activate
./run.sh all              # generate parts -> audit -> mesh -> report
./run.sh view block_hole  # open the result in Gmsh
```

`setup` uses [uv](https://github.com/astral-sh/uv) if it is on your PATH and
falls back to `venv` + `pip` otherwise. Either way it produces an ordinary
`.venv`, which `run.sh` activates for you on every invocation — no manual
`source` needed. `PYTHON=3.12 ./run.sh setup` pins the interpreter, and
`./run.sh sync` makes the venv match `requirements.txt` exactly.

`./run.sh help` lists every command. The most useful ones:

| command | what it does |
|---|---|
| `./run.sh test` | 300 tests, no Gmsh required, about a second |
| `./run.sh doctor` | check python, uv, Gmsh, packages, models |
| `./run.sh stage0 <cad>` | geometry audit only — fast triage, no meshing |
| `./run.sh mesh <cad>` | full adaptive pipeline, writes `.msh` + `.stl` + manifest |
| `./run.sh corpus <dir> -j 6` | many files: resumable, parallel, per-file timeout |
| `./run.sh log [part]` | what a run did: iterations, actions, rationale (`--failed`, `--raw`) |
| `./run.sh report` | aggregate an existing run |
| `./run.sh compare` | the three research meshers vs Gmsh |

Options include `-b/--budget`, `-i/--iterations`, `-t/--timeout`, `-j/--jobs`,
`--fidelity`, `--heal`, and `-n/--dry-run` to print the commands instead of
running them. If no files are given, commands default to `models/`.

Prefer the modules directly? Every command is a thin wrapper:

```bash
python -m tools.run_pipeline models/*.step --out out/meshes --json out/run.json
```

---

## Repository layout

```
stage0/      CAD geometry audit — deterministic measurement, no search
quality/     mesh evaluation — the arbiter; imports no gmsh
backends/    meshing backends — gmsh production path (black box)
pipeline/    entity IDs, size field, policy, loop, contracts, export
research/    three from-scratch meshers — reference only, not production
tools/       CLI drivers: verification, calibration, corpus runner
docs/        architecture notes and research takeaways
tests/       300 tests, all runnable without gmsh
```

| package | lines | role |
|---|---|---|
| [`pipeline/`](pipeline) | 2,791 | the intelligence layer — identity, sizing, policy, loop, contracts |
| [`tools/`](tools) | 2,318 | CLI drivers and verification harnesses |
| [`research/`](research) | 2,227 | Parametric CDT, Parametric AF, Direct 3D AF — reference only |
| [`stage0/`](stage0) | 1,375 | geometry audit |
| [`quality/`](quality) | 1,166 | backend-independent mesh evaluation |
| [`backends/`](backends) | 1,066 | gmsh backend, CAD surface adapter |
| [`tests/`](tests) | 5,011 | 300 tests |

---

## The pipeline

```
  STEP / IGES / BREP
        │
        ▼
  IMPORT + HEAL ──▶ persistent CAD entity IDs (survive healing and imprinting)
        ▼
  ANALYZE ────────▶ curvature, wall thickness, topology + feature contracts
        ▼
  SIZE FIELD ─────▶ declarative typed rules, scoped to entity IDs
        ▼
  MESH (gmsh) ────▶ surface mesh + per-triangle CAD face provenance
        ▼
  EVALUATE ───────▶ validity gate + objective scores + contract check
        │
        ├── ready? ────▶ EXPORT  .msh + .stl + manifest + run record
        ▼
  ADAPT ──────────▶ one action from a closed set, with recorded evidence
        └──────────▶ back to SIZE FIELD or MESH
```

### Where to start reading

| if you want to understand… | read |
|---|---|
| what "a good mesh" means here | [`quality/criteria.py`](quality/criteria.py) — the validity gate vs objective score split |
| how the loop decides what to change | [`pipeline/policy.py`](pipeline/policy.py) — the closed action set |
| how CAD identity survives healing | [`pipeline/entities.py`](pipeline/entities.py) — fingerprints + provenance graph |
| how sizing is expressed | [`pipeline/sizefield.py`](pipeline/sizefield.py) — declarative rules |
| how the mesh is checked against the CAD | [`pipeline/contracts.py`](pipeline/contracts.py) — topology + feature contracts |
| the from-scratch meshers | [`research/`](research) and [`docs/research_takeaways.md`](docs/research_takeaways.md) |

---

## Key design decisions

**The validity gate is separate from the objective score.** Hard failures
(inverted elements, non-manifold topology) make a mesh unusable — no score redeems
them and no refinement fixes them, so the response is a *strategy change*.
Objectives (shape, size, gradation, fidelity) are continuous and drive iteration.
`Verdict.score` is `0.0` when invalid, so a search can never trade validity for a
better average.

**Thresholds are fractions of model scale, never absolute.** Validated across a
**3,700× scale range** in the ABC corpus with no per-part configuration.

**`quality/` imports no Gmsh.** One evaluator scores the production backend and the
research meshers alike, so a comparison between them measures algorithms rather
than normalization differences.

**CAD entity IDs are persistent.** OCC tags change across healing, imprinting and
re-import. `pipeline/entities.py` fingerprints scale-normalized geometric
invariants and maintains a provenance graph (SAME / SPLIT / MERGE / CREATED /
DELETED), so evidence gathered in iteration 3 still refers to the same face in
iteration 7.

**Healing is opt-in and must be approved.** Forcing repair on import destroys valid
geometry. And when healing *does* remove a face, that removal is a contract
violation until approved — automatically if the face was shaped like a sliver,
explicitly otherwise.

---

## Input formats

| format | extensions | status |
|---|---|---|
| STEP | `.step` `.stp` | primary path; everything is tested on it |
| IGES | `.iges` `.igs` | accepted, untested |
| BREP | `.brep` | accepted, untested |

Mesh formats (STL, OBJ, PLY) and native CAD (`.sldprt`, `.x_t`, `.sat`) are
rejected with a named error before OpenCASCADE is invoked. Mesh formats are refused
on principle, not for lack of a reader: entity identity, per-face sizing and the
contracts are all keyed to CAD faces, which a triangle soup does not have.

---

## Tools

| command | purpose |
|---|---|
| `python -m tools.run_pipeline <cad>` | full adaptive pipeline with export |
| `python -m tools.corpus <dir> -j 6` | corpus runner: resumable, parallel, per-file timeout, aggregated |
| `python -m stage0.run <cad>` | geometry audit only |
| `python -m tools.verify_backend <cad>` | single-shot mesh, no adaptation — calibration numbers |
| `python -m tools.verify_entity_ids <cad>` | entity ID stability across heal / imprint / re-import |
| `python -m tools.verify_gmsh_api <cad>` | known-answer checks on Gmsh API assumptions |
| `python -m tools.compare_meshers --gmsh` | three research meshers vs Gmsh on a shared boundary |
| `python -m tools.calibrate collect out/ --csv calibration.csv` | threshold calibration workflow |
| `python -m tools.make_test_parts models/` | generate the synthetic corpus |

---

## Results at a glance

**Synthetic corpus (6 parts):** all six resolved. Five accepted at score 1.000;
`sliver_block` accepted after the loop enabled healing, which removed two 0.005 mm
artifact faces and took the mesh from 10,386 triangles to 1,858.

**ABC dataset corpus (46 runs):**

| outcome | count |
|---|---|
| accepted | 25 |
| valid but below target (`score_plateau`) | 13 |
| over element budget | 6 |
| stopped for a human decision | 2 |
| **silent failures** | **0** |

Every run either passed defined criteria or stopped and said why. Corpus spans a
3,700× scale range and 1–894 faces per part.

The open problem is **element shape**: all 13 plateau cases miss `min_angle`,
`min_shape` or `max_angle`, and no sizing rule fixes a triangle sitting on a 100:1
CAD face. See [RESEARCH.md §10.3](RESEARCH.md#103-element-shape-is-the-wall).

---

## Requirements

Python 3.12, `gmsh`, `numpy`, `trimesh`, `rtree`, `scipy`; `embreex` optional (~50×
faster ray casting). `pip install gmsh` ships arm64 wheels with OpenCASCADE built
in — no Homebrew OCC needed.
