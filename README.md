# Stage 0 — CAD geometry audit

Deterministic geometry inspection for a CAD→FEM meshing agent. No LLM in this
stage, by design: Stage 0 is a *measurement*, so it must be reproducible. The
agent consumes its JSON output in Stage 1.

## Install (macOS / Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install gmsh numpy trimesh rtree
# Optional but strongly recommended — ~50x faster ray casting for thickness:
pip install embreex
```

`pip install gmsh` ships arm64 wheels with OpenCASCADE built in; no Homebrew OCC
needed. Without `embreex`, trimesh falls back to a pure-numpy ray intersector,
which works but is slow enough to matter on a few thousand samples.

## Layout

`stage0/` must be a direct child of your working directory, or `python -m
stage0.run` will not find it. Do **not** put an `__init__.py` at the project
root -- that makes the root a package and breaks `-m`.

```
CADtoMesh_Agent/
├── .gitignore
├── README.md
├── requirements.txt
├── .venv/                          (you create this; not in the archive)
├── stage0/                         Stage 0 package
│   ├── __init__.py
│   ├── config.py                   ALL thresholds; serialized into each report
│   ├── metrics.py                  pure differential geometry, no gmsh
│   ├── topology.py                 pure edge-incidence classification, no gmsh
│   ├── report.py                   dataclasses = the Stage 0 -> Stage 1 contract
│   ├── geometry_audit.py           0a  topology / tolerance / imprinting
│   ├── parametric_map.py           0b  3D -> 2D reduction, per-face classification
│   ├── boundary_discretization.py  0c  reference 1D mesh + conformity checks
│   ├── thickness.py                0d  inward-ray wall thickness guard
│   └── run.py                      driver / CLI entry point
├── tools/
│   ├── __init__.py
│   ├── make_test_parts.py          generates synthetic STEP with known answers
│   ├── verify_gmsh_api.py          known-answer checks on gmsh API assumptions
│   ├── diagnose_import.py          isolates STEP-export vs import-healing damage
│   └── calibrate.py                threshold calibration log
├── tests/
│   ├── test_metric.py              analytic tests, runs without gmsh
│   ├── test_topology.py            edge classification + seam regression
│   └── test_geometry_metrics.py    sliver metric + config loading
├── thresholds.toml                 threshold overrides (commit this)
├── calibration.csv                 labelled findings (commit this)
├── models/                         CAD input goes here (gitignored)
└── out/                            JSON reports (gitignored)
```

## Run, in this order

```bash
# 1. math only, no gmsh needed
python tests/test_metric.py
python tests/test_topology.py
python tests/test_geometry_metrics.py

# 2. generate synthetic parts with known-by-construction answers
python -m tools.make_test_parts models/

# 3. verify the gmsh API assumptions before trusting any output
python -m tools.verify_gmsh_api models/cylinder.step --radius 12.0
python -m tools.verify_gmsh_api models/block_hole.step

# 4. now run Stage 0 — on a known part first, then your own
python -m stage0.run models/block_hole.step --json out/block_hole.stage0.json
python -m stage0.run models/your_part.step  --json out/your_part.stage0.json
```

Any STEP (`.step`/`.stp`), IGES, or BREP file works. From
SolidWorks/Fusion/Onshape, export STEP AP214 or AP242.

Exit code 2 means a blocking finding — do not proceed to Stage 1.

### What the synthetic parts are for

| Part | Establishes |
|---|---|
| `cylinder.step` | `|Xu| == 12`, `|Xv| == 1` by construction — settles the `getDerivative` layout question exactly. Also periodic, so it exercises seam handling. |
| `sphere.step` | Two pole degeneracies; must classify `DEGENERATE` |
| `block_hole.step` | Chunky and well behaved; a trimmed face for the `isInside` check |
| `two_blocks.step` | Coincident faces; must report a shared interface after imprinting |
| `thin_plate.step` | 1.5 mm wall; must trip the thickness guard |
| `sliver_block.step` | Sub-tolerance faces; must trip `face.sliver` |

If `verify_gmsh_api` reports `WARN`, the test part lacked the needed feature —
try a different one. `FAIL` on the derivative-layout check means the `du`/`dv`
slicing in `parametric_map.analyze_face()` needs swapping, and it tells you so.

## Healing is opt-in

`import_cad()` does **not** repair geometry by default, and that default matters.

Forcing `OCCFixDegenerated`, `OCCFixSmallEdges`, `OCCFixSmallFaces`, `OCCSewFaces`
and `healShapes()` on every import destroys valid solids. A sphere's poles are
degenerate edges *by construction*; `FixDegenerated` strips them and the shell can
no longer validate as closed. Two solids sharing a coincident face get sewn into
shells, dropping both solid wrappers. Both then report `topology.no_volume` — a
BLOCK caused entirely by the repair, on geometry that was never broken.

Healing is a **remedial action**, not an import setting. Import raw, audit, and
heal only if the raw audit shows a real defect:

```bash
python -m stage0.run models/part.step            # raw (default)
python -m stage0.run models/part.step --heal     # opt in, and compare
python -m tools.diagnose_import models/part.step # isolate what damaged what
```

The report records `healed: true/false`, so you can always tell which mode
produced a finding.

## Threshold calibration

Don't keep a threshold log by hand. Every finding records what it tripped and by
how much (`threshold_name`, `threshold`, `measured`), and every report records the
full config it ran under — so the reports *are* the log.

```bash
# 1. run the corpus, always with --json
for f in models/*.step; do
  python -m stage0.run "$f" --json "out/$(basename "$f" .step).stage0.json"
done

# 2. aggregate into a labelling sheet
python -m tools.calibrate collect out/ --csv calibration.csv

# 3. fill the `verdict` column: real / false / blank

# 4. ask what your verdicts imply
python -m tools.calibrate fit calibration.csv

# 5. accepted values go in thresholds.toml
python -m stage0.run models/part.step --config thresholds.toml
```

Verdicts carry forward across re-collection, so re-running after a code change
does not lose labelling work. Commit `thresholds.toml` and `calibration.csv`
together — the pair records what you decided and the evidence behind it.

Rows are sorted **borderline first** — those whose `measured/threshold` ratio is
within 2x of 1.0, meaning the threshold rather than the geometry decided the
outcome. Label those first; a finding three orders of magnitude clear of its cut
tells you nothing about the cut. Stage 0's summary flags them inline too.

The `fit` step reports **OVERLAPPING** when no threshold on that metric separates
real from false findings. That is the most useful output it produces: it means the
metric is wrong rather than the number, and no amount of tuning will fix it.

## The four sub-stages

| Module | What it establishes |
|---|---|
| `geometry_audit.py` | Topology counts, model scale, assembly imprinting, sub-tolerance entities, shell closure |
| `parametric_map.py` | Per-face 3D→2D reduction quality; classifies each face for parametric-space meshing |
| `boundary_discretization.py` | Reference 1D mesh; the shared skeleton the surface mesh conforms to |
| `thickness.py` | Local wall thickness distribution; guard against under-resolved walls |

`metrics.py` holds the pure differential geometry, deliberately free of gmsh so
it can be tested against closed-form surfaces.

## Two design decisions worth understanding

**Thresholds are fractions of the bbox diagonal, never absolute lengths.** A 5 mm
bracket and a 5 m weldment should not need separate config files. `model_scale`
is computed once in `audit()` and everything keys off it.

**The 1D discretization here is a reference, not a deliverable.** Fixed uniform
sizing, curvature adaption explicitly off. Its job is to expose curves that
cannot carry a segment and to give the agent a baseline element count. Stage 1
throws it away and re-discretizes with agent-chosen sizing fields. Letting the
agent influence Stage 0 would turn a measurement into a search step.

## What is and is not verified

The differential geometry in `metrics.py` is unit-tested against closed-form
surfaces (8/8), and the edge classification in `topology.py` against hand-built
topologies (6/6). The gmsh API calls were written without a gmsh install available,
so `tools/verify_gmsh_api.py` exists to close that gap with known-answer checks
rather than inspection. Run it before trusting Stage 0 output.

The check that matters most is the `getDerivative` output layout. Stage 0 assumes
`[dX/du (3), dX/dv (3)]` per point. `analyze_face()` asserts the value *count*, so
a length change is loud — but a silent *reordering* of the two blocks would
corrupt every anisotropy and area-scale number without raising anything. The
cylinder test pins it: `|Xu|` must be 12.0 and `|Xv|` must be 1.0, and those are
distinguishable, so a swap is unambiguous. **Verified passing** on gmsh/macOS
arm64 for both `cylinder.step` (radius 12.0000) and `block_hole.step` (8.0000).

### Known-issue log

* **Seam curves misreported as free edges** (fixed). Counting *distinct faces*
  per curve classified every periodic face's seam as a one-sided free edge, so
  any part with a through hole produced a spurious BLOCK. Incidence counting with
  multiplicity fixes it: a seam gives 2 incidences on 1 face, a genuine free edge
  gives 1 on 1. Regression test in `tests/test_topology.py`.

* **Import healing destroyed valid solids** (fixed). Healing was forced on every
  import, which stripped a sphere's degenerate pole edges and sewed coincident
  faces of touching solids, producing false `topology.no_volume` BLOCKs. Healing
  is now opt-in via `--heal`; `tools/diagnose_import.py` isolates export-side from
  import-side damage.

* **`face.sliver` tested area instead of shape** (fixed). `sliver_block.step` has
  two 50 x 0.005 rail faces and produced zero sliver findings: area 0.25 is 4e-5
  of scale^2, well above the "tiny" cut. The faces are thin, not small, and *no*
  area threshold separates the two cases. Replaced with circularity
  `4*pi*A/P^2`, which reads 3e-4 for the rail against 0.46 for a legitimate face
  with a hole. `face.tiny` still covers genuinely small faces — different defect.

* **Assembly interface edges reported as non-manifold** (fixed). `two_blocks.step`
  produced four `nonmanifold_edge` WARNs. Each of those curves bounds 3 faces —
  the imprinted interface plus one face from each block — which is *expected*
  topology after `fragment()`, not a defect. Incidence counting alone cannot see
  the difference, so `enforce_conformal_assembly` now records the interface face
  list on the report and the 1D check cross-references it, emitting
  `interface_edge` (INFO). A genuine 3-incidence edge with no interface face
  involved still WARNs.

* **Thickness measured to internal walls** (fixed). After imprinting, a shared
  interface is a real face inside the material and landed in the surface mesh, so
  inward rays stopped there: two bonded 30 mm blocks reported 30 mm through a
  60 mm continuous region, and the interface edges made the triangulation
  non-manifold, tripping `not_watertight`. Ray casting now uses only the outer
  skin (faces bounding exactly one volume), which fixes both symptoms.

* **Borderline findings were invisible** (fixed). `thin_plate`'s slivers measured
  4.574e-2 against a 5e-2 cut — ratio 0.91. A slightly different default would
  have reported nothing at all, and nothing in the output said so. `Finding` now
  exposes `margin` and `borderline`, the summary marks them, and
  `tools/calibrate.py` sorts them first.

Thresholds in `geometry_audit.py` (`TINY_CURVE_FRAC`, `TINY_FACE_FRAC`) and
`parametric_map.py` (`DISTORTION_RATIO_WARN`, `ANISOTROPY_WARN`) are opening
guesses. Calibrate against a real corpus before trusting the WARN/BLOCK split.

## Next

`thickness.check_against_element_size(report, target_h)` is the guard, and it
cannot run in Stage 0 — it needs a proposed element size, which is a Stage 1
output. Wire it in as the first check after the agent proposes sizing.
