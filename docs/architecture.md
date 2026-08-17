# Architecture

## Objective

Take difficult CAD and automatically produce the best simulation-ready mesh possible,
using mature meshing tools where appropriate.

Not: develop novel meshing algorithms. Not: replace Gmsh.

The value is the **intelligence around the mesher** — deciding what a given piece of
geometry needs, detecting when a mesh is not simulation-ready and *why*, and iterating
on meshing parameters until defined criteria are met.

## Layout

```
CADtoMesh_Agent/
├── stage0/          geometry audit -- deterministic, no LLM
├── quality/         backend-independent mesh evaluation  <-- the arbiter
├── backends/        meshing backends; gmsh is production
├── research/        the three from-scratch meshers -- REFERENCE ONLY
├── tools/           CLI drivers, calibration, comparison
├── docs/            research_takeaways.md, architecture.md
└── tests/
```

## The three layers

**`stage0/` — geometry audit.** Runs before any mesher. Topology and tolerance audit,
3D→2D parametric reduction diagnostics, reference boundary discretization, wall
thickness guard. Deterministic and reproducible: no LLM, no agent, no search. Its
output is the JSON that everything downstream reasons about.

**`quality/` — the arbiter.** One evaluator scores every mesh, regardless of origin.
Four metric families plus a verdict layer:

| family | module | what it answers |
|---|---|---|
| shape | `core` | are the elements well-formed? |
| topology | `core` | is it a valid surface? |
| fidelity | `core.chordal_deviation`, `normals` | does it represent the geometry? |
| size | `size` | did it honour the size field, and is it graded? |
| verdict | `criteria` | is it *usable*, and how close to *ready*? |

**Nothing in `quality/` imports gmsh.** Anything needing surface geometry takes an
object satisfying the `Surface` protocol. This is load-bearing: if the production path
were scored by gmsh's own metrics and the research meshers by ours, normalization
differences would be indistinguishable from real quality differences, and the
comparison that justified adopting gmsh could not be re-run as gmsh gets tuned.

**`backends/` — meshers.** Gmsh is the initial production backend, used as a black box:
build geometry, set a size field, generate, read the mesh. Its algorithms are not
inspected or reproduced.

**`research/` — reference implementations.** Parametric CDT, Parametric AF, Direct 3D
AF, plus their shared kernel (`predicates`, `metric`, `boundary`, `surface`,
`af_core`). **Not the production path.** Kept because:

- they are the only meshers whose internals we can instrument freely
- they produced the calibration behind `quality/` (see `docs/research_takeaways.md`)
- `tools/compare_meshers.py` keeps gmsh honest, and needs peers to compare against

Their tests and the comparison driver are maintained. They are not extended toward
production robustness.

## The validity gate

The single most important distinction in `quality/criteria.py`:

**Hard validity failures** make a mesh unusable — inverted elements, non-manifold
edges, inconsistent orientation, degenerate shape below an absolute floor, unmerged
duplicate nodes. No score redeems them; no further refinement fixes them. The response
is to reject and change strategy.

**Optimization objectives** are continuous — shape distribution, size adequation,
gradation, chordal deviation, normal deviation. These are what an iterative loop pushes
on.

`Verdict.is_valid` is a boolean gate that ignores scores entirely. `Verdict.score` is
0.0 when invalid, so a search can never trade validity for a better average.
`Verdict.ready` — valid **and** all objectives satisfied — is the loop's stopping
condition. `Verdict.unsatisfied()` returns objectives worst-attainment-first: the work
list.

## The adaptive loop (next)

```
    stage0 audit
        |
        v
    propose size field  <------------------+
        |                                  |
        v                                  |
    gmsh backend: mesh                     | adjust
        |                                  |
        v                                  |
    quality.assess -> Verdict              |
        |                                  |
        +-- invalid?  --> change strategy --+
        |                                  |
        +-- not ready? --> worst objective-+
        |
        v
    ready: emit mesh + verdict
```

Two properties to preserve:

1. **The loop reads `Verdict`, not raw metrics.** Hard failures branch to strategy
   change; unsatisfied objectives branch to parameter adjustment. Those are different
   actions and must not be reached by the same code path.
2. **Every iteration is recorded** — parameters in, verdict out. The loop is only as
   trustworthy as its scoring function, and the record is what lets thresholds be
   calibrated against real parts rather than guessed (same argument as
   `tools/calibrate.py` for Stage 0).

Not yet built. The quality framework was finished first deliberately: an adaptive loop
wired to an unvalidated scoring function cannot be debugged, because a bad decision and
a bad metric look identical from the outside.

## Practices carried forward

- **Build the known-answer case before the feature.** Analytic surfaces with
  closed-form answers caught nearly every bug in the research phase.
- **Prefer metrics with closed-form references.** `sagitta()` validates chordal
  deviation exactly; `facet_angle_bound()` does the same for normal deviation.
- **Assert conventions at startup.** Orientation errors do not raise exceptions; they
  produce plausible, wrong output.
- **Thresholds are calibrated, not guessed**, and recorded in the report that used
  them.
