# CADtoMesh_Agent — Research and Engineering Report

**An automated CAD → simulation-ready surface mesh pipeline, and what building it taught us.**

This document accompanies the repository. It records the methodology, the
experiments actually run, the results, and the limitations. It is written to be
read without running the code.

*For a top-level tour of the repository, see [README.md](README.md).*

---

## Contents

1. [Objective and scope](#1-objective-and-scope)
2. [Methodology](#2-methodology)
3. [Phase I — Three surface meshers built from scratch](#3-phase-i--three-surface-meshers-built-from-scratch)
4. [Why we pivoted to Gmsh](#4-why-we-pivoted-to-gmsh)
5. [Architecture](#5-architecture)
6. [Stage 0 — Geometry audit](#6-stage-0--geometry-audit)
7. [Stage 1 — Adaptive surface meshing](#7-stage-1--adaptive-surface-meshing)
8. [The quality framework](#8-the-quality-framework)
9. [Experiments and results](#9-experiments-and-results)
10. [Failure analysis](#10-failure-analysis)
11. [Limitations](#11-limitations)
12. [Future work](#12-future-work)
13. [Reproducing the experiments](#13-reproducing-the-experiments)

---

## 1. Objective and scope

**Goal.** Take difficult CAD and automatically produce the best simulation-ready
surface mesh possible, using mature meshing tools where appropriate.

**Explicitly not the goal.** Developing novel meshing algorithms, or replacing
Gmsh.

**Scope.** Surface meshing only. Volume meshing, tetrahedralization, boundary
conditions and solver integration are deliberately out of scope for this stage.

**Determinism.** No LLM is present anywhere in the pipeline. Geometry operations,
meshing, numerical metrics and final acceptance are all deterministic. The
architecture reserves a seam for an agent (`pipeline/policy.LLMSelector`), but it
is unimplemented by design: an agent would choose *among* validated candidate
actions and would never decide whether a mesh is acceptable.

**Target analysis.** Linear-triangle FEM on predominantly chunky (not thin-walled)
mechanical solids.

---

## 2. Methodology

Three practices shaped the work, and each of them caught defects that the others
would have missed.

**Build the known-answer case before the feature.** Every algorithm was validated
against analytic surfaces — plane, cylinder, sphere — chosen to be discriminating
rather than convenient. A plane has an identity metric, so any anisotropy in the
output is the algorithm's fault. A cylinder at *R*=10 is uniformly anisotropic and
catches metric-blindness. A sphere has pole degeneracies where *det I* → 0. Nearly
every bug in this report was caught by one of the three.

**Prefer metrics with a closed-form reference.** Chordal deviation is validated
against the exact sagitta *R* − √(*R*² − (*c*/2)²); normal deviation against the
exact facet angle asin(*c*/2*R*). A metric you can only eyeball cannot be trusted
to gate an automated decision.

**Separate the measurement from the search.** Stage 0 contains no agent, no
search, and no adaptive behaviour. It is a reproducible measurement, and keeping it
so is what makes the adaptive loop debuggable — a bad decision and a bad
measurement look identical from the outside otherwise.

The test suite is **300 tests, all runnable without Gmsh**, achieved by defining a
`Surface` protocol with analytic implementations and injecting the meshing backend
through a protocol rather than importing it.

---

## 3. Phase I — Three surface meshers built from scratch

Before adopting a production mesher we implemented three surface meshing
algorithms from first principles, sharing one boundary discretization and one
quality evaluator:

1. **Parametric Constrained Delaunay Triangulation** — Bowyer-Watson incremental
   Delaunay in (*u,v*), Anglada edge-flip constraint recovery, winding-number
   domain classification, metric-driven longest-edge refinement.
2. **Parametric Advancing Front** — front advanced in metric-normalized parametric
   coordinates.
3. **Direct 3D Advancing Front** — front advanced on the surface in ℝ³, validated
   in a tangent plane at each front edge.

**These implementations are not the deliverable.** They are preserved under
`research/` as reference implementations with their tests and comparison driver
maintained. The deliverable was calibrated judgement about what makes a surface
mesh good and how meshing fails.

### 3.1 Fairness protocol

All three receive one immutable `FaceBoundary`. Each mesh records the SHA
fingerprint of the boundary it was built from, and the comparison driver **asserts
all three match** before reporting anything. Without that check, a future change to
one mesher could quietly re-derive or re-space the boundary, and the resulting
difference would be indistinguishable from an algorithmic one.

They also share: the same target size, the same `apex_height()` function (which
sets how fast a front marches, so differing values would make element counts
incomparable), the same configuration, the same metric linearization, and the same
quality evaluator.

### 3.2 Results

`ideal` = area / (√3/4 · *h*²). `chord/h` is chordal deviation over element size.

| case | method | tris (ideal) | shape min | shape mean | min angle | chord/h | state |
|---|---|---|---|---|---|---|---|
| plane, h=0.15 | CDT | 140 (103) | 0.628 | 0.901 | 30.3° | 0 | closed |
| | Parametric AF | 106 | **0.692** | 0.952 | 28.3° | 0 | closed |
| | Direct 3D AF | 102 | 0.477 | 0.948 | 19.5° | 0 | closed |
| L-shape, h=0.3 | CDT | 100 (77) | **0.689** | 0.906 | **27.4°** | 0 | closed |
| | Parametric AF | 82 | 0.063 | 0.925 | 3.1° | 0 | closed |
| | Direct 3D AF | 82 | 0.063 | 0.922 | 3.1° | 0 | closed |
| cylinder R=10, h=1.0 | CDT | 172 (115) | 0.677 | 0.916 | **31.0°** | 0.0195 | closed |
| | Parametric AF | 116 | 0.630 | **0.956** | 25.9° | 0.0204 | closed |
| | Direct 3D AF | 116 | **0.681** | 0.946 | 28.2° | **0.0173** | closed |
| sphere band, h=1.5 | CDT | 212 (149) | **0.630** | 0.888 | **26.8°** | 0.0377 | closed |
| | Parametric AF | 156 | 0.516 | **0.962** | 21.1° | 0.0454 | closed |
| | Direct 3D AF | 158 | 0.290 | 0.948 | 13.4° | 0.0432 | closed |
| **sphere near pole**, h=1.5 | CDT | 128 (78) | 0.535 | 0.886 | 19.3° | 0.0337 | closed |
| | Parametric AF | 100 | 0.136 | 0.862 | 4.5° | 0.4412 | **STALL(6)** |
| | Direct 3D AF | **82** | **0.602** | **0.930** | **21.9°** | 0.0435 | closed |

### 3.3 What the comparison established

**Advancing front wins on element count and mean quality.** Both AF methods land
at 1.0–1.1× the ideal count; CDT overshoots by 1.3–1.6% because refinement
*bisects* rather than *places*. AF mean shape quality is 0.92–0.96 against CDT's
0.89–0.92.

**Delaunay refinement wins on worst-case robustness.** CDT never stalled on any
case, and its `shape_min` never fell below 0.53. That is the difference between a
method with a termination guarantee and one whose closure is heuristic.

**Direct 3D AF wins where the parametrization degrades.** At the sphere pole it
beat both parametric methods on every column while using the fewest elements. This
is the one case where the three genuinely diverge rather than trade.

**Both AF methods fail at reflex corners.** The L-shape notch produced a 3.1°
minimum angle in both, against CDT's 27.4°. They share the failure because they
share the front-management core — which is itself a finding: front *closure* is
where the engineering effort in advancing front lives, not front *advance*.

### 3.4 Metric normalization — the highest-leverage decision

Parametric methods work in (*u,v*), which is not real space. A step of 0.1 in *u*
may be 1.2 mm in one place and 40 mm in another, and may not be perpendicular to a
step in *v*.

The fix is the first fundamental form **I** = [[E,F],[F,G]]. Cholesky-factor it and
work in **ξ** = **Lᵀ**(uv − uv₀), where |*d*|_M = |**Lᵀ***d*|₂. Euclidean geometry
in ξ *is* metric geometry in (*u,v*).

Measured on a cylinder of radius 10 (metric diag(100, 1), anisotropy 10):

| | raw (*u,v*) | metric-normalized |
|---|---|---|
| shape min | 0.087 | **0.654** |
| shape mean | 0.354 | **0.866** |
| min angle | 3.1° | **28.4°** |
| triangles (ideal 29) | 134 | **42** |
| area error | 0.96% | **0.09%** |

It produced *fewer* elements as well as better ones: in ξ the domain becomes the
unrolled 10×5 rectangle, where ordinary Delaunay is correct. The linearization is
**exact** for constant-metric surfaces (planes, cylinders, cones) and
**first-order** where the metric varies — and that residual error is exactly what
shows up at the sphere pole.

### 3.5 Orientation conventions are the most dangerous bug class

Two separate instances, both producing entirely plausible output.

**Front update.** Advancing edge (*i,j*) with apex *k* must add the **reverses** of
the triangle's CCW traversal — (*k,j*) and (*i,k*), not (*j,k*) and (*k,i*). Front
edges carry unmeshed material on their left; the triangle is on the left of its own
traversal, so remaining material is on the other side. With the wrong sign, new
edges duplicated active ones instead of cancelling, the front never shrank, and a
unit square emitted **200,000 triangles over 14 nodes**.

**Surface normal.** The 3D front's inward direction is *p* = *n* × *e*, and *n*
must be unit(∂X/∂u × ∂X/∂v) — *not* whatever the surface reports as outward.
Material-on-the-left was established by loop orientation **in (u,v)**. On our
sphere the two are exactly opposite; the front marched off the patch and wrapped
the sphere: **538 triangles against an ideal of 149, 240% area error.** The plane
and cylinder happened to agree, which is why only the sphere exposed it.

**Orientation errors do not raise exceptions.** `af_direct3d` now asserts the
convention at startup against real boundary edges and fails with a named error.

### 3.6 Refinement operator choice matters more than refinement criterion

Circumcenter insertion (textbook Ruppert/Chew) failed three ways: circumcenters of
obtuse boundary triangles fall *outside* the domain; encroachment splitting
*cascaded* (a 14-node boundary became 53 nodes, ~8× the intended element count);
and each pass needed a full domain re-classification, making the loop cubic.

**Longest-edge bisection fixed all three.** It drives the stopping criterion
directly, so progress is guaranteed and termination follows; edge midpoints of
interior triangles are always inside the domain; and when the longest edge *is* a
boundary segment, the split is exactly what the encroachment rule existed to
trigger — so no encroachment rule is needed.

### 3.7 Area is a bad fidelity proxy — the Schwarz lantern

A test asserting that mesh area converges to true area *from below* failed. Area
converged **from above**: 50.83 → 50.15 → 50.06 against an exact 50.

This is the Schwarz lantern. Area convergence for an inscribed triangulation
requires the triangle **normals** to converge, not merely the vertices to lie on
the surface. Inscribed area can exceed the true area and can be made to converge to
the wrong value entirely.

Two consequences shaped the quality framework:

**Chordal deviation** (distance) is a real fidelity metric with a closed-form
reference. Our implementation reproduces the exact sagitta to a ratio of 0.99 at
fine sizes:

| target | max edge | measured chordal | exact sagitta | ratio |
|---|---|---|---|---|
| 2.0 | 2.500 | 0.049958 | 0.078433 | 0.637 |
| 1.0 | 1.287 | 0.019525 | 0.020723 | 0.942 |
| 0.5 | 0.644 | 0.005130 | 0.005189 | **0.988** |

(The ratio is below 1 at coarse sizes because the longest edge is only partly
circumferential; an axial edge has zero curvature.)

**Normal deviation** (angle) is the metric that actually governs the failure, and
it scales differently: normal error is O(*c*/*R*) while chordal error is
O(*c*²/*R*). Halving element size *halves* normal deviation but *quarters* chordal
deviation. **Normal deviation is therefore the binding constraint at coarse sizes** —
an adaptive loop watching only chordal error stops too early.

### 3.8 Shape quality and geometric fidelity are independent

A sphere has constant curvature, so chordal error tracks element size wherever you
are. The pole wrecks the parametrization, not the geometry.

| | shape min | min angle | chordal (at equal max edge) |
|---|---|---|---|
| sphere mid-band | 0.630 | 26.8° | 0.0566 |
| sphere near pole | 0.535 | 19.3° | 0.0506 |

Shape degrades; fidelity does not. A framework reporting only shape would call the
pole mesh bad for the wrong reason; one reporting only fidelity would call it fine.
The same argument applies to **size** (shape quality is scale-invariant, so a mesh
of flawless triangles at 10× the requested size scores 0.95) and to **gradation**
(no per-element metric can see it, because it is a property of pairs).

That reasoning produced the four-family structure now in `quality/`.

### 3.9 Other findings from Phase I

**Exact predicates are not optional.** Delaunay is decided by the *sign* of two
determinants. A wrong sign gives an inverted triangle, a non-terminating cavity
walk, or an infinite flip loop — not a slightly-wrong mesh. We used a
floating-point filter with a `fractions.Fraction` fallback and verified the
fallback fires at the noise floor via `np.nextafter`.

**O(n²) traps.** Brute-force point location and per-insertion domain
classification each made the suite time out. A directed walk from a cached hint and
pass-based batch refinement took a 2,480-triangle face from minutes to **2.21 s**.

---

## 4. Why we pivoted to Gmsh

Everything in Phase I was measured on the **easy** cases: single faces, analytic
surfaces, no seams, no assemblies, uniform target size, a few thousand elements.

Production CAD is not that. It brings trimmed NURBS with degenerate
parametrizations, **seams** (which both parametric methods must *refuse* outright —
a (u,v) loop across a seam is ambiguous), multi-face conformity, curvature- and
proximity-driven size fields, sliver faces below tolerance, and robustness across
thousands of faces without a stall. Our parametric AF stalled on a *sphere patch*.

Gmsh has decades of hardening against exactly that. Reimplementing it would be a
multi-year project to arrive at a worse version of something free.

**Where the value is:** the intelligence *around* the mesher — deciding what size
field a given piece of geometry needs, detecting when a mesh is not
simulation-ready and why, choosing which parameter to change, and iterating until
defined criteria are met. That work needs a rigorous, backend-independent
definition of "good mesh" and a calibrated sense of how meshers fail, which is what
Phase I produced.

Gmsh is used strictly as a **black box**: build geometry, set a size field, call
`generate(2)`, read the mesh. Its algorithms are neither inspected nor reproduced.

---

## 5. Architecture

```
stage0/     CAD geometry audit          deterministic measurement, no search
quality/    mesh evaluation             the arbiter; imports no gmsh
backends/   meshing backends            gmsh production path (black box)
pipeline/   entity IDs, size field,     the intelligence layer
            policy, loop, contracts,
            export
research/   three from-scratch meshers  reference only, not production
tools/      CLI drivers                 verification, calibration, corpus runner
```

| package | lines | files | role |
|---|---|---|---|
| `pipeline` | 2,791 | 9 | entity IDs, size field, policy, loop, contracts, export |
| `tools` | 2,318 | 12 | CLI drivers, verification, calibration, corpus runner |
| `research` | 2,227 | 9 | three from-scratch meshers, reference only |
| `stage0` | 1,375 | 10 | CAD geometry audit |
| `quality` | 1,166 | 5 | backend-independent evaluation |
| `backends` | 1,066 | 4 | gmsh production backend, CAD surface adapter |
| `tests` | 5,011 | 14 | 300 tests, all runnable without gmsh |

### 5.1 Data flow

```
  STEP / IGES / BREP
        │
        ▼
  ┌──────────────┐
  │ IMPORT+HEAL  │──▶ load report, EntityRegistry seeded (persistent PIDs)
  └──────────────┘
        ▼
  ┌──────────────┐
  │ ANALYZE      │──▶ curvature, wall thickness, face adjacency
  │              │──▶ TopologyContract + FeatureContract derived from CAD
  └──────────────┘
        ▼
  ┌──────────────┐
  │ SIZE FIELD   │──▶ SizeFieldSpec: typed rules scoped to PIDs
  └──────────────┘
        ▼
  ┌──────────────┐
  │ MESH (gmsh)  │──▶ SurfaceMesh + MeshProvenance (triangle → face PID)
  └──────────────┘
        ▼
  ┌──────────────┐
  │ EVALUATE     │──▶ Verdict (validity gate + objective scores)
  │              │──▶ ContractReport, per-face attribution
  └──────────────┘
        ├── ready? ─────────▶ EXPORT .msh + .stl + manifest + run record
        ▼
  ┌──────────────┐
  │ ADAPT        │──▶ one Action from a closed set, with evidence
  └──────────────┘
        └──────────▶ back to SIZE FIELD or MESH
```

### 5.2 Persistent CAD entity identity

OCC/Gmsh entity tags are **not stable** — they change across `healShapes()`,
`fragment()`, re-import, and tolerance changes. That makes them useless as report
keys: a finding saying "face 47 has poor chordal error" is meaningless after the
next re-import.

`pipeline/entities.py` assigns a **PID** = SHA of scale-normalized geometric
invariants (mass, centroid, bbox, topological arity, optional curvature and
outward-normal signatures). Two mechanisms are needed, not one:

- **Fingerprint** handles the common case where an entity is unchanged.
- **Provenance matching** handles the rest, because identity is genuinely *not
  bijective*: `fragment()` merges coincident faces and splits overlapping ones, and
  a hash cannot express "this face became those three."

The registry maintains a provenance graph with relations SAME / SPLIT / MERGE /
CREATED / DELETED, so a finding recorded against a parent can be translated onto
its children.

**Measured stability** on the synthetic corpus (`tools/verify_entity_ids.py`):
re-import, a 10× tolerance change, and healing all preserved **100%** of surviving
PIDs. Imprinting `two_blocks` correctly recorded 10 SAME + 1 MERGE with both
parents traceable to the merged interface face.

One honest limit: **geometry alone cannot distinguish coincident faces before
imprinting.** Two solids meeting at a shared face produce faces with identical
mass, centroid and bbox. The outward-normal signature separates them when the
backend supplies it; otherwise the matcher defers to geometric correspondence
rather than guessing. A false SAME is the failure mode we optimized against — it
silently attaches evidence to the wrong face.

### 5.3 The size field is declarative data

`SizeFieldSpec` is a list of typed rules (`base`, `curvature`, `thickness`,
`feature`, `clamp`, `gradation`), each scoped to PIDs rather than tags. It buys
three things: a future agent edits a validated structure instead of emitting mesher
code; successive specs can be **diffed**, so a run record shows exactly what
changed; and a spec **hashes**, which is how the loop detects oscillation.

Curvature sizing takes the **minimum** of two bounds:

```
chordal:  h ≤ √(8 R δ)
normal:   h ≤ 2 R sin(θ)
```

Both are needed because they scale differently — the normal bound binds at coarse
sizes, so a field driven by sagitta alone systematically under-refines.

### 5.4 Compiling the size field: constant fields cannot ramp

The first implementation applied one `Constant` field per face. That is wrong, and
the failure is instructive. A constant cannot ramp *inside* a face, so elements of
two different sizes meet at every shared edge and **the measured gradation equals
the raw face-size ratio** — 2.44 on `block_hole` against a 1.5 limit. Per-face
smoothing only trades that against element count, one for one.

The fix compiles each fine face to a `Constant` pin plus a `Distance` + `Threshold`
ramp extending outward from its boundary, over a coarse background, combined with
`Min`. The ramp length has a closed form: elements growing geometrically from *h*
by ratio *g* cover *h*(*gⁿ*−1)/(*g*−1), so

```
ramp_distance = (h_far − h_near) / (g − 1)
```

An intermediate version measured distance from the face *boundary* rather than the
face, which made the middle of a 100 mm face ramp to the coarse size — producing a
mesh **4.5× coarser than requested** (`in_band` 0.134). Pinning the face with a
`Constant` and ramping only outward fixed it.

---

## 6. Stage 0 — Geometry audit

Stage 0 runs before any mesher and produces a structured, serializable report. It
is deterministic by design: no LLM, no agent, no search.

| sub-stage | what it establishes |
|---|---|
| topology / tolerance audit | entity counts, model scale, assembly imprinting, sub-tolerance entities, shell closure |
| 3D→2D parametric reduction | per-face parametrization quality; classifies each face for parametric meshing |
| reference 1D discretization | the shared skeleton the surface mesh conforms to |
| wall thickness | local thickness distribution via inward ray casting |

### 6.1 Design decisions that mattered

**Thresholds are fractions of model scale, never absolute.** A 5 mm bracket and a
5 m weldment should not need separate configuration. The ABC corpus spans a
**3,700× scale range** (3.695 to 13,790) and required no per-part configuration —
the strongest available validation of this decision.

**Healing is opt-in.** An early version forced `OCCFixDegenerated`,
`OCCFixSmallEdges`, `OCCFixSmallFaces`, `OCCSewFaces` and `healShapes()` on every
import. That *destroyed valid geometry*: a sphere's poles are degenerate edges by
construction, and stripping them means the shell can no longer validate as closed;
two solids sharing a coincident face get sewn into shells. Both then reported
`topology.no_volume` — a BLOCK caused entirely by the repair. **Healing is a
remedial action, not an import setting.**

**Circularity, not area, identifies slivers.** `face.sliver` originally tested
area. The 50 × 0.005 rail faces on `sliver_block` have area 0.25 mm² — 4.2×10⁻⁵ of
scale², nowhere near "tiny" — and were missed entirely. They are **thin, not
small**, and no area threshold separates the two cases. The isoperimetric quotient
4π*A*/*P*² reads 3.1×10⁻⁴ for the rail against 0.46 for a legitimate face with a
hole.

| face | area | area/scale² | circularity | aspect |
|---|---|---|---|---|
| 50 × 0.005 rail | 0.25 | 4.2e-05 | **3.1e-04** | 10000:1 |
| 30 × 1.5 fillet strip | 45.0 | 7.6e-03 | 1.4e-01 | 20:1 |
| 0.001 × 0.001 patch | 1e-06 | 1.7e-10 | 7.9e-01 | 1:1 |

**Seam curves are not free edges.** A periodic face's seam appears *twice* in its
own boundary loop, but both times with the *same* face. Counting distinct faces
reported every seam as a one-sided free edge — a spurious BLOCK on any part with a
through hole. Counting incidences *with multiplicity* distinguishes them: 1
incidence on 1 face is a genuine free edge; 2 incidences on 1 face is a seam.

---

## 7. Stage 1 — Adaptive surface meshing

### 7.1 The validity gate versus the objective score

The central distinction in `quality/criteria.py`, and the one that makes an
adaptive loop possible at all.

**Hard validity failures** make a mesh unusable — inverted elements (negative
Jacobian, so the element matrix is meaningless), non-manifold edges (the topology
is not a surface), inconsistent orientation, degenerate shape below an absolute
floor, unmerged duplicate nodes. No score redeems them and no refinement fixes
them. The correct response is to **reject and change strategy**.

**Optimization objectives** are continuous — shape distribution, size adequation,
gradation, chordal deviation, normal deviation. These are what an iterative loop
pushes on.

Conflating the two breaks the loop in both directions: treat everything as a soft
score and it converges happily on a mesh with three inverted elements because the
average looks good; treat everything as a hard gate and it never terminates,
because "minimum angle above 30° everywhere" is not achievable on real CAD.

`Verdict.is_valid` is a boolean gate that ignores scores. `Verdict.score` is **0.0
when invalid**, so a search can never trade validity for a better average.
`Verdict.ready` — valid **and** all objectives satisfied — is the stopping
condition.

### 7.2 The adaptation policy

`pipeline/policy.py` maps evidence onto a **closed set** of actions. Hard failures
and unsatisfied objectives are reached by *separate functions*, so the type system
makes it impossible to propose "refine those faces" against an inverted element.

| trigger | response class | examples |
|---|---|---|
| hard failure | strategy change | enable healing, raise heal tolerance, re-imprint, switch algorithm, escalate |
| unsatisfied objective | size field edit | tighten curvature bound, tighten gradation, refine worst faces, raise floor |
| contract violation | strategy change or escalate | re-imprint, escalate for a human decision |

Three constraints bound the search: **localized before global** (a global change is
only proposed after a localized one failed to move the objective), an **element
budget** any refinement must declare, and **no repeats** (a parameter set already
meshed is rejected).

**Recovery ladders** are explicit and terminate. For example, healing is a
*tolerance*, not a switch:

```
heal=off  →  enable healing (1e-4 of scale)
          →  raise to 1e-3
          →  raise to 1e-2
          →  escalate (above ~1% of scale, healing removes real features)
```

### 7.3 Termination

| outcome | meaning |
|---|---|
| `accepted` | valid **and** every objective satisfied |
| `escalated` | the policy asked for a human decision (e.g. defeaturing) |
| `no_action_available` | no untried candidate remains |
| `oscillation` | a parameter set already meshed came back around |
| `score_plateau` | the score stopped moving |
| `budget_exhausted` | the size field implies more elements than allowed |
| `max_iterations` | attempt budget exhausted |

Whatever the outcome, the loop returns the **best valid iteration**, which is often
not the last — the final iteration is usually the most aggressive edit.

### 7.4 Contracts — comparing the mesh against the CAD

Two questions no quality metric can answer, because both compare against the input.

**Topology contract, derived not configured.** "Require watertight" as a flag is
wrong in both directions: a closed solid that comes out open is a defect, and a
sheet body that comes out *closed* is also a defect.

| CAD | expectation |
|---|---|
| 1 volume | closed skin, zero boundary |
| 0 volumes, faces present | boundary length matches the CAD's own free edges |
| 2+ volumes | closed per body, interfaces meshed and conformal |

Sheets are checked on **boundary length, not edge count** — the count depends on
element size, the length does not. That is what catches a hole in the middle of a
sheet, which "don't require watertightness" would miss entirely.

**Feature contract — removals need approval.** Healing deletes geometry. On
`sliver_block` that was exactly right: two rail faces vanished, element count fell
10,386 → 1,858, and the mesh became valid. But nothing distinguished that from
healing quietly removing a *real* small fillet — the output would look identical
and the accepted mesh would be of geometry that is not the input geometry.

A removal is therefore a **violation until approved**: automatically when the
removed face is *shaped* like a sliver (circularity ≤ 1e-2, ~300:1 aspect),
explicitly otherwise. Note this uses the same shape-versus-size lesson from §6.1 —
an area-based rule refused the rails and demanded a human decision about an obvious
artifact.

### 7.5 Export

Four artifacts per part: `.msh`, `.stl`, `.manifest.json`, `.run.json`. The `.msh`
is Gmsh 2.2 ASCII with **one physical group per CAD face**, so face identity
survives into volume meshing and boundary conditions can be applied per face rather
than by picking triangles. Tags are assigned in sorted PID order so runs are
diffable, and the manifest maps tag → PID because a `.msh` carries only integers.

---

## 8. The quality framework

Backend-independent by construction: nothing in `quality/` imports Gmsh, and
anything needing surface geometry takes a `Surface` protocol object. The same
evaluator scores the production backend and the three research meshers — otherwise
normalization differences would be indistinguishable from real quality differences.

| family | metrics | answers |
|---|---|---|
| **shape** | `shape_quality` = 4√3A/Σl², radius ratio, angles, aspect ratio | are the elements well-formed? |
| **topology** | watertight, manifold, orientation, boundary length | is it a valid surface? |
| **fidelity** | chordal deviation (distance), normal deviation (angle) | does it represent the geometry? |
| **size** | size adequation vs the request, gradation between neighbours | did it honour the size field? |
| **verdict** | validity gate + weighted objective score | is it usable, and how close to ready? |

Two implementation notes worth recording. **Size adequation is symmetric** —
over-refinement counts as a miss, because our own CDT produced 1.3–1.6× the ideal
element count with excellent shape scores and a one-sided metric would have called
that perfect. **Normal deviation is measured unsigned**, to the nearer of ±*n*;
global orientation is `topology_checks`' job, and conflating them would report
~180° of "deviation" for every facet on a surface whose analytic normal opposes
unit(∂X/∂u × ∂X/∂v) — a case this project has already hit.

---

## 9. Experiments and results

### 9.1 Synthetic corpus

Six parts generated by `tools/make_test_parts.py`, each targeting one specific
finding, with properties known by construction.

| part | faces | tris | valid | score | shape min | min angle | in_band | grad | watertight |
|---|---|---|---|---|---|---|---|---|---|
| block_hole | 11 | 3,506 | ✓ | 1.000 | 0.776 | 31.6° | 0.995 | 1.36 | ✓ |
| cylinder | 3 | 1,812 | ✓ | 1.000 | 0.849 | 39.5° | 1.000 | 1.19 | ✓ |
| sliver_block | 8 | 10,386 | **✗** | 0.000 | 0.030 | 1.4° | 0.648 | 1.93 | ✓ |
| sphere | 1 | 1,120 | ✓ | 1.000 | 0.721 | 31.8° | 1.000 | 1.19 | ✓ |
| thin_plate | 6 | 189,844 | ✓ | 1.000 | 0.600 | 27.7° | 0.986 | 1.13 | ✓ |
| two_blocks | 11 | 1,650 | ✓ | 1.000 | 0.899 | 42.8° | 1.000 | 1.23 | ✓ |

**Calibration result:** `in_band` 0.986–1.000 across the corpus. Gmsh honours a
per-face size field essentially exactly, which was an open question — the
`min_size_in_band` threshold is not binding.

**Budget estimator:** actual/estimated ran 1.03–1.38 on healthy parts. The
estimator is a lower bound (it ignores ramp bands on neighbouring faces), so a
budget wants ~1.5× headroom.

**`sliver_block` demonstrates the full loop.** Iteration 0 produced 10,386
triangles and failed `validity.degenerate_shape`. The policy enabled healing;
iteration 1 removed the two rail faces and produced **1,858 triangles, valid,
score 1.000**. All that refinement had existed only to chase a 0.005 mm artifact.

### 9.2 ABC dataset corpus

49 CAD files pulled from one chunk of the ABC dataset (Koch et al., ~1M CAD models
from Onshape), plus the 8 synthetic parts and 3 Gmsh sample files present in the
same directory. **46 pipeline runs completed.**

**Corpus profile** (40 parts parsed by Stage 0):

| | min | median | max |
|---|---|---|---|
| model scale | 3.695 | 86.6 | **13,790** |
| face count | 1 | 30 | **894** |

The 3,700× scale range is the headline: scale-relative thresholds required **no
per-part configuration** across it.

**Stage 0 triage:** 38 clean, 7 error, 2 timeout, 2 block.

**Stage 0 findings across the corpus:**

| finding | count |
|---|---|
| `discretization.interface_edge` | 455 |
| `face.sliver` | 399 |
| `param.distorted` | 180 |
| `curve.degenerate` | 173 |
| `param.degenerate` | 166 |
| `discretization.seam_edge` | 133 |
| `discretization.empty_curve` | 32 |
| `discretization.free_edge` | 32 |
| `curve.tiny` | 26 |

**Final pipeline outcomes (46 runs):**

| outcome | count | share |
|---|---|---|
| accepted | 25 | 54.3% |
| score_plateau | 13 | 28.3% |
| budget_exhausted | 6 | 13.0% |
| no_action_available | 1 | 2.2% |
| escalated | 1 | 2.2% |

Best-mesh triangles: median 18,410, max 228,854. Runtime: median 4.2 s, max
296.6 s. Adapted (more than one iteration): 16 of 46. Meshes exported: 33.

**Framing matters here.** "54% success" is the wrong reading. The 13
`score_plateau` parts produce *valid* meshes that miss a shape objective — usable,
just not certified ready. The honest summary is: **25 accepted, 13 valid but below
target, 6 over budget, 2 stopped for a human decision, and 0 silent failures.**
Every run either passed defined criteria or stopped and said why.

### 9.3 Threshold calibration

`tools/calibrate.py` collected **780 threshold-bearing findings**, of which **111
are borderline** (measured within 2× of the threshold).

| threshold | findings |
|---|---|
| `sliver_circularity` | 399 |
| `distortion_ratio_warn` | 180 |
| `degenerate_curve_frac` | 173 |
| `tiny_curve_frac` | 26 |
| `tiny_face_area_frac` | 2 |

The borderline `sliver_circularity` rows cluster at ratio 0.51–0.59 — circularity
≈ 0.026–0.030 against a 0.05 threshold, i.e. faces around **100–120:1 aspect**.
Real mechanical CAD is full of those and most are legitimate ribs and flanges. At
~10 sliver findings per part, the threshold is likely too loose or the severity too
high. **This calibration is not yet complete** — the rows need labelling.

---

## 10. Failure analysis

Failures were the most productive part of the project. Four are worth recording.

### 10.1 Four silent early-returns in the adaptive loop

The corpus exposed the same bug class four times: **an early return in `_iterate`
that skipped assessment**, leaving an iteration with no failures *and* no
objectives. The policy then saw nothing to act on and the run reported
`no_action_available` — which describes the *policy*, not the part.

| # | condition | symptom | fix |
|---|---|---|---|
| 1 | empty mesh | 14 runs uninformative | assess anyway; `mesh.empty` failure already existed but was unreachable |
| 2 | over budget | mislabelled as no-action | distinct `budget_exhausted` outcome |
| 3 | mesher raised | 9 parts, zero recovery attempts | `mesher.failed` hard failure + two recovery ladders |
| 4 | thickness raised | 6 parts aborted mid-recovery | thickness is an *input*, not a precondition |

Outcome evolution across the four fixes:

| | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| accepted | 24 | 24 | 24 | **25** |
| score_plateau | 7 | 7 | 8 | 13 |
| budget_exhausted | 0 | 5 | 6 | 6 |
| no_action_available | **14** | 9 | 1 | **1** |
| backend_error | 0 | 0 | 6 | **0** |
| **adapted (iters > 1)** | 8 | 8 | 10 | **16** |

**Fix 4 is the most instructive.** `measure_thickness()` runs its own
`generate(2)`, so on a part the mesher cannot handle it raises *exactly the error
the policy just switched algorithm to avoid* — and it sat inside the reload
try-block, aborting the run before the new algorithm was ever tested. It also never
set `Mesh.Algorithm`, so it meshed with the default even after the switch. **The
recovery was defeating itself.**

**Structural lesson:** `_iterate` should have exactly one exit that always produces
an assessed verdict. Each early return looked locally reasonable; collectively they
meant a third of the corpus produced outcomes describing the pipeline rather than
the geometry.

### 10.2 Healing that did nothing

Enabling healing on `sliver_block` produced a **byte-identical mesh** — 10,386
triangles, same failure, same score. Gmsh's `healShapes()` defaults to an
**absolute** tolerance of 1e-8. On a 76 mm part that is 500,000× smaller than the
0.005 mm rail it was asked to remove.

This is the one place in the project that silently used an absolute default when
every other threshold is scale-relative. The failure mode is worth noting: enabling
healing *looked* like a legitimate remedy that simply did not work, and without the
reload report and an "ineffective action" flag, the natural conclusion would have
been "this geometry needs defeaturing" — a correct-sounding diagnosis reached for
entirely the wrong reason.

### 10.3 Element shape is the wall

Of the 13 `score_plateau` parts, the unsatisfied objectives are **`min_angle` (9),
`min_shape` (8), `max_angle` (7)** — element shape, in every case.

This is not a sizing problem. Refinement cannot fix triangle quality when the
underlying CAD face is a 100:1 sliver; the elements on that face are poor at any
size. The local→global refinement escalation exists and is not enough. **The
missing operation is defeaturing**, which changes the feature contract and is
therefore not currently automatic.

### 10.4 Scale and performance

The largest part in the corpus has **894 faces**; the largest synthetic had 11.
Runtime median is 4.2 s but the maximum is 296.6 s, and 2 parts timed out at 120 s
in Stage 0 alone — which does no meshing at all.

Two likely causes, neither measured directly: `parametric_map.analyze` samples a
12×12 grid per face with an `isInside` call per sample (894 faces ≈ 129,000 OCC
calls before any meshing), and the backend creates a `Constant` + `Distance` +
`Threshold` triple per fine face.

---

## 11. Limitations

**Thresholds are under-calibrated.** 780 findings collected, 111 borderline, none
labelled. `sliver_circularity` fires ~10× per part on real CAD.

**Element shape cannot currently be fixed** when the CAD face itself is a sliver.
Defeaturing is flagged but never applied.

**Performance is untested at scale.** 894 faces is the current ceiling and it
costs minutes.

**Only faces have persistent IDs.** No curve-level identity, so a feature contract
cannot say "preserve this fillet edge."

**Conformity is asserted, not verified.** We check that interface faces produced
elements, not that adjacent faces share nodes along shared curves. Gmsh does this
by construction; we do not confirm it.

**IGES is untested.** The format is accepted but only STEP has been exercised.

**No point-cloud path.** The original brief mentioned point clouds; the pivot to
CAD-derived evidence removed any route for them, since a reconstructed surface has
no CAD faces to attach PIDs or contracts to.

**Stage 0 and the pipeline overlap.** `stage0/thickness.py` and
`backend.measure_thickness()` do the same job twice, and Stage 0's audit report is
not consumed by the loop.

---

## 12. Future work

Ordered by expected value.

1. **Label the calibration set** and re-fit thresholds. The tooling exists; the
   data exists; the labels do not. `OVERLAPPING` output from `calibrate fit` would
   mean a metric needs a companion, not a new number.
2. **Defeaturing as an explicit operation**, with removals entering the feature
   contract as approved exceptions. This is the direct remedy for §10.3.
3. **Performance at scale** — profile the 894-face part, batch the `isInside`
   sampling, and consider grouping faces of similar size into shared fields.
4. **Single-exit refactor of `_iterate`**, so a fifth instance of the §10.1 bug
   class cannot occur.
5. **Curve-level entity IDs**, enabling edge-level feature contracts.
6. **Verify conformity** rather than asserting it.
7. **Volume meshing** — out of scope here, but the `.msh` export with per-face
   physical groups was designed as its handoff.

---

## 13. Reproducing the experiments

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 300 tests, none requiring gmsh
for t in tests/*.py; do python "$t" | tail -1; done

# Phase I: three-way mesher comparison + the metric ablation
python -m tools.make_test_parts models/
python -m tools.compare_meshers --gmsh
python -m tools.compare_meshers --case cylinder --target 1.0 --no-linearize

# Verify gmsh API assumptions against known-answer geometry
python -m tools.verify_gmsh_api models/cylinder.step --radius 12.0

# Entity ID stability across re-import, tolerance change, healing, imprinting
python -m tools.verify_entity_ids models/*.step

# Stage 0 audit
python -m stage0.run models/block_hole.step --json out/block_hole.stage0.json

# Single-shot backend calibration
python -m tools.verify_backend models/*.step --json out/backend.json

# Full adaptive pipeline with export
python -m tools.run_pipeline models/*.step --out out/meshes --json out/run.json

# Corpus: resumable, parallel, per-file timeout, aggregated
python -m tools.corpus models/abc -j 6
python -m tools.corpus models/abc --report

# Threshold calibration
python -m tools.calibrate collect out/abc --csv calibration.csv
python -m tools.calibrate fit calibration.csv
```

**Environment.** All figures in this report were produced on a Mac Studio (M4 Max,
48 GB), Python 3.12, Gmsh via `pip install gmsh` (arm64 wheels with OpenCASCADE
built in).

---

## References

- Koch et al., *ABC: A Big CAD Model Dataset for Geometric Deep Learning*, CVPR
  2019 — source of the evaluation corpus.
- Geuzaine & Remacle, *Gmsh: a three-dimensional finite element mesh generator*,
  IJNME 2009 — the production backend, used as a black box.
- Shewchuk, *What Is a Good Linear Finite Element?*, 2002 — why large angles matter
  more than small ones for interpolation error.
- Anglada, *An improved incremental algorithm for constructing restricted Delaunay
  triangulations*, 1997 — constraint recovery by edge flipping.
- Babuška & Aziz, *On the angle condition in the finite element method*, 1976.
