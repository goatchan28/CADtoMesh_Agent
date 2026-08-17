# Research Takeaways: Three Surface Meshers From Scratch

We implemented Parametric Constrained Delaunay Triangulation, Parametric Advancing
Front, and Direct 3D Advancing Front from first principles, all sharing one boundary
discretization and one quality evaluator. This records what that bought us.

**The implementations are not the deliverable.** The deliverable is calibrated
judgement about what makes a surface mesh good, where meshing goes wrong, and which
of those failures are worth automating around. Gmsh is now the production backend.
These three are preserved under `research/` as reference implementations, with their
tests and `tools/compare_meshers.py` kept working, because they remain the only
meshers whose internals we can instrument freely.

---

## 1. The measured comparison

Identical boundary (verified by fingerprint), identical target size, identical
quality evaluator. `h` is the target element size; `ideal` is
`area / (√3/4 · h²)`.

| case | method | tris (ideal) | shape min | shape mean | min angle | chord/h | state |
|---|---|---|---|---|---|---|---|
| plane, h=0.15 | CDT | 140 (103) | 0.628 | 0.901 | 30.3° | 0 | closed |
| | param AF | 106 | **0.692** | 0.952 | 28.3° | 0 | closed |
| | direct 3D AF | 102 | 0.477 | 0.948 | 19.5° | 0 | closed |
| L-shape, h=0.3 | CDT | 100 (77) | **0.689** | 0.906 | **27.4°** | 0 | closed |
| | param AF | 82 | 0.063 | 0.925 | 3.1° | 0 | closed |
| | direct 3D AF | 82 | 0.063 | 0.922 | 3.1° | 0 | closed |
| cylinder R=10, h=1.0 | CDT | 172 (115) | 0.677 | 0.916 | **31.0°** | 0.0195 | closed |
| | param AF | 116 | 0.630 | **0.956** | 25.9° | 0.0204 | closed |
| | direct 3D AF | 116 | **0.681** | 0.946 | 28.2° | **0.0173** | closed |
| sphere band, h=1.5 | CDT | 212 (149) | **0.630** | 0.888 | **26.8°** | 0.0377 | closed |
| | param AF | 156 | 0.516 | **0.962** | 21.1° | 0.0454 | closed |
| | direct 3D AF | 158 | 0.290 | 0.948 | 13.4° | 0.0432 | closed |
| **sphere near pole**, h=1.5 | CDT | 128 (78) | 0.535 | 0.886 | 19.3° | 0.0337 | closed |
| | param AF | 100 | 0.136 | 0.862 | 4.5° | 0.4412 | **STALL(6)** |
| | direct 3D AF | **82** | **0.602** | **0.930** | **21.9°** | 0.0435 | closed |

### What it says

**Advancing front wins on element count and mean quality.** Both AF methods land at
1.0–1.1× the ideal count; CDT consistently overshoots by 1.3–1.6× because refinement
bisects rather than places. AF's mean shape quality is 0.92–0.96 against CDT's
0.89–0.92.

**Delaunay refinement wins on worst-case robustness.** CDT never stalled, on any
case, and its `shape_min` never fell below 0.53. That is the difference between a
method with a termination guarantee and one whose closure is heuristic.

**Direct 3D AF wins where the parametrization degrades.** At the sphere pole it beat
both parametric methods on every column while using the fewest elements. This is the
one place the three methods genuinely diverge rather than trade.

**Both AF methods share a failure at reflex corners.** The L-shape notch produced a
3.1° minimum angle in both, against CDT's 27.4°. They share it because they share the
front-management core — which is itself a finding: front *closure* is where the real
engineering effort in advancing front lives, not front *advance*.

---

## 2. Metric normalization is the highest-leverage decision in a parametric mesher

Parametric methods work in (u,v), which is not real space. A step of 0.1 in u may be
1.2 mm here and 40 mm there, and may not be perpendicular to a step in v.

The fix is the first fundamental form `I = [[E,F],[F,G]]`. Cholesky-factor it,
`Lt = Lᵀ`, and work in `ξ = Lt·(uv − uv₀)`, where `|d|_M = |Lt·d|₂`. Euclidean
geometry in ξ *is* metric geometry in (u,v).

Measured on a cylinder of radius 10 (metric `diag(100, 1)`, anisotropy 10):

| | raw (u,v) | metric-normalized |
|---|---|---|
| shape min | 0.087 | **0.654** |
| shape mean | 0.354 | **0.866** |
| min angle | 3.1° | **28.4°** |
| triangles (ideal 29) | 134 | **42** |
| area error | 0.96% | **0.09%** |

Note it produced *fewer* elements as well as better ones. In ξ the domain becomes the
unrolled 10×5 rectangle, where ordinary Delaunay is correct.

The linearization is **exact** for constant-metric surfaces (planes, cylinders, cones)
and **first-order** where the metric varies. That residual error is exactly what shows
up at the sphere pole, and it is a structural limit rather than an implementation
defect.

---

## 3. Orientation conventions are the most dangerous class of bug

Two separate instances, both of which produced output that looked entirely plausible.

**Front update.** Advancing edge `(i,j)` with apex `k` must add the **reverses** of the
triangle's CCW traversal — `(k,j)` and `(i,k)`, not `(j,k)` and `(k,i)`. Front edges
carry unmeshed material on their left; the triangle is on the left of its own
traversal, so the remaining material is on the other side. With the wrong sign, new
edges duplicated active ones instead of cancelling, the front never shrank, and a unit
square emitted **200,000 triangles over 14 nodes**.

**Surface normal.** The 3D front's inward direction is `p = n × e`, and `n` must be
`unit(∂X/∂u × ∂X/∂v)` — *not* whatever the surface reports as its outward normal.
Material-on-the-left was established by loop orientation **in (u,v)**, so `n` must
agree with that. On our sphere the two are exactly opposite; the front marched off the
patch and wrapped the sphere: **538 triangles against an ideal of 149, 240% area
error**. The plane and cylinder happened to agree, which is why only the sphere
exposed it.

**The lesson that generalizes:** orientation errors do not raise exceptions. Assert
the convention at startup against geometry whose answer you know. `af_direct3d` now
tests real boundary edges and fails with a named error if `n × e` points outward.

---

## 4. Refinement operator choice matters more than refinement criterion

We started with circumcenter insertion (textbook Ruppert/Chew) and it failed three
ways:

- circumcenters of obtuse boundary triangles fall **outside** the domain, so the loop
  either aborted (silently under-refining the whole face) or blacklisted the triangle
  (stalling with 9-unit edges against a 2.6 limit)
- encroachment splitting **cascaded**: a 14-node boundary became 53 nodes and produced
  ~8× the intended triangle count
- each pass needed a full domain re-classification, making the loop cubic

**Longest-edge bisection fixed all three.** It drives the stopping criterion directly
(bisecting the offending edge halves the quantity being tested, so progress is
guaranteed and termination follows), edge midpoints of interior triangles are always
inside the domain, and when the longest edge *is* a boundary segment the split is
exactly what the encroachment rule existed to trigger — so no encroachment rule is
needed. Delaunay quality is preserved because Bowyer-Watson re-triangulates after each
insertion.

---

## 5. Area is a bad fidelity proxy — the Schwarz lantern

We wrote a test asserting mesh area converges to true area *from below*: flat
triangles inscribed in a curve should under-measure. It failed. Area converged
**from above**: 50.83 → 50.15 → 50.06 against an exact 50.

This is the Schwarz lantern. Area convergence for an inscribed triangulation requires
the triangle **normals** to converge, not just the vertices to lie on the surface.
Inscribed area can exceed the true area, and can be made to converge to the wrong
value entirely.

Two consequences that shaped the quality framework:

1. **Chordal deviation** (distance) is a real fidelity metric; area is not. It has a
   closed-form reference — `sagitta(R, c) = R − √(R² − (c/2)²)` — so it can be
   validated exactly rather than eyeballed. Our implementation reproduces it to a
   ratio of 0.99 at fine sizes.
2. **Normal deviation** (angle) is the metric that actually governs the failure, and
   it scales differently: normal error is O(c/R) while chordal error is O(c²/R).
   Halving element size *halves* normal deviation but *quarters* chordal deviation.
   **Normal deviation is therefore the binding constraint at coarse sizes and the
   slower one to improve** — an adaptive loop watching only chordal error will stop
   too early.

---

## 6. Shape quality and geometric fidelity are independent

A sphere has **constant curvature**, so chordal error tracks element size wherever you
are on it. The pole wrecks the parametrization, not the geometry.

| | shape min | min angle | chordal (at equal maxL) |
|---|---|---|---|
| sphere mid-band | 0.630 | 26.8° | 0.0566 |
| sphere near pole | 0.535 | 19.3° | 0.0506 |

Shape degrades; fidelity does not. A framework reporting only shape would call the
pole mesh bad for the wrong reason, and one reporting only fidelity would call it
fine. Both are needed, and they are not substitutes.

The same argument applies to **size**: shape quality is scale-invariant, so a mesh of
flawless equilateral triangles at 10× the requested size scores 0.95. Our parametric
AF hit `maxL/target = 4.77` near the pole while reporting `shape_mean` 0.862. And to
**gradation**, which no per-element metric can see at all because it is a property of
pairs.

That reasoning produced the four-family structure now in `quality/`: shape, topology,
fidelity (chordal + normal), size (adequation + gradation).

---

## 7. Hard validity failures are categorically different from low quality

An inverted element gives a negative Jacobian: the element matrix is meaningless and
the solve diverges or returns confident nonsense. A non-manifold edge means the
topology is not a surface. Coincident-but-distinct nodes are a crack that looks
perfect and transmits no load.

None of these are "low quality". No score redeems them and no further refinement fixes
them. The correct response is to reject the mesh and change strategy.

Conflating the two breaks an adaptive loop in both directions: treat everything as a
soft score and it converges happily on a mesh with three inverted elements because the
average looks good; treat everything as a hard gate and it never terminates, because
"minimum angle above 30° everywhere" is not achievable on real CAD.

`quality/criteria.py` enforces the split — `Verdict.is_valid` is a boolean gate that
ignores scores entirely, and `Verdict.score` is 0.0 when invalid so a search can never
trade validity for a better average.

---

## 8. Exact predicates are not optional

Delaunay is decided by the **sign** of two determinants. A single wrong sign does not
give a slightly-wrong mesh — it gives an inverted triangle, a non-terminating cavity
walk, or an infinite flip loop. We used a floating-point filter with a
`fractions.Fraction` fallback (every Python float is exactly a binary rational) and
verified the fallback actually fires at the noise floor via `np.nextafter`.

A related trap: `_sign()` written as `(x>0)-(x<0)` raises `TypeError` on `np.bool_`.
Coordinates arrive as `np.float64` from any `tuple(numpy_row)`, so that is the *normal*
path, not an edge case.

---

## 9. O(n²) traps are easy to write and easy to miss

Two, both of which made the test suite time out rather than merely run slowly:

- **Point location** by brute-force scan → O(n²) insertion. Fixed with a directed walk
  from a cached hint, falling back to brute force if the walk cycles.
- **Domain classification** recomputed per inserted point, at O(triangles × boundary)
  winding tests each time. Fixed by pass-based batch refinement: classify once per
  pass, bisect every offending edge, then re-classify. Sweeps drop from O(inserts) to
  O(log(initial/target)).

A 2480-triangle face went from minutes to **2.21 s**.

---

## 10. The abstraction that made all of this testable

Defining a `Surface` protocol — `point`, `derivatives`, `normal`, `project`,
`is_inside` — with **analytic implementations** (plane, cylinder, sphere) meant every
algorithm could be validated against closed-form answers with no gmsh in the loop.

The three surfaces were chosen to be discriminating rather than convenient:

- **plane** — identity metric, so any anisotropy in the output is the algorithm's fault
- **cylinder R=10** — uniformly anisotropic, catches metric-blindness
- **sphere** — pole degeneracy, where `det I → 0`

Nearly every bug in this document was caught by one of them. That is the practice
worth carrying into the production work: **build the known-answer case before the
feature**, and prefer a metric with a closed-form reference over one you can only
eyeball.

---

## 11. Why this justifies gmsh in production

Everything above was measured on the **easy** cases: single faces, analytic surfaces,
no seams, no assemblies, uniform target size, a few thousand elements.

Production CAD is not that. It brings trimmed NURBS with degenerate parametrizations,
seams (which both parametric methods must **refuse** outright — a (u,v) loop across a
seam is ambiguous), multi-face conformity so adjacent faces share nodes exactly,
curvature- and proximity-driven size fields, sliver faces below tolerance, and
robustness across thousands of faces without a single stall. Our parametric AF stalled
on a *sphere patch*.

Gmsh has decades of hardening against exactly that. Reimplementing it would be a
multi-year project to arrive at a worse version of something free.

**Where the value actually is:** the intelligence *around* the mesher. Deciding what
size field a given piece of geometry needs, detecting when a mesh is not
simulation-ready and why, choosing which parameter to change in response, and
iterating until defined criteria are met. That work needs a rigorous, backend-
independent definition of "good mesh" and a calibrated sense of how meshers fail —
which is what building these three produced.

---

## Reproducing

```bash
python -m tools.compare_meshers --gmsh          # all cases, three methods + gmsh
python -m tools.research_demo --case all        # STL + (u,v) SVG per case
python -m tools.compare_meshers --case cylinder --target 1.0 --no-linearize
```

That last command reproduces the metric-normalization result in §2 — the single
clearest demonstration in the project.

Tests: `tests/test_research_*.py` (kernel, CDT, AF), `tests/test_quality_framework.py`.
