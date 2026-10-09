# CAD-to-Mesh: Automated Mesh Generation for Engineering Simulations

Turning a 3D CAD model into a mesh suitable for engineering simulations can be a complicated, time-consuming process. Engineers often need to manually adjust meshing parameters, inspect the results, and repeat the process until the mesh meets their requirements.

**This project automates that process.**

Given a CAD model, the system analyzes its geometry, generates a surface mesh using Gmsh, evaluates the mesh's quality, and automatically adjusts its approach until the mesh meets predefined simulation-readiness criteria.

Instead of requiring an engineer to manually troubleshoot every failed mesh, the system attempts to resolve issues on its own and explains when human intervention is necessary.

## How It Works

```text
       3D CAD Model
            |
            v
     Analyze Geometry
            |
            v
    Generate Surface Mesh
            |
            v
     Evaluate Mesh Quality
            |
       +----+----+
       |         |
      Pass      Fail
       |         |
       v         v
    Export    Adjust Meshing
     Mesh       Parameters
                 |
                 +----> Retry
```

The system follows five main steps:

1. **Import CAD:** Accepts 3D CAD files in STEP format and checks the geometry for potential issues.
2. **Analyze Geometry:** Examines features such as curvature, wall thickness, and topology to determine appropriate meshing parameters.
3. **Generate Mesh:** Uses Gmsh to convert the CAD geometry into a triangular surface mesh.
4. **Evaluate and Improve:** Checks mesh quality, identifies problems, and automatically adjusts meshing parameters to improve the result.
5. **Export Results:** Saves the validated mesh as `.msh` and `.stl` files, along with a report explaining the results.

If the system cannot produce an acceptable mesh, it stops and explains why rather than silently returning an unusable result.

---

## Project Results

The system has been evaluated on both synthetic CAD models and models from the ABC dataset.

| Metric | Result |
|---|---|
| Synthetic CAD models | 6/6 successfully resolved |
| ABC dataset evaluations | 46 |
| Meshes meeting all acceptance criteria | 25 |
| Valid meshes below quality targets | 13 |
| Meshes exceeding element budget | 6 |
| Cases requiring human intervention | 2 |
| Silent failures | 0 |

The evaluation covered CAD models ranging from 1 to 894 faces and spanning a 3,700× difference in geometric scale.

**Current limitation:** Some geometries produce poorly shaped triangles that cannot be fixed by adjusting mesh sizing alone. Improving element shape on difficult CAD surfaces remains an open research problem.

For detailed experiments, evaluation results, and failure analysis, see [RESEARCH.md](RESEARCH.md).

---

## Research Direction

This project is part of my undergraduate research at Columbia University.

The broader research goal is to develop an **AI agent capable of automating the process of converting raw 3D geometry into simulation-ready models**.

The current implementation focuses on building a reliable, deterministic meshing pipeline. It uses Gmsh for mesh generation and rule-based logic to evaluate results and adjust parameters.

**LLM integration is planned but not yet implemented.** Future work will explore using an AI agent to select meshing strategies and troubleshoot failures while preserving deterministic quality checks.

---

## Getting Started

### Installation

Clone the repository and install the required dependencies:

```bash
chmod +x run.sh
./run.sh setup
```

The setup script creates a Python virtual environment and installs the necessary packages.

### Run the Full Pipeline

```bash
./run.sh all
```

This generates test CAD models, analyzes their geometry, creates meshes, and produces a report.

### Mesh Your Own CAD Model

```bash
./run.sh mesh path/to/model.step
```

The system will analyze the CAD model, generate a surface mesh, evaluate its quality, and attempt improvements when necessary.

### View a Generated Mesh

```bash
./run.sh view block_hole
```

Opens the generated mesh in Gmsh for visualization.

### Other Commands

| Command | Description |
|---|---|
| `./run.sh test` | Run the 300 automated tests |
| `./run.sh doctor` | Check dependencies and environment |
| `./run.sh stage0 <cad>` | Analyze CAD geometry without generating a mesh |
| `./run.sh mesh <cad>` | Run the complete adaptive meshing pipeline |
| `./run.sh corpus <dir> -j 6` | Process multiple CAD models in parallel |
| `./run.sh log` | Review previous runs and meshing decisions |
| `./run.sh report` | Generate an evaluation report |
| `./run.sh compare` | Compare experimental meshing algorithms with Gmsh |

Run `./run.sh help` for the complete list of available commands.

---

## Technical Architecture

The system is organized into several modules:

| Module | Purpose |
|---|---|
| `stage0/` | Analyzes CAD geometry and identifies potential issues |
| `quality/` | Evaluates mesh validity, quality, and geometric accuracy |
| `backends/` | Interfaces with Gmsh to generate meshes |
| `pipeline/` | Manages meshing parameters, iteration, and decision-making |
| `research/` | Contains three experimental meshing algorithms developed during the research |
| `tools/` | Provides command-line utilities and evaluation tools |
| `tests/` | Contains 300 automated tests |

### Design Principles

**1. Use established meshing tools.**

Rather than reinventing mesh generation, the production pipeline uses Gmsh. The research focuses on automating the decisions surrounding mesh generation.

**2. Never accept an invalid mesh.**

Every generated mesh must pass explicit validity checks. A high quality score cannot compensate for fundamental geometric problems.

**3. Preserve the original geometry.**

The system tracks CAD surfaces throughout the meshing process to ensure important geometric features are not accidentally lost or modified.

**4. Make decisions traceable.**

Every iteration records what the system changed, why it made that change, and whether the resulting mesh improved.

**5. Know when to stop.**

If the system cannot satisfy its acceptance criteria, it reports the problem rather than continuing indefinitely or returning a misleading result.

---

## Supported Formats

| Format | Extensions | Status |
|---|---|---|
| STEP | `.step`, `.stp` | Fully supported and tested |
| IGES | `.iges`, `.igs` | Accepted, not extensively tested |
| BREP | `.brep` | Accepted, not extensively tested |

The pipeline currently generates **surface meshes only**. Volume meshing, boundary conditions, and integration with simulation solvers are outside the scope of this stage.

---

## Documentation

For more detailed technical information:

- [RESEARCH.md](RESEARCH.md) — Research methodology, experiments, results, and limitations.
- [Architecture Notes](docs/) — System architecture and implementation details.
- [Research Takeaways](docs/research_takeaways.md) — Findings from developing and comparing different meshing algorithms.

---

## Technologies

**Language:** Python 3.12

**Meshing & Geometry:** Gmsh, OpenCASCADE, Trimesh

**Scientific Computing:** NumPy, SciPy

**Testing:** Pytest

**Research Areas:** Computational Geometry, Finite Element Meshing, Automated Mesh Optimization, AI Agents
