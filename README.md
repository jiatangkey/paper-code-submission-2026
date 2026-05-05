# GraphNav

Benchmarking Spatial Cognitive Graph Reasoning in Vision-Language Models.

## Overview

GraphNav is a controlled benchmark for evaluating cognitive-graph reasoning in vision-language models (VLMs). Unlike prior benchmarks built on real-world scans or virtual 3D scenes that entangle topological reasoning with execution-level confounds, GraphNav isolates each confound through procedural 3D maze generation, enabling clean attribution of VLM failures.

## Key Design Principles

- **Vision-Action Alignment as an Explicit Variable** — Four annotation conditions (C1: unlabeled, C2: arrows, C3: L-F-R, C4: 1-2-3) decouple visual grounding from topological reasoning.
- **Stable Cross-View Identity** — Distinctive 3D landmark objects (127-model catalog) at every key node decouple visual place recognition from cognitive-graph reasoning.
- **Discrete Graph-Node Movement** — Movement restricted to predefined nodes with discrete front/back/left/right relations, eliminating continuous metric estimation.

## Tasks

| Task | Description |
|------|-------------|
| **Repeated Navigation** | Retrace a previously explored path from memory |
| **Reversed Navigation** | Retrace a path in the opposite direction of exploration |
| **Shortcut Discovery** | Find a shorter path than the one previously explored |

## Annotations

| Condition | Description |
|-----------|-------------|
| C1 | No labels (implicit spatial inference) |
| C2 | Arrow icons (← ↑ →) |
| C3 | Semantic letters (L F R) |
| C4 | Numeric indices (1 2 3) |

## Metrics

- **SR** — Success Rate
- **PFS** — Path Fidelity Score (path overlap for repeated/reversed navigation)
- **SPL** — Success weighted by Path Length (efficiency for shortcut discovery)
- **DPS** — Directional Progress Score (step-wise goal alignment)

## Configuration

Data directories and API settings are configured in the `toolKit_core.py` files within each module. Edit those files to customize paths and parameters.

## Project Structure

```
shortcut/      - Shortcut discovery experiments
forward/       - Repeated (forward) navigation experiments
backward/      - Reversed navigation experiments
matching/      - Direction-object matching experiments
```
