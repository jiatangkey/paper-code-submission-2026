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
