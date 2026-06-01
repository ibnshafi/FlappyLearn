# First-Time Contributor Guide

Welcome. You do not need to be a machine-learning expert to contribute to FlappyLearn.

## Good First Contributions

- Improve a concept explanation.
- Add a small test for a config or CLI behavior.
- Make an error message clearer.
- Add a benchmark result to docs.
- Improve replay or metrics visual polish.
- Try a config change and document what happened.

## Local Setup

```bash
python -m pip install -e ".[dev,docs]"
pytest
python -m flappylearn train --config configs/smoke.json
```

## How To Pick Work

Choose issues labeled:

- `good first issue`
- `documentation`
- `tests`
- `visualization`
- `education`

If an issue is unclear, ask a short question in the issue. Maintainers should help narrow the next step.

## Pull Request Tips

- Keep your first PR small.
- Explain what changed and how you tested it.
- Add screenshots for visual changes.
- Do not worry about being perfect; maintainers can help shape the final patch.

## Learning The Code

Start with these files:

- `src/flappylearn/env.py` for the game simulator.
- `src/flappylearn/genome.py` for the evolved circuit.
- `src/flappylearn/learner.py` for the training loop.
- `src/flappylearn/visualize.py` for generated HTML artifacts.
