# Contributing To FlappyLearn

Thanks for helping make machine learning easier to see, understand, and modify. FlappyLearn values clear code, reproducible experiments, friendly documentation, and small pull requests that are easy to review.

## Setup

```bash
git clone https://github.com/ibnshafi/FlappyLearn.git
cd flappylearn
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
pre-commit install
pytest
```

On macOS or Linux, activate the virtual environment with `source .venv/bin/activate`.

## Development Workflow

1. Create a focused branch from `main`.
2. Make the smallest coherent change that solves the issue.
3. Add or update tests when behavior changes.
4. Run `ruff check .`, `ruff format .`, and `pytest`.
5. Update docs when user-facing behavior, configuration, or workflows change.
6. Open a pull request with a clear summary and verification notes.

## Branching Strategy

Use short descriptive names:

- `feature/network-visualizer`
- `fix/replay-frame-scaling`
- `docs/neuroevolution-guide`
- `chore/ci-cache`

Maintainers may use release branches named `release/vX.Y.Z` when preparing a public release.

## Pull Request Process

Every pull request should include:

- A concise explanation of the change.
- Screenshots or generated artifact links for visualization changes.
- Tests or a rationale when tests are not appropriate.
- Documentation updates for public behavior.
- A note about performance impact if training speed, memory, or generated artifact size changes.

Review is based on correctness, maintainability, educational clarity, reproducibility, and kindness toward future contributors.

## Commit Conventions

Use Conventional Commits:

- `feat: add network topology export`
- `fix: keep replay canvas dimensions stable`
- `docs: explain mutation operators`
- `test: cover benchmark output schema`
- `chore: update release workflow`

Breaking changes should include `!`, for example `feat!: rename trainer config key`.

## Coding Standards

- Prefer simple Python and explicit data flow.
- Keep the core package dependency-light.
- Preserve deterministic behavior for seeded evaluation.
- Use dataclasses for structured configuration.
- Keep generated artifacts human-readable JSON or dependency-free HTML when practical.
- Avoid hidden game-state access in policies or evaluation.
- Add succinct comments only where they clarify non-obvious algorithmic choices.

## Testing

Run the full test suite:

```bash
pytest
```

Run linting and formatting:

```bash
ruff check .
ruff format .
```

Run a smoke training pass:

```bash
python -m flappylearn train --config configs/smoke.json
```

## Documentation

Docs are built with MkDocs Material:

```bash
mkdocs serve
```

Write docs for learners first. Explain concepts plainly, then link to implementation files for readers who want to go deeper.
