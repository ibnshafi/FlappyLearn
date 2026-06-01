# Release Process

FlappyLearn uses Semantic Versioning.

## Versioning Strategy

- `MAJOR`: incompatible public API, checkpoint schema, CLI, or config changes.
- `MINOR`: backward-compatible features, new visualizations, new docs sections, new training modes.
- `PATCH`: bug fixes, docs corrections, CI updates, and security fixes that do not change public behavior.

Initial public version recommendation: `0.1.0`.

Before `1.0.0`, minor versions may refine APIs, but maintainers should document migration notes for any user-visible changes.

## Release Checklist

1. Confirm `main` is green in CI.
2. Run `ruff check .`, `ruff format --check .`, and `pytest` locally.
3. Run a smoke training pass with `python -m flappylearn train --config configs/smoke.json`.
4. Update `CHANGELOG.md` from `Unreleased` to the target version and date.
5. Confirm `src/flappylearn/__init__.py` and `pyproject.toml` versions match.
6. Build the package with `python -m build`.
7. Validate artifacts with `python -m twine check dist/*`.
8. Create a signed tag named `vX.Y.Z`.
9. Publish the GitHub release.
10. Attach notable replay and metrics screenshots to the release notes.

## Publishing Workflow

The included release workflow builds distributions when a version tag is pushed:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

Package publishing should use PyPI trusted publishing after the repository is connected to a PyPI project. Until then, keep the workflow in build-and-verify mode and upload manually only from clean release artifacts.

## Release Notes Guidance

Release notes should lead with the user-visible story: what learners can now see, run, configure, or trust. Include upgrade notes when a command, config key, checkpoint schema, or generated artifact changes. End with verification commands so users know the release was tested.
