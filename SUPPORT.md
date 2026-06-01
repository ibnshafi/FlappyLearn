# Support

FlappyLearn is designed to be approachable for learners and contributors. Please choose the support path that matches what you need.

## Get Help

- Use GitHub Discussions for questions, experiment ideas, learning help, and community showcases.
- Use GitHub Issues for reproducible bugs, documentation problems, and concrete feature requests.
- Use pull request comments for review-specific questions.

## Before Opening An Issue

Run these checks first:

```bash
python --version
python -m pip install -e ".[dev]"
pytest
python -m flappylearn train --config configs/smoke.json
```

Include your Python version, operating system, command, config file, and the relevant error output.

## FAQ Routing

- "How does neuroevolution work?" Start with `docs/concepts/neuroevolution.md`.
- "How do I change training settings?" Read `docs/reference/configuration.md`.
- "The replay looks wrong." Open a bug report with the checkpoint, replay JSON, and browser.
- "I want to add a learning algorithm." Open a discussion first so the architecture stays coherent.
- "I found a security issue." Follow `SECURITY.md` instead of opening a public issue.

## Community Expectations

Be precise, kind, and patient. Many people will discover this project while learning machine learning for the first time, so explanations matter as much as code.
