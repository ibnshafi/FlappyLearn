# FlappyLearn Docs

FlappyLearn is a neuroevolution playground where an AI learns Flappy Bird from scratch. The project is built for people who want to see learning happen, not just read final scores.

## What You Can Do

- Train a population of recurrent neural circuits.
- Watch learning curves improve over generations.
- Export browser-playable replays of the best discovered agent.
- Benchmark against random and no-op baselines.
- Study evolution concepts through a concrete game.

## Quick Start

```bash
python -m pip install -e ".[dev,docs]"
pytest
python -m flappylearn train --config configs/smoke.json
python -m flappylearn replay --checkpoint runs/smoke/latest/best.json --output runs/smoke/latest/best_replay.html
```

## Recommended Learning Path

1. Run the smoke config and open `runs/smoke/latest/best_replay.html`.
2. Open `runs/smoke/latest/metrics.html` and inspect the learning curve.
3. Read the neuroevolution and fitness function guides.
4. Change one config value and run again.
5. Compare two experiments with their metrics and benchmark JSON.

## Documentation System Choice

MkDocs Material is the best fit for FlappyLearn because it is Python-native, easy for beginners to run, fast to build in CI, and excellent for educational projects with code snippets and concept pages.

Docusaurus is stronger for large product sites with React-heavy customization. Mintlify is polished for hosted API docs. FlappyLearn benefits more from a lightweight docs stack contributors can run locally with one Python install.
