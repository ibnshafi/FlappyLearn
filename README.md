# FlappyLearn

> Watch an AI discover Flappy Bird from scratch with neuroevolution, live metrics, replay exports, and beginner-friendly learning material.

[![CI](https://github.com/ibnshafi/FlappyLearn/actions/workflows/ci.yml/badge.svg)](https://github.com/ibnshafi/FlappyLearn/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](pyproject.toml)
[![Project status](https://img.shields.io/badge/status-public%20launch%20ready-16a34a.svg)](LAUNCH_AUDIT.md)

FlappyLearn is a complete autonomous learning system for Flappy Bird. It evolves compact recurrent neural circuits, visualizes training progress, benchmarks learned agents against baselines, and exports browser-playable replays with no heavyweight game engine.

**Banner recommendation:** place a 1280 x 640 social banner at `docs/assets/flappylearn-social-card.png` showing a learning curve behind a Flappy Bird replay frame. Put the logo in the top-left, the one-line value proposition centered, and a small "Neuroevolution playground" label in the bottom-right.

**Animated demo recommendation:** add a 12-second GIF at `docs/assets/demo.gif` that cycles through training metrics, the best replay, and a small network diagram. Use it directly below this opening section.

## Why FlappyLearn

- **Learn by watching:** see generations improve in real time through metrics and replay artifacts.
- **Real neuroevolution:** population search evolves weights, recurrent memory, activations, topology, mutation scales, and exploration behavior.
- **Deterministic simulator:** seeded Flappy Bird physics makes experiments reproducible and easy to benchmark.
- **No hidden cheating:** the agent observes the current game state only, not future pipes or collision answers.
- **Dependency-light:** the core package uses Python and NumPy; visualizations are plain HTML/SVG.
- **Contributor-friendly:** tests, configs, CI, governance, release docs, issue templates, and educational docs are included.

## Demo

Run a smoke experiment, then open the generated HTML files:

```bash
python -m pip install -e ".[dev]"
python -m flappylearn train --config configs/smoke.json
python -m flappylearn replay --checkpoint runs/smoke/latest/best.json --output runs/smoke/latest/best_replay.html
python -m flappylearn visualize --metrics runs/smoke/latest/metrics.jsonl --output runs/smoke/latest/metrics.html
```

Useful local artifacts after a run:

- `runs/smoke/latest/best_replay.html`: animated browser replay of the best learned policy.
- `runs/smoke/latest/metrics.html`: SVG learning curves and recent-generation table.
- `runs/smoke/latest/best.json`: serialized best genome and metadata.
- `runs/smoke/latest/checkpoints/`: population snapshots for resume and inspection.

## Screenshots And GIFs

Add these assets before a public announcement:

| Asset | Path | Purpose |
| --- | --- | --- |
| Hero GIF | `docs/assets/demo.gif` | First visual proof that the AI learns. |
| Replay screenshot | `docs/assets/replay.png` | Shows the game state and learned behavior. |
| Metrics screenshot | `docs/assets/metrics.png` | Shows generation-over-generation progress. |
| Social card | `docs/assets/flappylearn-social-card.png` | GitHub social preview and launch posts. |
| Logo | `docs/assets/logo.png` | Repository identity and docs header. |

## Quick Start

```bash
git clone https://github.com/ibnshafi/FlappyLearn.git
cd flappylearn
python -m pip install -e ".[dev]"
pytest
python -m flappylearn train --config configs/smoke.json
```

Prefer not to install the package while exploring? The helper scripts add `src` to `PYTHONPATH`:

```bash
python scripts/train.py --config configs/smoke.json
python scripts/evaluate.py --checkpoint runs/smoke/latest/best.json --episodes 20
python scripts/benchmark.py --checkpoint runs/smoke/latest/best.json --episodes 20
python scripts/replay.py --checkpoint runs/smoke/latest/best.json --output runs/smoke/latest/replay.html
```

## Installation

FlappyLearn requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Usage

Train a learner:

```bash
python -m flappylearn train --config configs/default.json
```

Resume from a population checkpoint:

```bash
python -m flappylearn train --resume runs/flappylearn/latest/checkpoints/checkpoint_00025.json
```

Evaluate a best genome:

```bash
python -m flappylearn eval --checkpoint runs/flappylearn/latest/best.json --episodes 100
```

Benchmark against random and no-op baselines:

```bash
python -m flappylearn benchmark --checkpoint runs/flappylearn/latest/best.json --episodes 100 --output runs/flappylearn/latest/benchmark.json
```

Render a replay:

```bash
python -m flappylearn replay --checkpoint runs/flappylearn/latest/best.json --output runs/flappylearn/latest/best_replay.html
```

## Architecture Overview

```text
configs/*.json
     |
     v
flappylearn.cli
     |
     v
DiscoveryTrainer -> evaluate_genome -> FlappyEnv
     |                    |
     |                    v
     |             CircuitPolicy + AdaptiveCircuitGenome
     v
ExperimentTracker -> checkpoints, metrics.jsonl, metrics.html, replay.html
```

Core modules:

- `src/flappylearn/env.py`: deterministic Flappy Bird simulator.
- `src/flappylearn/genome.py`: recurrent circuit genome, mutation, crossover, serialization.
- `src/flappylearn/policy.py`: thresholded executable policy.
- `src/flappylearn/evaluation.py`: seeded episode execution and baselines.
- `src/flappylearn/learner.py`: population training, novelty search, curriculum scheduling.
- `src/flappylearn/visualize.py`: dependency-free HTML/SVG metrics and replay renderers.
- `src/flappylearn/benchmark.py`: robust evaluation against baselines.

See [Architecture](docs/ARCHITECTURE.md) for the deeper design rationale.

## How It Works

FlappyLearn does not train a hand-coded controller. It starts with a population of small recurrent neural circuits. Each circuit observes normalized game state, decides whether to flap, receives a score, and competes for survival.

Each generation:

1. Genomes play seeded Flappy Bird episodes.
2. Fitness rewards score, survival, robustness, and behavioral novelty.
3. The best genomes are preserved.
4. New genomes are produced through crossover and mutation.
5. The curriculum gradually hardens toward target game dynamics.
6. Metrics, checkpoints, and replay artifacts are written to disk.

## AI Explanation

The agent is a compact recurrent neural circuit. It has direct observation links, hidden memory, heterogeneous activations, and an output threshold. Recurrence matters because Flappy Bird is about timing: the best action now depends on recent vertical movement and previous flaps.

The system uses neuroevolution instead of gradient-heavy deep reinforcement learning because Flappy Bird has a tiny observation space, a binary action, deterministic physics, and sparse score events. A population of small policies can search this space quickly while staying transparent enough for education.

## Evolution Explanation

FlappyLearn combines several evolutionary ideas:

- **Selection:** better policies are more likely to produce the next generation.
- **Elitism:** the strongest genomes survive unchanged.
- **Mutation:** weights, biases, activations, thresholds, topology, and mutation rates can change.
- **Crossover:** two parent circuits can combine into a child.
- **Novelty pressure:** unusual behaviors receive a small bonus to prevent premature collapse.
- **Curriculum:** training starts easier and moves toward the target environment.

## Configuration

Configs are JSON files under `configs/`.

- `configs/smoke.json`: fast verification run for local development and CI-style checks.
- `configs/default.json`: balanced default training profile.
- `configs/train_full.json`: longer experiment for stronger policies.

Common settings:

| Key | Meaning |
| --- | --- |
| `run.name` | Output folder name under `runs/`. |
| `run.seed` | Reproducibility seed. |
| `environment.pipe_gap` | Vertical gap between pipes. |
| `environment.scroll_speed` | Horizontal pipe speed. |
| `trainer.population_size` | Number of genomes per generation. |
| `trainer.generations` | Maximum generations to train. |
| `trainer.episodes_per_genome` | Training episodes per candidate. |
| `trainer.evaluation_episodes` | Higher-quality evaluation episodes for best candidates. |
| `trainer.max_steps` | Per-episode step cap. |
| `trainer.target_score` | Early stopping score. |

Every CLI training command supports practical overrides such as `--seed`, `--generations`, `--population`, `--max-steps`, `--output-dir`, and `--run-name`.

## Training Modes

- **Smoke:** fast correctness run for development.
- **Default:** balanced experiment suitable for everyday exploration.
- **Full evolution:** longer run intended to discover stronger policies and richer training curves.
- **Resume:** continue from a saved population checkpoint.
- **Profile:** run a short profiled training session to locate performance bottlenecks.

## Visualization Modes

- **Metrics HTML:** line chart for best evaluation score and population mean.
- **Replay HTML:** canvas animation of a saved genome playing a seeded episode.
- **Benchmark JSON:** agent, random baseline, and no-op baseline results for reproducibility.
- **Checkpoint JSON:** inspect genome topology, weights, mutation metadata, and run metadata.

## Documentation

The documentation is designed for MkDocs Material because it is simple, fast, Python-native, and excellent for open-source educational projects.

- [Quick start](docs/index.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Neuroevolution](docs/concepts/neuroevolution.md)
- [NEAT](docs/concepts/neat.md)
- [Neural networks](docs/concepts/neural-networks.md)
- [Fitness functions](docs/concepts/fitness-functions.md)
- [Genetic algorithms](docs/concepts/genetic-algorithms.md)
- [Configuration reference](docs/reference/configuration.md)
- [Visualization guide](docs/reference/visualizations.md)
- [Launch audit](LAUNCH_AUDIT.md)

## Contributing

FlappyLearn welcomes contributors who care about readable learning systems, reproducible experiments, and kind technical writing.

Start with [CONTRIBUTING.md](CONTRIBUTING.md), then look at [First-time contributor onboarding](docs/community/first-time-contributors.md). Every pull request should include tests or a clear explanation of why tests are not needed.

## Roadmap

- **Phase 1:** launch-ready repository, docs, CI, reproducible artifacts.
- **Phase 2:** richer visualizations, network diagrams, and experiment comparisons.
- **Phase 3:** optional browser dashboard and interactive educational walkthroughs.
- **Phase 4:** research extensions, published benchmarks, and community challenge tracks.

The full roadmap lives in [docs/ROADMAP.md](docs/ROADMAP.md).

## FAQ

**Is this NEAT?**  
It is NEAT-inspired, but not a strict NEAT implementation. FlappyLearn evolves topology and recurrent circuits while using its own compact genome representation and fitness shaping.

**Does the agent see future pipes?**  
No. It observes normalized current game state and active pipe information exposed by the environment.

**Why not use DQN or PPO?**  
Those methods can work, but they add machinery that is not necessary for this compact deterministic control problem. Neuroevolution is easier to visualize and explain here.

**Can I use this in a class or workshop?**  
Yes. The MIT license allows educational and commercial use, and the docs include beginner-focused concept pages.

**Where do generated runs go?**  
Training writes to `runs/<run-name>/<timestamp>/` and mirrors the latest run to `runs/<run-name>/latest/`.

## License

FlappyLearn is released under the [MIT License](LICENSE). MIT is the best fit for this project because it maximizes adoption, permits educational and commercial use, and keeps contribution obligations simple for beginners.

## Credits

FlappyLearn is built by its contributors for people who want machine learning systems that are understandable, hackable, and fun to watch.
