# Architecture

## Environment Analysis

Flappy Bird has continuous vertical dynamics, a binary action, sparse terminal failures, and score events when pipes are passed. The state needed for strong play is compact: bird height, vertical velocity, position of the next visible pipe, and the current gap. Memory is useful for smoothing action cadence, but a large model is not required. The environment is deterministic once the pipe sequence and action sequence are fixed, which makes it ideal for repeated seeded evaluation.

The action space is binary: flap or do nothing. The important information structure is not high-dimensional perception; it is control under delay and gravity. A strong learner therefore benefits more from fast policy search, robust evaluation, and self-adaptive exploration than from a large gradient-trained network.

## Chosen Learning System

The implemented learner is a self-adaptive sparse recurrent circuit population:

- Each genome is a compact decision circuit with direct observation links, recurrent hidden state, and heterogeneous nonlinear units.
- Evolution changes weights, biases, topology, activation functions, thresholds, and mutation operator parameters.
- Tournament selection and elitism preserve high performers while crossover and mutation explore alternatives.
- Novelty pressure prevents collapse into one brittle flap rhythm.
- A curriculum starts with slightly easier dynamics and automatically hardens toward the target environment.
- Final benchmarking always evaluates on the requested target dynamics.

This is intentionally not DQN/PPO-first. Those methods can work, but their machinery is poorly matched to the tiny state, sparse action, and deterministic physics. A population of small recurrent circuits trains quickly, is easy to checkpoint, and can produce superhuman reactive policies without human demonstrations.

## No Cheating Boundary

The agent observes only normalized current game quantities that a normal game renderer would expose: bird position and velocity, current visible pipes, and relative gap distances. It does not read future pipe samples before they exist in the active state, and it does not receive collision answers in advance.

## Directory Map

- `src/flappylearn/env.py`: deterministic Flappy Bird simulator.
- `src/flappylearn/genome.py`: adaptive circuit genome, mutation, crossover, serialization.
- `src/flappylearn/policy.py`: executable policies.
- `src/flappylearn/evaluation.py`: episode execution, behavior signatures, baselines.
- `src/flappylearn/learner.py`: population training, novelty, curriculum, meta-adaptation.
- `src/flappylearn/tracking.py`: run directories and metrics.
- `src/flappylearn/checkpoint.py`: checkpoint and best-genome IO.
- `src/flappylearn/visualize.py`: HTML/SVG metrics and replay renderers.
- `src/flappylearn/benchmark.py`: robust evaluation against seeded episodes and baselines.
- `src/flappylearn/profiling.py`: training profiler.
- `src/flappylearn/cli.py`: command-line interface.
- `tests/`: deterministic unit and smoke tests.

## Maintainability Review

Strengths:

- The package is layered cleanly: simulator and genome primitives feed evaluation, evaluation feeds training, and tracking/visualization stay at the edges.
- Config dataclasses make experiment settings explicit and serializable.
- Checkpoints and metrics are JSON-based, which makes artifacts inspectable and stable for education.
- The visualization layer avoids heavy browser dependencies.

Risks and improvement opportunities:

- Training is currently single-process; parallel evaluation would improve long experiments.
- Replay and metrics HTML are useful but visually plain; richer network overlays would improve shareability.
- Checkpoint schema versions should be added before `1.0.0` to support future migration.
- More benchmark fixtures would help compare algorithm changes fairly.

Highest-impact technical improvements:

1. Add parallel population evaluation with deterministic seed partitioning.
2. Add schema versioning for checkpoints and metrics.
3. Add interactive network visualization for the best genome.
4. Add experiment comparison tooling for multiple `metrics.jsonl` files.
5. Add performance benchmarks to CI on a scheduled cadence.
