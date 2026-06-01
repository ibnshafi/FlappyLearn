# PROJECT KNOWLEDGE BASE: FlappyLearn

Date: 2026-05-31
Repository: flappylearn (workspace root)

> Executive summary

FlappyLearn is a compact, dependency-light research system that evolves recurrent policy circuits to play a deterministic Flappy Bird simulator. The project is a neuroevolution research harness: a deterministic environment, a self-adaptive genome representation (recurrent circuits with per-unit activations and structural mutations), a genetic-style trainer with novelty incentives and curriculum scheduling, experiment tracking, checkpointing, benchmarking, and HTML-based visualizations and replay exports.

This knowledge base documents the repository structure, execution flow, algorithms, models, design decisions, failure modes, and recommended improvements so that an engineer can rebuild, extend, or port the system.

---

## System Architecture (high level)

- Simulator: `FlappyEnv` — deterministic, headless implementation of Flappy Bird with a fixed observation vector of size 10.
- Genome / Policy: `AdaptiveCircuitGenome` + `CircuitPolicy` — an adaptive recurrent circuit with input-to-hidden, recurrent, and hidden-to-output pathways plus a direct input-to-output shortcut. Hidden units have per-unit activation names.
- Trainer: `DiscoveryTrainer` — population-based evolution with tournament selection, elitism, crossover, mutation (weights + topology), novelty archive and curriculum scheduler.
- IO/Experiment: `ExperimentTracker`, checkpointing helpers, CLI (`flappylearn.cli`) and thin scripts in `/scripts/` for local convenience.
- Reporting: `visualize.write_metrics_html` and `write_replay_html` produce dependency-free HTML artifacts for inspection.

---

## Directory Structure (project tree)

- pyproject.toml
- README.md
- configs/
  - default.json
  - smoke.json
  - train_full.json (added/edited)
- docs/
  - ARCHITECTURE.md
- runs/ (runtime outputs)
  - <run_name>/
    - <timestamp>/
      - metrics.jsonl
      - checkpoints/
      - best.json
      - best_replay.json
      - metrics.html
      - summary.json
    - latest/ (overwritten)
- scripts/
  - train.py
  - benchmark.py
  - replay.py
- src/
  - flappylearn/
    - __init__.py
    - __main__.py
    - cli.py
    - config.py
    - env.py
    - genome.py
    - learner.py
    - evaluation.py
    - policy.py
    - checkpoint.py
    - tracking.py
    - visualize.py
    - benchmark.py
    - profiling.py
- tests/
  - test_*.py

(See `pyproject.toml` for declared package entry point `flappylearn = flappylearn.cli:main`.)

---

## Dependency Graph (module → dependency)

- `scripts/*` → `src` (`flappylearn.cli`) (simple wrappers)
- `flappylearn.cli` → `flappylearn.*` (primary entry point)
- `flappylearn.learner` → `flappylearn.genome`, `flappylearn.evaluation`, `flappylearn.tracking`, `flappylearn.visualize`, `flappylearn.checkpoint`
- `flappylearn.evaluation` → `flappylearn.env`, `flappylearn.policy`, `flappylearn.genome`
- `flappylearn.policy` → `flappylearn.genome` (only type reference at bottom)
- `flappylearn.genome` → numpy
- `flappylearn.env` → config dataclasses
- `flappylearn.tracking` → filesystem, config
- `flappylearn.benchmark` → `flappylearn.evaluation`, `flappylearn.checkpoint`

Core modules: `learner.py`, `genome.py`, `evaluation.py`, `env.py` (tight coupling around evolution loop). Utility modules: `checkpoint.py`, `tracking.py`, `visualize.py` (I/O and reporting). CLI glue: `cli.py`.

No problematic circular imports observed in the current layout; modules reference each other in a layered fashion (genome/env/policy → evaluation → learner → tracking/visualize/checkpoint).

---

## Execution Flow (end-to-end)

1. Program Entry Points
   - Developer: `python -m flappylearn train --config <path>` (via `pyproject` entrypoint)
   - Scripts: `python scripts/train.py --config ...` (adds `src` to PYTHONPATH and calls `flappylearn.cli.main`)

2. CLI parsing (`flappylearn.cli.build_parser`)
   - Subcommands: `train`, `eval`, `benchmark`, `replay`, `visualize`, `profile`.
   - `--config` points to a JSON file; defaults to `configs/default.json`.
   - Overrides from CLI may change seed, population, generations, max-steps, etc.

3. Configuration Loading
   - `load_config(path)` reads JSON into dataclasses: `RunConfig`, `EnvConfig`, `TrainerConfig`.
   - `override_config` applies CLI overrides to dataclass copies.

4. Resource allocation / tracking
   - `ExperimentTracker` computes run directories, prepares `latest` and run timestamped directories, writes `config.json`.

5. Main Training Loop (`DiscoveryTrainer.train`)
   - Initialize population (`AdaptiveCircuitGenome.random`) sized by `trainer.population_size`.
   - For each generation:
     - Build per-generation training env via `CurriculumScheduler.env_for_generation(config.environment)`.
     - Evaluate population via `_evaluate_population`: for each genome, call `evaluate_genome(genome, env_config, seeds, max_steps)` which runs multiple seeded episodes and returns mean/min/max etc.
     - Compute novelty (via `NoveltyArchive.novelty` on the per-episode signature mean), and compute `fitness` using the configured weighting formula (score_mean * 55 + score_min * 20 + reward_mean + steps_mean / max_steps + novelty * novelty_weight - complexity * complexity_penalty).
     - Sort population by fitness, keep `elites` directly, generate rest by tournament selection and either crossover (probabilistic) or asexual copying, then mutate (weights and possibly topology).
     - Evaluate best candidate with `evaluation_episodes` seeds to produce `eval_result` used for curriculum update and early stopping.
     - Log metrics to `metrics.jsonl` and write periodic checkpoints and best genome.

6. Persistence and Artifacts
   - `metrics.jsonl`: one record per generation (JSON lines). `visualize.write_metrics_html` renders an SVG and table.
   - `checkpoints/checkpoint_XXXXX.json`: population snapshots with `best_genome_id`.
   - `best.json`: single-genome checkpoint with metadata.
   - `best_replay.json` and `best_replay.html`: replay frames for inspection.
   - `summary.json`: final training summary.

7. Termination
   - Training stops either after configured generations or when the `eval_score_mean` >= `trainer.target_score`.
   - On termination, final metrics are written and `metrics.html` is produced for both run_dir and latest.

---

## Core Concepts & Rationale

This section describes important design choices and where they appear in code.

- Neuroevolution (custom GA-style): the system evolves neural-like circuits. Implementation: `genome.AdaptiveCircuitGenome` with `random`, `mutate`, `crossover` and `complexity` scoring.
  - Rationale: deterministic environment with sparse reward benefits from population search and explicit exploration schedules.

- Recurrent adaptive circuits vs MLPs: stateful `w_rec` recurrence and explicit memory vector. Implemented in `genome.forward` which computes `new_memory = tanh(0.62 * memory + 0.38 * hidden)`.
  - Rationale: temporal dependencies (timing flaps) are central to Flappy Bird; recurrence stores short-term memory.

- Novelty search augmentation: novelty of behavior signatures encourages behavioral diversity and helps escape local optima. Implemented via `NoveltyArchive` and added to fitness multiplied by `novelty_weight`.

- Curriculum scheduler: environment difficulty (pipe_gap, scroll_speed, gravity) is annealed from easy to hard based on `curriculum.phase`. Implemented in `CurriculumScheduler.env_for_generation` and updated in `DiscoveryTrainer.curriculum.update`.

- Fitness shaping: heavy weight on `score_mean` and `score_min` to prefer robust policies, with penalties for complexity (promote sparse controllers) and novelty boosting.

- Structural (topology) mutations: genomes can add or remove hidden units (calls to `_add_unit`/_remove_unit). Topology change is controlled by `MutationParams.topology_rate`.

- Per-unit heterogeneous activations: each hidden unit selects an activation function from `ACTIVATIONS` and can change activation via mutation.

- Determinism via seeds: All randomness is seedable; evaluation and training use deterministic seeds spaces constructed in `_seeds()`.

---

## Implementation Deep Dive (significant classes)

### `FlappyEnv` (`src/flappylearn/env.py`)
- Purpose: deterministic simulation of Flappy Bird for evaluation and training.
- Key state: `bird_y`, `bird_velocity`, `pipes` list, `score`, `steps`.
- API:
  - `reset(seed) -> observation` (initializes pipes using RNG)
  - `step(action) -> StepResult` (applies flap, steps physics, moves pipes, computes reward and done)
  - `observe() -> np.ndarray` returns 10-d observation vector (normalized values)
  - `next_pipe()` returns the closest pipe in front of the bird
- Observation features (10 elements): y_norm, velocity_norm, dx to pipe, gap_center, bird-gap diff, bird-gap-top distance, gap-bottom-bird distance, next pipe dx, next pipe gap center, last_action
- Failure modes: inaccurate clipping of obs, mismatch between `input_size` expected by genome and `OBSERVATION_SIZE`.

Complexity: O(1) per step; memory O(P) where P is number of pipes in memory (bounded by spawn_x & spacing).

---

### `AdaptiveCircuitGenome` (`src/flappylearn/genome.py`)
- Purpose: encode a stateful controller (recurrent circuit) with evolvable topology.
- Internal state:
  - weight matrices: `w_in` (hidden x input), `w_rec` (hidden x hidden), `w_out` (hidden), `w_direct` (input)
  - `b_hidden` vector, `b_out` scalar
  - `activations` list per hidden unit
  - `threshold`, `exploration_noise`, `mutation: MutationParams` metadata
- Public methods:
  - `random(rng, input_size, hidden_units)` — initialize a random genome
  - `forward(observation, memory)` — compute scalar logit and updated memory (recurrent update)
  - `mutate(rng, max_hidden_units)` — mutate weights, biases, possibly topology and activation choices
  - `crossover(rng, parent_a, parent_b)` — combine rows/blocks from parents
  - `copy()`, `to_dict()`, `from_dict()` — standard serialization utilities
  - `complexity()` — heuristic complexity measure (counts active weights and hidden size)
- Mutation details:
  - Weight mutation: `_mutate_array` applies Gaussian perturbations where a mask indicates which elements mutate; uses `weight_sigma` and `weight_mutation_rate`.
  - Recurrent weights are mutated at `0.6 * weight_sigma` by default.
  - Topology: with `topology_rate` a new unit may be added (`_add_unit`) or with `topology_rate * 0.45` a unit may be removed.
  - Activation mutation: per-unit swaps to a randomly chosen activation from `ACTIVATIONS` at `activation_rate`.
- Forward pass details:
  - pre-activation: `pre = w_in @ obs + w_rec @ memory + b_hidden`
  - per-activation application: groups pre entries by activation name then apply `_activation`.
  - new memory: `tanh(0.62 * memory + 0.38 * hidden)` — this interpolates previous memory with the computed hidden activity; a design choice to bias stability.
  - logit: `w_out . new_memory + w_direct . obs + b_out` — thresholded in policy.

Failure modes:
- Topology changes re-dimension `w_rec` and can destabilize dynamics if added/removed too frequently.
- Heterogeneous activations can lead to non-monotonic instabilities (e.g., `sin`, `gauss`) — note: current code restricts `ACTIVATIONS` to `tanh/relu/elu` after a previous patch.

Complexity and scaling:
- Forward cost O(H^2 + H*I) where H=hidden_size, I=input_size.
- Memory dominated by weight matrices; topology growth increases squarely with hidden units due to `w_rec`.

---

### `DiscoveryTrainer` (`src/flappylearn/learner.py`)
- Purpose: orchestrate population evaluation and evolution.
- Public API: `train()` returns final summary.
- Internals:
  - population: list of genomes
  - novelty archive: `NoveltyArchive` (keeps recent signatures up to `max_size`)
  - curriculum: `CurriculumScheduler`
  - novelty weight and episodes per genome that adapt during training
- Key algorithmic choices:
  - Fitness composition: heavy weighting of `score_mean` (x55) and `score_min` (x20) to promote robust, high-scoring agents while `reward_mean` and `steps_mean` provide additional signal.
  - Selection: tournament selection with `tournament_size` (configurable).
  - Reproduction: elitism copies top N; for the rest, with probability p crossover else asexual copy; mutated child appended.
  - Mutation hyperparameters are adaptive per-genome through `MutationParams.adapted`.
  - Early stopping on reaching `config.trainer.target_score` on `eval_score_mean`.

Failure modes and tradeoffs:
- Too-high structural mutation rate causes catastrophic changes in recurrent dynamics.
- Tournament size affects selection pressure (in code, previously 5; recommended reductions were applied).
- Crossover mixing of recurrent matrices may create inconsistent dynamics if parents have different hidden sizes; the crossover handles this by working on shared indices only.

---

### `Evaluation` and `Policy`
- `run_episode` (evaluation.py): runs deterministic episodes using provided seeds, builds a 5-element signature for novelty (score/50, steps/max_steps, action_count/denom, y_error_sum/denom, velocity_abs_sum/denom).
- `evaluate_genome` executes multiple seeds and returns mean/min/max scores plus average signature.
- `CircuitPolicy.act` uses genome.forward and thresholding; exploration is implemented by adding Gaussian noise with `exploration_noise` when not deterministic.

Key points:
- `evaluate_genome` returns aggregated statistics used in fitness computation and curriculum adaptation.
- `run_episode` prepares replay frames when requested (used by `record_genome_replay`).

---

### `Checkpointing` and `Tracking`
- `save_best_genome` and `save_population_checkpoint` serialize genomes and populations to JSON files; `load_genome`/`load_population_checkpoint` read them back and reconstruct `AdaptiveCircuitGenome` objects.
- `ExperimentTracker` manages run directories, writes `config.json`, appends metrics to `metrics.jsonl`, and writes artifacts into both `run_dir` and `latest`.
- `metrics.jsonl` records a dictionary per generation with the following important fields: `generation`, `population_score_mean`, `best_score_mean`, `best_score_max`, `eval_score_mean`, `novelty_weight`, `curriculum_phase`, etc.

---

## AI System Analysis (detailed)

### Model Architecture
- Recurrent single hidden-layer circuit with per-unit activations.
- Inputs: 10-dimensional observation vector.
- Hidden: variable-size H (config `hidden_units`, grows via mutation up to `max_hidden_units`).
- Recurrent connections: full HxH `w_rec` matrix.
- Hidden-to-output: `w_out` length H plus direct input `w_direct` length I.
- Memory update: `new_memory = tanh(0.62 * memory + 0.38 * hidden)`.

### Genome representation
- Dense numeric arrays for all weights/biases, plus metadata for activations, thresholds, and mutation hyperparameters.
- `to_dict` and `from_dict` allow full serialization; population checkpoints store entire populations as JSON.

### Training Pipeline
- Population size: configured via `trainer.population_size`.
- Generational loop: evaluate -> compute fitness -> selection + reproduction -> mutation -> archive & checkpoint.
- Evaluation seeds: `episodes_per_genome` seeds per genome during population evaluation, `evaluation_episodes` used for the best genome evaluation.
- Early stopping: `eval_score_mean >= trainer.target_score`.

### Selection Methods
- Tournament selection (`_tournament`) picking `tournament_size` samples without replacement and returning the highest fitness.

### Genetic Operators
- Crossover: row-wise mixing for `w_in`, `w_out`, `b_hidden`, `w_rec` element-wise mixing with a mask, partial mixing of `w_direct`, and metadata mixing for thresholds and mutation params.
- Mutation: Gaussian mutations applied to selected weights (mask-based), occasional resets with larger sigma, topology add/remove, activation swaps at `activation_rate`.

### Fitness function
- Weighted linear combination focused on mean score and minimum score across evaluation episodes; adds `reward_mean`, normalized `steps_mean`, novelty and subtracts `complexity * complexity_penalty`.

### Exploration / Exploitation
- Exploration noise per genome controls stochastic action during evaluation when not deterministic.
- Novelty weight increases when training stalls, implemented in `_adapt_meta`.

---

## Data Flow Diagrams (text form)

1. Training data flow

Input: `configs/*.json` → CLI `--config` → `load_config` → `Config` dataclasses
↓
Init: `DiscoveryTrainer` creates population
↓
For each generation:
  seeds = _seeds(generation)
  for each genome in population:
    evaluate_genome -> run_episode (multiple seeds) -> returns stats + signature
  compute novelty from signatures and archive
  compute fitness for each genome
  select elites, reproduce via tournament, crossover & mutate
  archive.add_many(signatures)
  track metrics -> `metrics.jsonl`
  if new best -> save_best_genome
  periodic checkpoint -> `checkpoints` persisted

Output: `runs/<run>/...` (metrics, best, checkpoints, html replays)

2. Evaluation/benchmark

Input: checkpoint file -> `benchmark_checkpoint` loads genome -> runs `evaluate_genome` with a fresh set of seeds -> returns agent stats and baseline comparisons -> writes JSON if asked

---

## Performance Analysis & Complexity

- Forward pass: O(H^2 + H*I) per step; recurrent `w_rec` is the dominant cost where H can vary.
- Evaluate_genome: cost = (#seeds) * (#steps per episode) * forward cost. This is the main CPU consumer.
- Population evaluation: cost scales linearly with population size and episodes per genome.
- Memory: dominated by population snapshots when saved (checkpoint includes all weights in JSON; large populations + large H produce big checkpoints). JSON serializations are expensive CPU and disk.

Bottlenecks and improvements:
- Bottleneck: JSON-based checkpointing of entire populations (large I/O). Consider binary checkpointing (numpy .npz) or incremental diffs.
- Bottleneck: Python loops over genomes and episodes. Consider vectorized batch evaluations (non-trivial due to variable topology), or parallelization across processes (evaluate genomes in multiple worker processes) since each evaluation is independent.
- Memory: reduce checkpoint frequency or store only `best` plus a compressed population representation.
- Complexity of `w_rec` is quadratic in H; limit `max_hidden_units` for large-scale experiments.

Asymptotic estimates:
- Per-step forward: O(H^2). For population evaluation per generation: O(P * E * S * H^2) where P=population_size, E=episodes_per_genome, S=steps_per_episode.

---

## Tests & Benchmarks
- `tests/` contains unit tests covering CLI, env, genome, and training components.
- `scripts/benchmark.py` provides a convenience bench harness to compute `score_mean` against baselines.

---

## Improvement Opportunities (concise)

1. Parallel evaluation of genomes using `multiprocessing` or joblib to utilize multiple CPU cores (easy win).
2. Binary checkpoint format (np.savez or HDF5) for population snapshots to cut I/O and storage costs.
3. Controlled topology mutations: implement gradual topology changes (e.g., add weights slowly instead of full-row initialization) or perform topology edits only on cloned offspring after several weight refinements.
4. Replace JSON disk writes for metrics with buffered writes or rotate files less frequently to reduce disk churn.
5. Expose `ACTIVATIONS` and `MutationParams` via config to avoid editing source for hyperparameter experiments.
6. Add a parallel evaluation mode and a small worker pool to evaluate genomes concurrently.

---

## Rebuild Guide

1. Clone repository.
2. Install minimal dependencies: `python -m pip install -e .[dev]` (installs `numpy`, and test deps).
3. Run quick smoke test: `python scripts/train.py --config configs/smoke.json`.
4. Produce benchmark for final checkpoint: `python scripts/benchmark.py --checkpoint runs/smoke/latest/best.json --episodes 20`.
5. Generate replay HTML: `python scripts/replay.py --checkpoint runs/smoke/latest/best.json --output runs/smoke/latest/replay.html`.

---

## Extension Guide

- To change the activation pool without editing source: add a config option and pass into `AdaptiveCircuitGenome.random` and `mutate`; currently `ACTIVATIONS` is hard-coded.
- To make mutation hyperparameters configurable: add fields to `TrainerConfig` referencing default `MutationParams` and wire them into new genome instantiation.
- To parallelize: add a `--workers` option in CLI and in `DiscoveryTrainer._evaluate_population` dispatch evaluations to a worker pool.

---

## Full Glossary (selected terms)
- Genome: the data structure encoding a single agent (weights, activations, mutation hyperparams).
- Population checkpoint: a snapshot of the entire population at a generation.
- Best genome: the single genome with maximal evaluation mean score (and reward tie-breaker) seen so far.
- Novelty signature: 5-dimensional vector describing an episode’s behavior used for novelty calculation.
- Curriculum phase: a float [0,1] controlling environment difficulty.

---

## Conclusion

This repository implements a targeted neuroevolution research toolkit for Flappy Bird. The code is compact, well-structured, and intentionally simple: serial evaluation, JSON artifacts for reproducibility, and a small set of configurable knobs. The central engineering tradeoffs are readability and reproducibility (JSON + dataclasses) versus I/O and compute efficiency. Replacing JSON population checkpoints with binary formats and parallelizing independent evaluations will substantially speed experiments.


---

File generated at repository root: `PROJECT_KNOWLEDGE_BASE.md`.
