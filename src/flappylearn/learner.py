from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from flappylearn.checkpoint import save_best_genome, save_population_checkpoint
from flappylearn.config import Config, EnvConfig, env_with
from flappylearn.env import OBSERVATION_SIZE
from flappylearn.evaluation import evaluate_genome, record_genome_replay
from flappylearn.genome import AdaptiveCircuitGenome
from flappylearn.tracking import ExperimentTracker
from flappylearn.visualize import write_metrics_html, write_replay_html


@dataclass
class GenomeScore:
    genome: AdaptiveCircuitGenome
    fitness: float
    score_mean: float
    score_min: float
    score_max: float
    reward_mean: float
    steps_mean: float
    novelty: float
    complexity: float
    signature: np.ndarray

    def to_metrics(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome.genome_id,
            "fitness": self.fitness,
            "score_mean": self.score_mean,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "reward_mean": self.reward_mean,
            "steps_mean": self.steps_mean,
            "novelty": self.novelty,
            "complexity": self.complexity,
            "hidden_size": self.genome.hidden_size,
        }


class NoveltyArchive:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.signatures: list[np.ndarray] = []

    def novelty(self, signature: np.ndarray, k: int = 8) -> float:
        if not self.signatures:
            return 0.0
        distances = np.array(
            [float(np.linalg.norm(signature - item)) for item in self.signatures],
            dtype=np.float64,
        )
        nearest = np.sort(distances)[: min(k, len(distances))]
        return float(np.mean(nearest))

    def add_many(self, signatures: list[np.ndarray]) -> None:
        self.signatures.extend(np.asarray(sig, dtype=np.float64) for sig in signatures)
        if len(self.signatures) > self.max_size:
            self.signatures = self.signatures[-self.max_size :]


class CurriculumScheduler:
    def __init__(self, enabled: bool, target_score: int):
        self.enabled = enabled
        self.target_score = max(1, target_score)
        self.phase = 0.0 if enabled else 1.0
        self.best_seen = 0.0

    def env_for_generation(self, base: EnvConfig) -> EnvConfig:
        if not self.enabled:
            return base
        phase = self.phase
        return env_with(
            base,
            pipe_gap=base.pipe_gap + (1.0 - phase) * 42.0,
            scroll_speed=base.scroll_speed * (0.82 + 0.18 * phase),
            gravity=base.gravity * (0.9 + 0.1 * phase),
        )

    def update(self, score_mean: float, score_max: float) -> None:
        if not self.enabled:
            self.phase = 1.0
            return
        self.best_seen = max(self.best_seen, score_max)
        target_fraction = min(1.0, self.best_seen / self.target_score)
        performance_phase = min(1.0, target_fraction * 1.25)
        if score_mean >= max(1.0, self.phase * self.target_score * 0.08):
            performance_phase = max(performance_phase, self.phase + 0.035)
        self.phase = float(np.clip(max(self.phase, performance_phase), 0.0, 1.0))

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "best_seen": self.best_seen,
            "target_score": self.target_score,
        }


class DiscoveryTrainer:
    def __init__(
        self,
        config: Config,
        *,
        tracker: ExperimentTracker | None = None,
        initial_population: list[AdaptiveCircuitGenome] | None = None,
        start_generation: int = 0,
    ):
        self.config = config
        self.rng = np.random.default_rng(config.run.seed)
        self.tracker = tracker or ExperimentTracker(config)
        self.start_generation = start_generation
        self.population = initial_population or self._initial_population()
        self.archive = NoveltyArchive()
        self.curriculum = CurriculumScheduler(
            enabled=config.trainer.curriculum,
            target_score=config.trainer.target_score,
        )
        self.novelty_weight = config.trainer.novelty_weight
        self.episodes_per_genome = config.trainer.episodes_per_genome
        self.best_genome: AdaptiveCircuitGenome | None = None
        self.best_key = (-1.0, -1.0)
        self.stall_generations = 0

    def _initial_population(self) -> list[AdaptiveCircuitGenome]:
        return [
            AdaptiveCircuitGenome.random(
                self.rng,
                input_size=OBSERVATION_SIZE,
                hidden_units=self.config.trainer.hidden_units,
            )
            for _ in range(self.config.trainer.population_size)
        ]

    def train(self) -> dict[str, Any]:
        start = perf_counter()
        final_metrics: dict[str, Any] = {}
        for generation in range(self.start_generation, self.config.trainer.generations):
            generation_start = perf_counter()
            train_env = self.curriculum.env_for_generation(self.config.environment)
            train_seeds = self._seeds(generation, self.episodes_per_genome, salt=11)
            scored = self._evaluate_population(train_env, train_seeds)
            scored.sort(key=lambda item: item.fitness, reverse=True)

            best = scored[0]
            eval_seeds = self._seeds(
                generation,
                self.config.trainer.evaluation_episodes,
                salt=7919,
            )
            eval_result = evaluate_genome(
                best.genome,
                self.config.environment,
                eval_seeds,
                self.config.trainer.max_steps,
            )
            is_new_best = self._consider_best(best.genome, eval_result, generation)
            self.curriculum.update(eval_result["score_mean"], eval_result["score_max"])
            self._adapt_meta(is_new_best, eval_result["score_mean"])

            metrics = self._generation_metrics(
                generation,
                scored,
                best,
                eval_result,
                generation_start,
            )
            self.tracker.log_metrics(metrics)
            final_metrics = metrics
            self.archive.add_many([record.signature for record in scored[: max(4, len(scored) // 5)]])

            if is_new_best and self.config.trainer.replay_best:
                self._write_best_replay(generation)
            should_checkpoint = (
                generation % self.config.trainer.checkpoint_interval == 0
                or generation == self.config.trainer.generations - 1
            )
            if should_checkpoint:
                self._save_checkpoint(generation, metrics)

            print(
                f"gen={generation:04d} fitness={best.fitness:.3f} "
                f"eval_score={eval_result['score_mean']:.3f} "
                f"pop_mean={metrics['population_score_mean']:.3f} "
                f"phase={self.curriculum.phase:.2f}"
            )

            if eval_result["score_mean"] >= self.config.trainer.target_score:
                break
            self.population = self._next_population(scored)

        summary = {
            "best_score_mean": self.best_key[0],
            "best_reward_mean": self.best_key[1],
            "elapsed_seconds": perf_counter() - start,
            "run_dir": str(self.tracker.run_dir),
            "latest_dir": str(self.tracker.latest_dir),
            "final_metrics": final_metrics,
        }
        self.tracker.write_summary(summary)
        write_metrics_html(self.tracker.metrics_path, self.tracker.run_dir / "metrics.html")
        write_metrics_html(self.tracker.latest_metrics_path, self.tracker.latest_dir / "metrics.html")
        return summary

    def _evaluate_population(self, env_config: EnvConfig, seeds: list[int]) -> list[GenomeScore]:
        scored: list[GenomeScore] = []
        for genome in self.population:
            result = evaluate_genome(genome, env_config, seeds, self.config.trainer.max_steps)
            novelty = self.archive.novelty(result["signature"])
            complexity = genome.complexity()
            fitness = (
                result["score_mean"] * 55.0
                + result["score_min"] * 20.0
                + result["reward_mean"]
                + result["steps_mean"] / max(1, self.config.trainer.max_steps)
                + novelty * self.novelty_weight
                - complexity * self.config.trainer.complexity_penalty
            )
            scored.append(
                GenomeScore(
                    genome=genome,
                    fitness=float(fitness),
                    score_mean=float(result["score_mean"]),
                    score_min=float(result["score_min"]),
                    score_max=float(result["score_max"]),
                    reward_mean=float(result["reward_mean"]),
                    steps_mean=float(result["steps_mean"]),
                    novelty=float(novelty),
                    complexity=float(complexity),
                    signature=result["signature"],
                )
            )
        return scored

    def _consider_best(
        self,
        genome: AdaptiveCircuitGenome,
        eval_result: dict[str, Any],
        generation: int,
    ) -> bool:
        key = (float(eval_result["score_mean"]), float(eval_result["reward_mean"]))
        if key <= self.best_key:
            self.stall_generations += 1
            return False
        self.best_key = key
        self.best_genome = genome.copy()
        self.stall_generations = 0
        metadata = {
            "generation": generation,
            "eval": {
                name: value.tolist() if hasattr(value, "tolist") else value for name, value in eval_result.items()
            },
            "config": self.config.to_dict(),
            "curriculum": self.curriculum.to_dict(),
        }
        save_best_genome(self.best_genome, self.tracker.run_dir / "best.json", metadata)
        save_best_genome(self.best_genome, self.tracker.latest_dir / "best.json", metadata)
        return True

    def _adapt_meta(self, improved: bool, eval_score: float) -> None:
        if improved:
            self.novelty_weight = max(
                self.config.trainer.novelty_weight * 0.35,
                self.novelty_weight * 0.97,
            )
        else:
            growth = 1.04 if self.stall_generations < 6 else 1.12
            self.novelty_weight = min(2.5, self.novelty_weight * growth + 0.002)

        if eval_score >= 10 and self.episodes_per_genome < self.config.trainer.evaluation_episodes:
            self.episodes_per_genome = min(
                self.config.trainer.evaluation_episodes,
                self.episodes_per_genome + 1,
            )

    def _next_population(self, scored: list[GenomeScore]) -> list[AdaptiveCircuitGenome]:
        elites = [item.genome.copy() for item in scored[: self.config.trainer.elites]]
        next_population = elites[:]
        while len(next_population) < self.config.trainer.population_size:
            parent_a = self._tournament(scored).genome
            if self.rng.random() < 0.60:
                parent_b = self._tournament(scored).genome
                child = AdaptiveCircuitGenome.crossover(self.rng, parent_a, parent_b)
            else:
                child = parent_a.copy()
            child = child.mutate(self.rng, self.config.trainer.max_hidden_units)
            next_population.append(child)
        return next_population

    def _tournament(self, scored: list[GenomeScore]) -> GenomeScore:
        size = min(self.config.trainer.tournament_size, len(scored))
        indices = self.rng.choice(len(scored), size=size, replace=False)
        contestants = [scored[int(index)] for index in indices]
        contestants.sort(key=lambda item: item.fitness, reverse=True)
        return contestants[0]

    def _generation_metrics(
        self,
        generation: int,
        scored: list[GenomeScore],
        best: GenomeScore,
        eval_result: dict[str, Any],
        generation_start: float,
    ) -> dict[str, Any]:
        population_scores = np.array([item.score_mean for item in scored], dtype=np.float64)
        population_fitness = np.array([item.fitness for item in scored], dtype=np.float64)
        return {
            "generation": generation,
            "elapsed_seconds": perf_counter() - generation_start,
            "population_score_mean": float(np.mean(population_scores)),
            "population_score_max": float(np.max(population_scores)),
            "population_fitness_mean": float(np.mean(population_fitness)),
            "population_fitness_max": float(np.max(population_fitness)),
            "best_score_mean": best.score_mean,
            "best_score_min": best.score_min,
            "best_score_max": best.score_max,
            "best_reward_mean": best.reward_mean,
            "best_steps_mean": best.steps_mean,
            "best_complexity": best.complexity,
            "best_hidden_size": best.genome.hidden_size,
            "eval_score_mean": float(eval_result["score_mean"]),
            "eval_score_min": float(eval_result["score_min"]),
            "eval_score_max": float(eval_result["score_max"]),
            "eval_reward_mean": float(eval_result["reward_mean"]),
            "novelty_weight": self.novelty_weight,
            "episodes_per_genome": self.episodes_per_genome,
            "curriculum_phase": self.curriculum.phase,
            "best_genome_id": best.genome.genome_id,
        }

    def _save_checkpoint(self, generation: int, metrics: dict[str, Any]) -> None:
        if self.best_genome is None:
            return
        metadata = {
            "metrics": metrics,
            "config": self.config.to_dict(),
            "curriculum": self.curriculum.to_dict(),
            "novelty_weight": self.novelty_weight,
            "episodes_per_genome": self.episodes_per_genome,
        }
        filename = f"checkpoint_{generation:05d}.json"
        save_population_checkpoint(
            self.tracker.checkpoint_dir / filename,
            generation=generation,
            population=self.population,
            best=self.best_genome,
            metadata=metadata,
        )
        save_population_checkpoint(
            self.tracker.latest_dir / "checkpoints" / filename,
            generation=generation,
            population=self.population,
            best=self.best_genome,
            metadata=metadata,
        )

    def _write_best_replay(self, generation: int) -> None:
        if self.best_genome is None:
            return
        replay = record_genome_replay(
            self.best_genome,
            self.config.environment,
            seed=self.config.run.seed + generation * 313 + 17,
            max_steps=self.config.trainer.max_steps,
        )
        self.tracker.write_artifact_json("best_replay.json", replay)
        write_replay_html(self.tracker.run_dir / "best_replay.json", self.tracker.run_dir / "best_replay.html")
        write_replay_html(self.tracker.latest_dir / "best_replay.json", self.tracker.latest_dir / "best_replay.html")

    def _seeds(self, generation: int, count: int, *, salt: int) -> list[int]:
        base = int(self.config.run.seed + generation * 100_003 + salt)
        return [base + index * 9_973 for index in range(count)]


def checkpoint_run_dir(path: str | Path) -> Path:
    return Path(path).resolve()
