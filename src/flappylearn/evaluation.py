from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flappylearn.config import EnvConfig
from flappylearn.env import FlappyEnv
from flappylearn.genome import AdaptiveCircuitGenome
from flappylearn.policy import CircuitPolicy, NoopPolicy, Policy, RandomPolicy


@dataclass
class EpisodeResult:
    score: int
    reward: float
    steps: int
    done: bool
    signature: np.ndarray
    replay: list[dict[str, Any]] | None = None


def run_episode(
    policy: Policy,
    env_config: EnvConfig,
    seed: int,
    max_steps: int,
    *,
    record: bool = False,
) -> EpisodeResult:
    env = FlappyEnv(env_config, seed=seed)
    observation = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    action_count = 0
    y_error_sum = 0.0
    velocity_abs_sum = 0.0
    replay: list[dict[str, Any]] | None = [] if record else None

    if replay is not None:
        replay.append(env.snapshot(action=0, reward=0.0))

    done = False
    for _ in range(max_steps):
        pipe = env.next_pipe()
        y_error_sum += abs(env.bird_y - pipe.gap_y) / env.config.height
        velocity_abs_sum += abs(env.bird_velocity) / env.config.max_velocity

        action = policy.act(observation)
        action_count += int(action)
        result = env.step(action)
        total_reward += result.reward
        observation = result.observation
        done = result.done

        if replay is not None:
            replay.append(env.snapshot(action=action, reward=result.reward))
        if done:
            break

    denom = max(1, env.steps)
    signature = np.array(
        [
            env.score / 50.0,
            env.steps / max(1, max_steps),
            action_count / denom,
            y_error_sum / denom,
            velocity_abs_sum / denom,
        ],
        dtype=np.float64,
    )
    return EpisodeResult(
        score=int(env.score),
        reward=float(total_reward),
        steps=int(env.steps),
        done=done,
        signature=signature,
        replay=replay,
    )


def evaluate_genome(
    genome: AdaptiveCircuitGenome,
    env_config: EnvConfig,
    seeds: list[int],
    max_steps: int,
) -> dict[str, Any]:
    results = [
        run_episode(CircuitPolicy(genome, deterministic=True), env_config, seed, max_steps)
        for seed in seeds
    ]
    scores = np.array([result.score for result in results], dtype=np.float64)
    rewards = np.array([result.reward for result in results], dtype=np.float64)
    steps = np.array([result.steps for result in results], dtype=np.float64)
    signatures = np.vstack([result.signature for result in results])
    return {
        "score_mean": float(np.mean(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "reward_mean": float(np.mean(rewards)),
        "steps_mean": float(np.mean(steps)),
        "signature": np.mean(signatures, axis=0),
    }


def evaluate_random_baseline(env_config: EnvConfig, seeds: list[int], max_steps: int) -> dict[str, Any]:
    results = [
        run_episode(RandomPolicy(probability=0.11, seed=seed + 1009), env_config, seed, max_steps)
        for seed in seeds
    ]
    return summarize_episode_results(results)


def evaluate_noop_baseline(env_config: EnvConfig, seeds: list[int], max_steps: int) -> dict[str, Any]:
    results = [run_episode(NoopPolicy(), env_config, seed, max_steps) for seed in seeds]
    return summarize_episode_results(results)


def summarize_episode_results(results: list[EpisodeResult]) -> dict[str, Any]:
    scores = np.array([result.score for result in results], dtype=np.float64)
    rewards = np.array([result.reward for result in results], dtype=np.float64)
    steps = np.array([result.steps for result in results], dtype=np.float64)
    return {
        "episodes": len(results),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "reward_mean": float(np.mean(rewards)),
        "steps_mean": float(np.mean(steps)),
    }


def record_genome_replay(
    genome: AdaptiveCircuitGenome,
    env_config: EnvConfig,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    result = run_episode(
        CircuitPolicy(genome, deterministic=True),
        env_config,
        seed,
        max_steps,
        record=True,
    )
    return {
        "score": result.score,
        "reward": result.reward,
        "steps": result.steps,
        "frames": result.replay or [],
    }
