from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from flappylearn.checkpoint import load_genome
from flappylearn.config import EnvConfig
from flappylearn.evaluation import (
    evaluate_genome,
    evaluate_noop_baseline,
    evaluate_random_baseline,
)


def benchmark_checkpoint(
    checkpoint_path: str | Path,
    env_config: EnvConfig,
    *,
    episodes: int = 100,
    seed: int = 123,
    max_steps: int = 12000,
) -> dict[str, Any]:
    genome, metadata = load_genome(checkpoint_path)
    rng = np.random.default_rng(seed)
    seeds = [int(value) for value in rng.integers(0, 2**31 - 1, size=episodes)]
    agent = evaluate_genome(genome, env_config, seeds, max_steps)
    random_baseline = evaluate_random_baseline(env_config, seeds, max_steps)
    noop_baseline = evaluate_noop_baseline(env_config, seeds, max_steps)
    return {
        "checkpoint": str(checkpoint_path),
        "episodes": episodes,
        "seed": seed,
        "agent": _json_ready(agent),
        "random_baseline": random_baseline,
        "noop_baseline": noop_baseline,
        "metadata": metadata,
    }


def write_benchmark(result: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(result), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_ready(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
