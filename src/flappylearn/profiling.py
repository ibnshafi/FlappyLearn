from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from flappylearn.config import Config, override_config
from flappylearn.learner import DiscoveryTrainer


def profile_training(config: Config, output_path: str | Path, generations: int = 3) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    small_config = override_config(
        config,
        generations=generations,
        population_size=min(24, config.trainer.population_size),
        max_steps=min(2400, config.trainer.max_steps),
    )
    profiler = cProfile.Profile()
    profiler.enable()
    DiscoveryTrainer(small_config).train()
    profiler.disable()
    with output.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle).sort_stats("cumtime")
        stats.print_stats(50)
    return output
