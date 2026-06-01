from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, Mapping, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RunConfig:
    name: str = "flappylearn"
    output_dir: str = "runs"
    seed: int = 42


@dataclass(frozen=True)
class EnvConfig:
    width: int = 288
    height: int = 512
    bird_x: float = 64.0
    bird_radius: float = 12.0
    gravity: float = 0.45
    flap_velocity: float = -7.5
    max_velocity: float = 10.0
    pipe_width: float = 52.0
    pipe_gap: float = 110.0
    pipe_spacing: float = 155.0
    scroll_speed: float = 2.5
    spawn_x: float = 320.0
    alive_reward: float = 0.01
    pass_reward: float = 1.0
    death_penalty: float = -1.0


@dataclass(frozen=True)
class TrainerConfig:
    population_size: int = 96
    generations: int = 120
    hidden_units: int = 10
    max_hidden_units: int = 64
    elites: int = 8
    tournament_size: int = 5
    episodes_per_genome: int = 4
    evaluation_episodes: int = 12
    max_steps: int = 12000
    novelty_weight: float = 0.15
    complexity_penalty: float = 0.0008
    checkpoint_interval: int = 5
    target_score: int = 200
    curriculum: bool = True
    replay_best: bool = True


@dataclass(frozen=True)
class Config:
    run: RunConfig = RunConfig()
    environment: EnvConfig = EnvConfig()
    trainer: TrainerConfig = TrainerConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataclass_from_mapping(cls: type[T], data: Mapping[str, Any] | None) -> T:
    values = dict(data or {})
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {', '.join(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return Config(
        run=_dataclass_from_mapping(RunConfig, raw.get("run")),
        environment=_dataclass_from_mapping(EnvConfig, raw.get("environment")),
        trainer=_dataclass_from_mapping(TrainerConfig, raw.get("trainer")),
    )


def config_from_mapping(raw: Mapping[str, Any]) -> Config:
    return Config(
        run=_dataclass_from_mapping(RunConfig, raw.get("run") if raw else None),
        environment=_dataclass_from_mapping(EnvConfig, raw.get("environment") if raw else None),
        trainer=_dataclass_from_mapping(TrainerConfig, raw.get("trainer") if raw else None),
    )


def save_config(config: Config, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def override_config(
    config: Config,
    *,
    seed: int | None = None,
    generations: int | None = None,
    population_size: int | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    max_steps: int | None = None,
) -> Config:
    run = config.run
    trainer = config.trainer
    if seed is not None:
        run = replace(run, seed=seed)
    if output_dir is not None:
        run = replace(run, output_dir=output_dir)
    if run_name is not None:
        run = replace(run, name=run_name)
    if generations is not None:
        trainer = replace(trainer, generations=generations)
    if population_size is not None:
        trainer = replace(trainer, population_size=population_size)
    if max_steps is not None:
        trainer = replace(trainer, max_steps=max_steps)
    return replace(config, run=run, trainer=trainer)


def env_with(config: EnvConfig, **updates: Any) -> EnvConfig:
    return replace(config, **updates)
