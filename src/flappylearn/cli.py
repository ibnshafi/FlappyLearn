from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flappylearn.benchmark import benchmark_checkpoint, write_benchmark
from flappylearn.checkpoint import load_genome, load_population_checkpoint
from flappylearn.config import (
    Config,
    config_from_mapping,
    load_config,
    override_config,
)
from flappylearn.evaluation import evaluate_genome, record_genome_replay
from flappylearn.learner import DiscoveryTrainer
from flappylearn.profiling import profile_training
from flappylearn.visualize import write_metrics_html, write_replay_html


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flappylearn")
    sub = parser.add_subparsers(required=True)

    train = sub.add_parser("train", help="train an autonomous learner")
    _add_config_args(train)
    train.add_argument("--resume", type=Path, help="resume from a population checkpoint")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("eval", help="evaluate a saved best genome")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    _add_config_args(evaluate, include_run=False, include_seed=False, include_max_steps=False)
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed", dest="eval_seed", type=int, default=123)
    evaluate.add_argument("--max-steps", type=int)
    evaluate.set_defaults(func=cmd_eval)

    replay = sub.add_parser("replay", help="write replay JSON and HTML for a checkpoint")
    replay.add_argument("--checkpoint", type=Path, required=True)
    _add_config_args(replay, include_run=False, include_seed=False, include_max_steps=False)
    replay.add_argument("--seed", dest="eval_seed", type=int, default=123)
    replay.add_argument("--max-steps", type=int)
    replay.add_argument("--json-output", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.set_defaults(func=cmd_replay)

    bench = sub.add_parser("benchmark", help="benchmark a checkpoint against baselines")
    bench.add_argument("--checkpoint", type=Path, required=True)
    _add_config_args(bench, include_run=False, include_seed=False, include_max_steps=False)
    bench.add_argument("--episodes", type=int, default=100)
    bench.add_argument("--seed", dest="eval_seed", type=int, default=123)
    bench.add_argument("--max-steps", type=int)
    bench.add_argument("--output", type=Path)
    bench.set_defaults(func=cmd_benchmark)

    viz = sub.add_parser("visualize", help="render metrics JSONL to an HTML chart")
    viz.add_argument("--metrics", type=Path, required=True)
    viz.add_argument("--output", type=Path, required=True)
    viz.set_defaults(func=cmd_visualize)

    profile = sub.add_parser("profile", help="profile a short training run")
    _add_config_args(profile)
    profile.add_argument("--profile-generations", type=int, default=3)
    profile.add_argument("--output", type=Path, default=Path("profiles/flappylearn_profile.txt"))
    profile.set_defaults(func=cmd_profile)
    return parser


def _add_config_args(
    parser: argparse.ArgumentParser,
    *,
    include_run: bool = True,
    include_seed: bool = True,
    include_max_steps: bool = True,
) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    if include_seed:
        parser.add_argument("--seed", type=int)
    if include_max_steps:
        parser.add_argument("--max-steps", type=int)
    if include_run:
        parser.add_argument("--generations", type=int)
        parser.add_argument("--population", type=int)
        parser.add_argument("--output-dir", type=str)
        parser.add_argument("--run-name", type=str)


def _load_with_overrides(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    return override_config(
        config,
        seed=getattr(args, "seed", None),
        generations=getattr(args, "generations", None),
        population_size=getattr(args, "population", None),
        output_dir=getattr(args, "output_dir", None),
        run_name=getattr(args, "run_name", None),
        max_steps=getattr(args, "max_steps", None),
    )


def cmd_train(args: argparse.Namespace) -> int:
    if args.resume:
        checkpoint = load_population_checkpoint(args.resume)
        metadata = checkpoint.get("metadata", {})
        raw_config = metadata.get("config")
        config = config_from_mapping(raw_config) if raw_config else _load_with_overrides(args)
        config = override_config(
            config,
            seed=getattr(args, "seed", None),
            generations=getattr(args, "generations", None),
            population_size=getattr(args, "population", None),
            output_dir=getattr(args, "output_dir", None),
            run_name=getattr(args, "run_name", None),
            max_steps=getattr(args, "max_steps", None),
        )
        trainer = DiscoveryTrainer(
            config,
            initial_population=checkpoint["population"],
            start_generation=int(checkpoint["generation"]) + 1,
        )
    else:
        trainer = DiscoveryTrainer(_load_with_overrides(args))
    summary = trainer.train()
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    config = _load_with_overrides(args)
    genome, metadata = load_genome(args.checkpoint)
    max_steps = args.max_steps or int(metadata.get("config", {}).get("trainer", {}).get("max_steps", config.trainer.max_steps))
    seeds = [int(args.eval_seed + index * 9973) for index in range(args.episodes)]
    result = evaluate_genome(genome, config.environment, seeds, max_steps)
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    config = _load_with_overrides(args)
    genome, metadata = load_genome(args.checkpoint)
    max_steps = args.max_steps or int(metadata.get("config", {}).get("trainer", {}).get("max_steps", config.trainer.max_steps))
    replay = record_genome_replay(genome, config.environment, args.eval_seed, max_steps)
    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_replay_html(json_output, args.output)
    print(json.dumps({"json": str(json_output), "html": str(args.output), "score": replay["score"]}, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = _load_with_overrides(args)
    result = benchmark_checkpoint(
        args.checkpoint,
        config.environment,
        episodes=args.episodes,
        seed=args.eval_seed,
        max_steps=args.max_steps or config.trainer.max_steps,
    )
    if args.output:
        write_benchmark(result, args.output)
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    write_metrics_html(args.metrics, args.output)
    print(json.dumps({"html": str(args.output)}, indent=2))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    output = profile_training(_load_with_overrides(args), args.output, generations=args.profile_generations)
    print(json.dumps({"profile": str(output)}, indent=2))
    return 0


def _json_ready(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
