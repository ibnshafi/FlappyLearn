from dataclasses import replace
from pathlib import Path

from flappylearn.config import load_config, override_config
from flappylearn.learner import DiscoveryTrainer


def test_trainer_smoke_run_writes_artifacts(tmp_path):
    config = load_config(Path("configs/smoke.json"))
    config = override_config(
        config,
        output_dir=str(tmp_path),
        run_name="test-smoke",
        generations=2,
        population_size=8,
        max_steps=500,
    )
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            elites=2,
            tournament_size=2,
            episodes_per_genome=1,
            evaluation_episodes=2,
            checkpoint_interval=1,
            hidden_units=4,
            max_hidden_units=8,
        ),
    )

    summary = DiscoveryTrainer(config).train()
    latest = Path(summary["latest_dir"])

    assert (latest / "metrics.jsonl").exists()
    assert (latest / "best.json").exists()
    assert (latest / "metrics.html").exists()
    assert summary["best_score_mean"] >= 0
