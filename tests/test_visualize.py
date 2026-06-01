import json
from pathlib import Path

from flappylearn.visualize import write_metrics_html, write_replay_html


def test_visualizers_write_html(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"generation": 0, "eval_score_mean": 1, "population_score_mean": 0.5}) + "\n",
        encoding="utf-8",
    )
    metrics_html = tmp_path / "metrics.html"
    write_metrics_html(metrics, metrics_html)
    assert "FlappyLearn Metrics" in metrics_html.read_text(encoding="utf-8")

    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "score": 0,
                "steps": 1,
                "frames": [
                    {
                        "step": 0,
                        "score": 0,
                        "bird": {"x": 64, "y": 256, "radius": 12},
                        "pipes": [],
                        "action": 0,
                        "config": {"pipe_width": 52, "pipe_gap": 110},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    replay_html = tmp_path / "replay.html"
    write_replay_html(replay, replay_html)
    assert "FlappyLearn Replay" in replay_html.read_text(encoding="utf-8")
