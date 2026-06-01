from __future__ import annotations

import json
from pathlib import Path
import shutil
from time import strftime
from typing import Any

from flappylearn.config import Config, save_config


class ExperimentTracker:
    def __init__(self, config: Config):
        root = Path(config.run.output_dir) / config.run.name
        self.run_root = root
        self.run_id = strftime("%Y%m%d-%H%M%S")
        self.run_dir = root / self.run_id
        self.latest_dir = root / "latest"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.latest_dir.exists() or self.latest_dir.is_symlink():
            if self.latest_dir.is_dir() and not self.latest_dir.is_symlink():
                shutil.rmtree(self.latest_dir)
            else:
                self.latest_dir.unlink()
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        (self.latest_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        save_config(config, self.run_dir / "config.json")
        save_config(config, self.latest_dir / "config.json")

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def latest_metrics_path(self) -> Path:
        return self.latest_dir / "metrics.jsonl"

    def log_metrics(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.metrics_path, record)
        self._append_jsonl(self.latest_metrics_path, record)

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "summary.json", summary)
        self._write_json(self.latest_dir / "summary.json", summary)

    def write_artifact_json(self, name: str, payload: dict[str, Any]) -> None:
        self._write_json(self.run_dir / name, payload)
        self._write_json(self.latest_dir / name, payload)

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_ready(record), sort_keys=True))
            handle.write("\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")


def _json_ready(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
