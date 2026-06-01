from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flappylearn.genome import AdaptiveCircuitGenome


def save_best_genome(
    genome: AdaptiveCircuitGenome,
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "flappylearn.best_genome",
        "metadata": metadata,
        "genome": genome.to_dict(),
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_genome(path: str | Path) -> tuple[AdaptiveCircuitGenome, dict[str, Any]]:
    checkpoint = Path(path)
    with checkpoint.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "genome" not in payload:
        raise ValueError(f"{checkpoint} does not contain a genome")
    return AdaptiveCircuitGenome.from_dict(payload["genome"]), dict(payload.get("metadata", {}))


def save_population_checkpoint(
    path: str | Path,
    *,
    generation: int,
    population: list[AdaptiveCircuitGenome],
    best: AdaptiveCircuitGenome,
    metadata: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "flappylearn.population_checkpoint",
        "generation": generation,
        "metadata": metadata,
        "best_genome_id": best.genome_id,
        "population": [genome.to_dict() for genome in population],
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_population_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    with checkpoint.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("kind") != "flappylearn.population_checkpoint":
        raise ValueError(f"{checkpoint} is not a population checkpoint")
    payload["population"] = [AdaptiveCircuitGenome.from_dict(raw) for raw in payload.get("population", [])]
    return payload
