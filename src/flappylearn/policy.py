from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from flappylearn.genome import AdaptiveCircuitGenome


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: np.ndarray) -> int: ...


@dataclass
class CircuitPolicy:
    genome: AdaptiveCircuitGenome
    deterministic: bool = True

    def __post_init__(self) -> None:
        self.memory = np.zeros(self.genome.hidden_size, dtype=np.float64)

    def reset(self) -> None:
        self.memory = np.zeros(self.genome.hidden_size, dtype=np.float64)

    def act(self, observation: np.ndarray) -> int:
        logit, self.memory = self.genome.forward(observation, self.memory)
        if self.deterministic or self.genome.exploration_noise <= 0:
            return int(logit > self.genome.threshold)
        noise = self.genome.rng.normal(0.0, self.genome.exploration_noise)
        return int(logit + noise > self.genome.threshold)


class RandomPolicy:
    def __init__(self, probability: float = 0.1, seed: int | None = None):
        self.probability = probability
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def act(self, observation: np.ndarray) -> int:
        return int(self.rng.random() < self.probability)


class NoopPolicy:
    def reset(self) -> None:
        return None

    def act(self, observation: np.ndarray) -> int:
        return 0
