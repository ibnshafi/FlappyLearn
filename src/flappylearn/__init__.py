"""Autonomous Flappy Bird learning system."""

from flappylearn.config import Config, EnvConfig, RunConfig, TrainerConfig
from flappylearn.env import FlappyEnv
from flappylearn.genome import AdaptiveCircuitGenome
from flappylearn.learner import DiscoveryTrainer

__all__ = [
    "AdaptiveCircuitGenome",
    "Config",
    "DiscoveryTrainer",
    "EnvConfig",
    "FlappyEnv",
    "RunConfig",
    "TrainerConfig",
]

__version__ = "0.1.0"
