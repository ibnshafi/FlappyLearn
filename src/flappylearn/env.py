from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from flappylearn.config import EnvConfig


OBSERVATION_SIZE = 10


@dataclass
class Pipe:
    x: float
    gap_y: float
    scored: bool = False

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class FlappyEnv:
    """Deterministic, headless Flappy Bird simulator."""

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.bird_y = self.config.height * 0.5
        self.bird_velocity = 0.0
        self.pipes: list[Pipe] = []
        self.score = 0
        self.steps = 0
        self.done = False
        self.last_action = 0
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_value = seed
        self.bird_y = self.config.height * 0.5
        self.bird_velocity = 0.0
        self.score = 0
        self.steps = 0
        self.done = False
        self.last_action = 0
        self.pipes = []
        x = self.config.spawn_x
        while x < self.config.width + self.config.pipe_spacing * 4:
            self.pipes.append(Pipe(x=x, gap_y=self._sample_gap_y()))
            x += self.config.pipe_spacing
        return self.observe()

    def _sample_gap_y(self) -> float:
        margin = max(self.config.pipe_gap * 0.55, self.config.bird_radius * 3)
        low = margin
        high = self.config.height - margin
        return float(self.rng.uniform(low, high))

    def _ensure_pipes(self) -> None:
        while self.pipes and self.pipes[0].x + self.config.pipe_width < -1:
            self.pipes.pop(0)
        while not self.pipes or self.pipes[-1].x < self.config.width + self.config.pipe_spacing * 3:
            last_x = self.pipes[-1].x if self.pipes else self.config.spawn_x
            self.pipes.append(Pipe(x=last_x + self.config.pipe_spacing, gap_y=self._sample_gap_y()))

    def next_pipe(self) -> Pipe:
        for pipe in self.pipes:
            if pipe.x + self.config.pipe_width >= self.config.bird_x - self.config.bird_radius:
                return pipe
        return self.pipes[0]

    def observe(self) -> np.ndarray:
        pipe = self.next_pipe()
        next_index = min(self.pipes.index(pipe) + 1, len(self.pipes) - 1)
        pipe_2 = self.pipes[next_index]

        half_h = self.config.height * 0.5
        dx = (pipe.x + self.config.pipe_width - self.config.bird_x) / self.config.width
        gap_center = (pipe.gap_y - half_h) / half_h
        gap_top = pipe.gap_y - self.config.pipe_gap * 0.5
        gap_bottom = pipe.gap_y + self.config.pipe_gap * 0.5
        y_norm = (self.bird_y - half_h) / half_h
        velocity_norm = self.bird_velocity / self.config.max_velocity

        obs = np.array(
            [
                y_norm,
                velocity_norm,
                dx,
                gap_center,
                (self.bird_y - pipe.gap_y) / half_h,
                (self.bird_y - gap_top) / half_h,
                (gap_bottom - self.bird_y) / half_h,
                (pipe_2.x + self.config.pipe_width - self.config.bird_x) / self.config.width,
                (pipe_2.gap_y - half_h) / half_h,
                float(self.last_action),
            ],
            dtype=np.float64,
        )
        return np.clip(obs, -5.0, 5.0)

    def step(self, action: int | bool) -> StepResult:
        if self.done:
            raise RuntimeError("Cannot step an environment after it is done. Call reset().")

        flap = 1 if bool(action) else 0
        self.last_action = flap
        if flap:
            self.bird_velocity = self.config.flap_velocity

        self.bird_velocity = min(
            self.config.max_velocity,
            self.bird_velocity + self.config.gravity,
        )
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
            pipe.x -= self.config.scroll_speed

        reward = self.config.alive_reward
        passed = 0
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + self.config.pipe_width < self.config.bird_x:
                pipe.scored = True
                self.score += 1
                passed += 1
        if passed:
            reward += self.config.pass_reward * passed

        self.steps += 1
        self._ensure_pipes()
        self.done = self._collides()
        if self.done:
            reward += self.config.death_penalty

        observation = self.observe()
        return StepResult(
            observation=observation,
            reward=float(reward),
            done=self.done,
            info={
                "score": self.score,
                "steps": self.steps,
                "passed": passed,
                "bird_y": self.bird_y,
                "bird_velocity": self.bird_velocity,
            },
        )

    def _collides(self) -> bool:
        if self.bird_y - self.config.bird_radius <= 0:
            return True
        if self.bird_y + self.config.bird_radius >= self.config.height:
            return True
        for pipe in self.pipes:
            overlaps_x = (
                self.config.bird_x + self.config.bird_radius >= pipe.x
                and self.config.bird_x - self.config.bird_radius <= pipe.x + self.config.pipe_width
            )
            if not overlaps_x:
                continue
            gap_top = pipe.gap_y - self.config.pipe_gap * 0.5
            gap_bottom = pipe.gap_y + self.config.pipe_gap * 0.5
            outside_gap = (
                self.bird_y - self.config.bird_radius < gap_top
                or self.bird_y + self.config.bird_radius > gap_bottom
            )
            if outside_gap:
                return True
        return False

    def snapshot(self, action: int | bool = 0, reward: float = 0.0) -> dict[str, Any]:
        return {
            "step": self.steps,
            "score": self.score,
            "done": self.done,
            "bird": {
                "x": self.config.bird_x,
                "y": self.bird_y,
                "radius": self.config.bird_radius,
                "velocity": self.bird_velocity,
            },
            "pipes": [pipe.to_dict() for pipe in self.pipes[:5]],
            "action": int(bool(action)),
            "reward": float(reward),
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "pipe_width": self.config.pipe_width,
                "pipe_gap": self.config.pipe_gap,
            },
        }
