from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ACTIVATIONS = ("tanh", "relu", "elu")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _activation(name: str, values: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(values)
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "sin":
        return np.sin(values)
    if name == "elu":
        return np.where(values > 0.0, values, np.expm1(values))
    if name == "gauss":
        clipped = np.clip(values, -6.0, 6.0)
        return np.exp(-(clipped * clipped))
    if name == "identity":
        return values
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class MutationParams:
    weight_sigma: float = 0.10
    weight_mutation_rate: float = 0.18
    bias_sigma: float = 0.12
    topology_rate: float = 0.02
    activation_rate: float = 0.05
    threshold_sigma: float = 0.05

    def adapted(self, rng: np.random.Generator) -> MutationParams:
        def positive(value: float, spread: float, low: float, high: float) -> float:
            return float(np.clip(value * math.exp(rng.normal(0.0, spread)), low, high))

        return MutationParams(
            weight_sigma=positive(self.weight_sigma, 0.12, 0.01, 1.25),
            weight_mutation_rate=positive(self.weight_mutation_rate, 0.10, 0.01, 0.85),
            bias_sigma=positive(self.bias_sigma, 0.12, 0.005, 1.0),
            topology_rate=positive(self.topology_rate, 0.18, 0.001, 0.35),
            activation_rate=positive(self.activation_rate, 0.16, 0.001, 0.45),
            threshold_sigma=positive(self.threshold_sigma, 0.10, 0.002, 0.5),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "weight_sigma": self.weight_sigma,
            "weight_mutation_rate": self.weight_mutation_rate,
            "bias_sigma": self.bias_sigma,
            "topology_rate": self.topology_rate,
            "activation_rate": self.activation_rate,
            "threshold_sigma": self.threshold_sigma,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MutationParams:
        return cls(**{key: float(value) for key, value in data.items()})


@dataclass
class AdaptiveCircuitGenome:
    input_size: int
    w_in: np.ndarray
    w_rec: np.ndarray
    b_hidden: np.ndarray
    w_out: np.ndarray
    w_direct: np.ndarray
    b_out: float
    activations: list[str]
    threshold: float = 0.0
    exploration_noise: float = 0.0
    mutation: MutationParams = field(default_factory=MutationParams)
    genome_id: str = field(default_factory=_new_id)
    parent_ids: tuple[str, ...] = ()
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    @property
    def hidden_size(self) -> int:
        return int(self.w_out.shape[0])

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        input_size: int,
        hidden_units: int,
    ) -> AdaptiveCircuitGenome:
        scale = 1.0 / math.sqrt(max(1, input_size))
        activations = [str(rng.choice(ACTIVATIONS)) for _ in range(hidden_units)]
        return cls(
            input_size=input_size,
            w_in=rng.normal(0.0, scale, size=(hidden_units, input_size)),
            w_rec=rng.normal(0.0, 0.08, size=(hidden_units, hidden_units)),
            b_hidden=rng.normal(0.0, 0.1, size=hidden_units),
            w_out=rng.normal(0.0, scale, size=hidden_units),
            w_direct=rng.normal(0.0, scale, size=input_size),
            b_out=float(rng.normal(0.0, 0.1)),
            activations=activations,
            threshold=float(rng.normal(0.0, 0.15)),
            exploration_noise=float(np.clip(rng.lognormal(-3.0, 0.4), 0.0, 0.25)),
            mutation=MutationParams(),
            rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
        )

    def forward(self, observation: np.ndarray, memory: np.ndarray) -> tuple[float, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (self.input_size,):
            raise ValueError(f"Expected observation shape {(self.input_size,)}, got {obs.shape}")
        if memory.shape != (self.hidden_size,):
            memory = np.zeros(self.hidden_size, dtype=np.float64)

        pre = self.w_in @ obs + self.w_rec @ memory + self.b_hidden
        hidden = np.empty_like(pre)
        for activation_name in set(self.activations):
            indices = [i for i, name in enumerate(self.activations) if name == activation_name]
            hidden[indices] = _activation(activation_name, pre[indices])
        new_memory = np.tanh(0.62 * memory + 0.38 * hidden)
        logit = float(np.dot(self.w_out, new_memory) + np.dot(self.w_direct, obs) + self.b_out)
        return logit, new_memory

    def copy(self) -> AdaptiveCircuitGenome:
        return AdaptiveCircuitGenome(
            input_size=self.input_size,
            w_in=self.w_in.copy(),
            w_rec=self.w_rec.copy(),
            b_hidden=self.b_hidden.copy(),
            w_out=self.w_out.copy(),
            w_direct=self.w_direct.copy(),
            b_out=float(self.b_out),
            activations=list(self.activations),
            threshold=float(self.threshold),
            exploration_noise=float(self.exploration_noise),
            mutation=MutationParams.from_dict(self.mutation.to_dict()),
            genome_id=self.genome_id,
            parent_ids=tuple(self.parent_ids),
            rng=np.random.default_rng(),
        )

    def mutate(self, rng: np.random.Generator, max_hidden_units: int) -> AdaptiveCircuitGenome:
        child = self.copy()
        child.genome_id = _new_id()
        child.parent_ids = (self.genome_id,)
        child.rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        child.mutation = child.mutation.adapted(rng)

        child._mutate_array(rng, child.w_in, child.mutation.weight_sigma, child.mutation.weight_mutation_rate)
        child._mutate_array(rng, child.w_rec, child.mutation.weight_sigma * 0.6, child.mutation.weight_mutation_rate)
        child._mutate_array(rng, child.w_out, child.mutation.weight_sigma, child.mutation.weight_mutation_rate)
        child._mutate_array(rng, child.w_direct, child.mutation.weight_sigma, child.mutation.weight_mutation_rate)
        child._mutate_array(rng, child.b_hidden, child.mutation.bias_sigma, child.mutation.weight_mutation_rate)
        child.b_out += float(rng.normal(0.0, child.mutation.bias_sigma))
        child.threshold += float(rng.normal(0.0, child.mutation.threshold_sigma))
        child.exploration_noise = float(
            np.clip(
                child.exploration_noise * math.exp(rng.normal(0.0, 0.14)),
                0.0,
                0.35,
            )
        )

        if child.hidden_size < max_hidden_units and rng.random() < child.mutation.topology_rate:
            child._add_unit(rng)
        if child.hidden_size > 1 and rng.random() < child.mutation.topology_rate * 0.45:
            child._remove_unit(int(rng.integers(0, child.hidden_size)))
        for index in range(child.hidden_size):
            if rng.random() < child.mutation.activation_rate:
                child.activations[index] = str(rng.choice(ACTIVATIONS))
        return child

    @staticmethod
    def _mutate_array(
        rng: np.random.Generator,
        array: np.ndarray,
        sigma: float,
        mutation_rate: float,
    ) -> None:
        mask = rng.random(array.shape) < mutation_rate
        if np.any(mask):
            array += mask * rng.normal(0.0, sigma, size=array.shape)
        reset_mask = rng.random(array.shape) < (mutation_rate * 0.015)
        if np.any(reset_mask):
            array[reset_mask] = rng.normal(0.0, sigma * 2.0, size=int(np.sum(reset_mask)))
        np.clip(array, -8.0, 8.0, out=array)

    def _add_unit(self, rng: np.random.Generator) -> None:
        scale = 1.0 / math.sqrt(max(1, self.input_size))
        self.w_in = np.vstack([self.w_in, rng.normal(0.0, scale, size=(1, self.input_size))])
        old = self.hidden_size
        expanded = np.zeros((old + 1, old + 1), dtype=np.float64)
        expanded[:old, :old] = self.w_rec
        expanded[old, :old] = rng.normal(0.0, 0.06, size=old)
        expanded[:old, old] = rng.normal(0.0, 0.06, size=old)
        expanded[old, old] = rng.normal(0.0, 0.06)
        self.w_rec = expanded
        self.b_hidden = np.append(self.b_hidden, rng.normal(0.0, 0.1))
        self.w_out = np.append(self.w_out, rng.normal(0.0, scale))
        self.activations.append(str(rng.choice(ACTIVATIONS)))

    def _remove_unit(self, index: int) -> None:
        self.w_in = np.delete(self.w_in, index, axis=0)
        self.w_rec = np.delete(np.delete(self.w_rec, index, axis=0), index, axis=1)
        self.b_hidden = np.delete(self.b_hidden, index)
        self.w_out = np.delete(self.w_out, index)
        del self.activations[index]

    @classmethod
    def crossover(
        cls,
        rng: np.random.Generator,
        parent_a: AdaptiveCircuitGenome,
        parent_b: AdaptiveCircuitGenome,
    ) -> AdaptiveCircuitGenome:
        if parent_b.hidden_size > parent_a.hidden_size:
            parent_a, parent_b = parent_b, parent_a
        child = parent_a.copy()
        child.genome_id = _new_id()
        child.parent_ids = (parent_a.genome_id, parent_b.genome_id)
        shared = min(parent_a.hidden_size, parent_b.hidden_size)

        row_indices = np.flatnonzero(rng.random(shared) < 0.5)
        if row_indices.size:
            child.w_in[row_indices] = parent_b.w_in[row_indices]
            child.w_out[row_indices] = parent_b.w_out[row_indices]
            child.b_hidden[row_indices] = parent_b.b_hidden[row_indices]
        row_set = set(int(index) for index in row_indices)
        for index in range(shared):
            if index in row_set:
                child.activations[index] = parent_b.activations[index]

        rec_mask = rng.random((shared, shared)) < 0.5
        child.w_rec[:shared, :shared][rec_mask] = parent_b.w_rec[:shared, :shared][rec_mask]
        direct_mask = rng.random(child.input_size) < 0.5
        child.w_direct[direct_mask] = parent_b.w_direct[direct_mask]
        if rng.random() < 0.5:
            child.b_out = parent_b.b_out
        if rng.random() < 0.5:
            child.threshold = parent_b.threshold
        child.exploration_noise = float((parent_a.exploration_noise + parent_b.exploration_noise) * 0.5)
        source_mutation = parent_a.mutation if rng.random() < 0.5 else parent_b.mutation
        child.mutation = MutationParams.from_dict(source_mutation.to_dict())
        return child

    def complexity(self) -> float:
        active = (
            np.count_nonzero(np.abs(self.w_in) > 1e-6)
            + np.count_nonzero(np.abs(self.w_rec) > 1e-6)
            + np.count_nonzero(np.abs(self.w_out) > 1e-6)
            + np.count_nonzero(np.abs(self.w_direct) > 1e-6)
        )
        return float(self.hidden_size * 2 + active * 0.05)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "w_in": self.w_in.tolist(),
            "w_rec": self.w_rec.tolist(),
            "b_hidden": self.b_hidden.tolist(),
            "w_out": self.w_out.tolist(),
            "w_direct": self.w_direct.tolist(),
            "b_out": self.b_out,
            "activations": self.activations,
            "threshold": self.threshold,
            "exploration_noise": self.exploration_noise,
            "mutation": self.mutation.to_dict(),
            "genome_id": self.genome_id,
            "parent_ids": list(self.parent_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptiveCircuitGenome:
        return cls(
            input_size=int(data["input_size"]),
            w_in=np.asarray(data["w_in"], dtype=np.float64),
            w_rec=np.asarray(data["w_rec"], dtype=np.float64),
            b_hidden=np.asarray(data["b_hidden"], dtype=np.float64),
            w_out=np.asarray(data["w_out"], dtype=np.float64),
            w_direct=np.asarray(data["w_direct"], dtype=np.float64),
            b_out=float(data["b_out"]),
            activations=list(data["activations"]),
            threshold=float(data.get("threshold", 0.0)),
            exploration_noise=float(data.get("exploration_noise", 0.0)),
            mutation=MutationParams.from_dict(data.get("mutation", MutationParams().to_dict())),
            genome_id=str(data.get("genome_id", _new_id())),
            parent_ids=tuple(data.get("parent_ids", ())),
        )
