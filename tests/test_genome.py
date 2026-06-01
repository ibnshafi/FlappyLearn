import numpy as np

from flappylearn.env import OBSERVATION_SIZE
from flappylearn.genome import AdaptiveCircuitGenome


def test_genome_roundtrip_preserves_forward_pass():
    rng = np.random.default_rng(5)
    genome = AdaptiveCircuitGenome.random(rng, OBSERVATION_SIZE, hidden_units=5)
    obs = rng.normal(size=OBSERVATION_SIZE)
    memory = np.zeros(genome.hidden_size)

    logit_a, mem_a = genome.forward(obs, memory)
    restored = AdaptiveCircuitGenome.from_dict(genome.to_dict())
    logit_b, mem_b = restored.forward(obs, memory)

    assert logit_a == logit_b
    np.testing.assert_allclose(mem_a, mem_b)


def test_mutation_respects_hidden_unit_limit():
    rng = np.random.default_rng(6)
    genome = AdaptiveCircuitGenome.random(rng, OBSERVATION_SIZE, hidden_units=4)
    for _ in range(30):
        genome = genome.mutate(rng, max_hidden_units=7)
        assert 1 <= genome.hidden_size <= 7
        assert len(genome.activations) == genome.hidden_size
