import numpy as np

from flappylearn.env import OBSERVATION_SIZE, FlappyEnv


def test_environment_is_deterministic_for_seed_and_actions():
    actions = [0, 1, 0, 0, 1, 0, 0, 0, 1]
    env_a = FlappyEnv(seed=123)
    env_b = FlappyEnv(seed=123)

    obs_a = env_a.reset(seed=123)
    obs_b = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    assert obs_a.shape == (OBSERVATION_SIZE,)

    for action in actions:
        result_a = env_a.step(action)
        result_b = env_b.step(action)
        np.testing.assert_allclose(result_a.observation, result_b.observation)
        assert result_a.reward == result_b.reward
        assert result_a.done == result_b.done
        assert result_a.info["score"] == result_b.info["score"]


def test_noop_policy_eventually_crashes():
    env = FlappyEnv(seed=9)
    env.reset(seed=9)
    done = False
    for _ in range(300):
        result = env.step(0)
        done = result.done
        if done:
            break
    assert done
    assert env.score == 0
