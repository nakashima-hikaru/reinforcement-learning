import numpy as np
import pytest

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import (
    run_monte_carlo_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_eval import RandomAgent


def test_mc_eval() -> None:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = RandomAgent(gamma=0.9, seed=0)
    for _ in range(2):
        run_monte_carlo_episode(env=env, agent=agent)

    assert agent.state_value == {
        (0, 2): pytest.approx(0.22042382796141025),
        (0, 1): pytest.approx(0.05544759280164322),
        (0, 0): pytest.approx(0.11230028389108984),
        (1, 0): pytest.approx(0.018681242818946488),
        (2, 0): pytest.approx(0.07457318402670388),
        (1, 3): pytest.approx(0.44999999999999996),
        (1, 2): pytest.approx(-0.5866904341957879),
        (2, 2): pytest.approx(-0.38389590313967503),
        (2, 1): pytest.approx(-0.1588166194308249),
        (2, 3): pytest.approx(-0.47275257189285014),
    }
