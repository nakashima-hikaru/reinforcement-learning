import numpy as np

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
    agent = RandomAgent(gamma=0.9)
    for _ in range(2):
        run_monte_carlo_episode(env=env, agent=agent)

    assert agent.v == {
        (0, 2): 0.22042382796141025,
        (0, 1): 0.05544759280164322,
        (0, 0): 0.11230028389108984,
        (1, 0): 0.018681242818946488,
        (2, 0): 0.07457318402670388,
        (1, 3): 0.44999999999999996,
        (1, 2): -0.5866904341957879,
        (2, 2): -0.38389590313967503,
        (2, 1): -0.1588166194308249,
        (2, 3): -0.47275257189285014,
    }
