import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.agent_episodes import (
    run_td_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.td_eval import TdAgent


def test_evaluation() -> None:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = TdAgent(gamma=0.9, alpha=0.01, seed=41)
    n_episodes: int = 2
    for _ in range(n_episodes):
        run_td_episode(env=env, agent=agent)

    assert agent.v == {
        (2, 1): -8.100000000000001e-07,
        (2, 0): 0.0,
        (2, 2): -8.901090000000001e-05,
        (1, 2): -0.0199891,
        (1, 3): 0.01019701,
        (2, 3): -0.009998209000000001,
    }
