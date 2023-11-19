import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import Action, GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import (
    run_monte_carlo_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_control_on_policy import (
    McOnPolicyAgent,
)


def test_mc_control_on_policy() -> None:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = McOnPolicyAgent(gamma=0.9, epsilon=0.1, alpha=0.1, seed=0)
    for _ in range(2):
        run_monte_carlo_episode(env=env, agent=agent)

    assert agent.q == {
        ((0, 2), Action.RIGHT): 0.19,
        ((0, 2), Action.UP): 0.09000000000000001,
        ((0, 2), Action.DOWN): 0.0,
        ((0, 2), Action.LEFT): 0.05314410000000002,
        ((0, 1), Action.RIGHT): 0.19865672100000004,
        ((0, 1), Action.UP): 0.0,
        ((0, 1), Action.DOWN): 0.0,
        ((0, 1), Action.LEFT): 0.05904900000000002,
        ((0, 0), Action.RIGHT): 0.17879104890000003,
        ((0, 0), Action.UP): 0.0,
        ((0, 0), Action.DOWN): 0.0,
        ((0, 0), Action.LEFT): 0.06561000000000002,
        ((1, 0), Action.UP): 0.10776784401000003,
        ((1, 0), Action.DOWN): 0.0,
        ((1, 0), Action.LEFT): 0.0,
        ((1, 0), Action.RIGHT): 0.0,
        ((2, 0), Action.UP): 0.09699105960900004,
        ((2, 0), Action.DOWN): 0.03138105960900002,
        ((2, 0), Action.LEFT): 0.02824295364810002,
        ((2, 0), Action.RIGHT): 0.0,
    }
