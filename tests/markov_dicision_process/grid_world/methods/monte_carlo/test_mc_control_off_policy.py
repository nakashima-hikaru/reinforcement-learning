import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import Action, GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import (
    run_monte_carlo_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_control_off_policy import (
    McOffPolicyAgent,
)


def test_mc_control_off_policy() -> None:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = McOffPolicyAgent(gamma=0.9, epsilon=0.1, alpha=0.1, seed=0)
    for _ in range(2):
        run_monte_carlo_episode(env=env, agent=agent)

    assert agent.action_value == {
        ((0, 2), Action.RIGHT): 0.19,
        ((0, 2), Action.UP): 0.09000000000000001,
        ((0, 2), Action.DOWN): 0.0,
        ((0, 2), Action.LEFT): 0.0,
        ((0, 1), Action.RIGHT): 0.09729729729729729,
        ((0, 1), Action.UP): 0.0,
        ((0, 1), Action.DOWN): 0.0,
        ((0, 1), Action.LEFT): 0.0,
        ((0, 0), Action.RIGHT): 0.0,
        ((0, 0), Action.UP): 0.0,
        ((0, 0), Action.DOWN): 0.0,
        ((0, 0), Action.LEFT): 0.0,
        ((1, 0), Action.UP): 0.0,
        ((1, 0), Action.DOWN): 0.0,
        ((1, 0), Action.LEFT): 0.0,
        ((1, 0), Action.RIGHT): 0.0,
        ((2, 0), Action.UP): 0.0,
        ((2, 0), Action.DOWN): 0.0,
        ((2, 0), Action.LEFT): 0.0,
        ((2, 0), Action.RIGHT): 0.0,
    }

    assert agent.evaluation_policy == {
        (0, 2): {Action.UP: 0.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 1.0},
        (0, 1): {Action.UP: 0.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 1.0},
        (0, 0): {Action.UP: 1.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0},
        (1, 0): {Action.UP: 1.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0},
        (2, 0): {Action.UP: 1.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0},
    }
