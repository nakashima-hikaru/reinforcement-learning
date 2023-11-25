import numpy as np
import pytest

from reinforcement_learning.errors import InvalidMemoryError
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, ActionResult, GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import (
    run_monte_carlo_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_control_off_policy import (
    McOffPolicyAgent,
)


def test_mc_add_memory() -> None:
    agent = McOffPolicyAgent(gamma=0.9, epsilon=0.1, alpha=0.1, seed=0)
    with pytest.raises(InvalidMemoryError):
        agent.add_memory(state=(0, 0), action=None, result=None)
    with pytest.raises(InvalidMemoryError):
        agent.add_memory(state=(0, 0), action=Action.UP, result=None)
    with pytest.raises(InvalidMemoryError):
        agent.add_memory(state=(0, 0), action=None, result=ActionResult(next_state=(0, 1), reward=1.0, done=False))


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
