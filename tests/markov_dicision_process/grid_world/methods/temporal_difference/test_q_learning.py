import numpy as np
import pytest

from reinforcement_learning.errors import NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.agent_episodes import (
    run_td_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.q_learning import (
    QLearningAgent,
)


def test_update_with_empty_memory() -> None:
    agent = QLearningAgent(seed=0)
    with pytest.raises(NotInitializedError):
        agent.update()


def test_q_learning() -> None:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = QLearningAgent(seed=0)
    for _ in range(2):
        run_td_episode(env=env, agent=agent)

    assert agent.action_value == {
        ((2, 0), Action.UP): 0.0,
        ((2, 0), Action.DOWN): 0.0,
        ((2, 0), Action.LEFT): 0.0,
        ((2, 0), Action.RIGHT): 0.0,
        ((1, 0), Action.UP): 0.0,
        ((1, 0), Action.DOWN): 0.0,
        ((1, 0), Action.LEFT): 0.0,
        ((1, 0), Action.RIGHT): 0.0,
        ((0, 0), Action.UP): 0.0,
        ((0, 0), Action.DOWN): 0.0,
        ((0, 0), Action.LEFT): 0.0,
        ((0, 0), Action.RIGHT): 0.0,
        ((0, 1), Action.UP): 0.0,
        ((0, 1), Action.DOWN): 0.0,
        ((0, 1), Action.LEFT): 0.0,
        ((0, 1), Action.RIGHT): 0.5760000000000001,
        ((0, 2), Action.UP): 0.0,
        ((0, 2), Action.DOWN): 0.0,
        ((0, 2), Action.LEFT): 0.0,
        ((0, 2), Action.RIGHT): 0.96,
        ((1, 2), Action.UP): 0.0,
        ((1, 2), Action.DOWN): 0.0,
        ((1, 2), Action.LEFT): 0.0,
        ((1, 2), Action.RIGHT): 0.0,
    }
