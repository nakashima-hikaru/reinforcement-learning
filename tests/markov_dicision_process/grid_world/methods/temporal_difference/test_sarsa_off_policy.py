import numpy as np
import pytest

from reinforcement_learning.errors import InvalidMemoryError
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, ActionResult, GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.agent_episodes import (
    run_sarsa_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.sarsa_off_policy import (
    SarsaOffPolicyAgent,
)


def test_update_with_empty_memories() -> None:
    agent = SarsaOffPolicyAgent()
    agent.update()
    assert len(agent.memories) == 0


def test_update() -> None:
    agent = SarsaOffPolicyAgent()
    agent.add_memory(state=(0, 0), action=Action.UP, result=ActionResult(next_state=(0, 1), reward=1.0, done=False))
    agent.add_memory(state=(0, 1), action=Action.UP, result=ActionResult(next_state=(0, 2), reward=1.0, done=False))
    agent.update()
    assert agent.action_value == {
        ((0, 1), Action.UP): 0.0,
        ((0, 0), Action.UP): 0.8,
        ((0, 0), Action.DOWN): 0.0,
        ((0, 0), Action.LEFT): 0.0,
        ((0, 0), Action.RIGHT): 0.0,
    }


def test_update_with_first_empty_action() -> None:
    agent = SarsaOffPolicyAgent()
    agent.add_state_as_memory(state=(0, 0))
    agent.add_memory(state=(0, 1), action=Action.UP, result=ActionResult(next_state=(0, 2), reward=1.0, done=False))
    with pytest.raises(InvalidMemoryError):
        agent.update()


def test_update_with_second_empty_action_not_done() -> None:
    agent = SarsaOffPolicyAgent()
    agent.add_memory(state=(0, 0), action=Action.UP, result=ActionResult(next_state=(0, 1), reward=1.0, done=False))
    agent.add_state_as_memory(state=(0, 1))
    with pytest.raises(InvalidMemoryError):
        agent.update()


def test_update_with_second_empty_action_done() -> None:
    agent = SarsaOffPolicyAgent()
    agent.add_memory(state=(0, 0), action=Action.UP, result=ActionResult(next_state=(0, 1), reward=1.0, done=True))
    agent.add_state_as_memory(state=(0, 1))
    agent.update()


def test_update_with_first_empty_reward() -> None:
    agent = SarsaOffPolicyAgent()
    agent.add_state_as_memory(state=(0, 0))
    agent.add_memory(state=(0, 1), action=Action.UP, result=ActionResult(next_state=(0, 2), reward=1.0, done=False))
    with pytest.raises(InvalidMemoryError):
        agent.update()


def test_sarsa() -> None:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = SarsaOffPolicyAgent(seed=412)
    episodes = 2

    for _ in range(episodes):
        run_sarsa_episode(env=env, agent=agent)

    assert agent.behavior_policy == {
        (0, 0): {Action.UP: 0.925, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.025},
        (0, 1): {Action.UP: 0.925, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.025},
        (0, 2): {Action.UP: 0.025, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.925},
        (1, 0): {Action.UP: 0.925, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.025},
        (2, 0): {Action.UP: 0.925, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.025},
    }

    assert agent.evaluation_policy == {
        (2, 0): {Action.UP: 1.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0},
        (1, 0): {Action.UP: 1.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0},
        (0, 0): {Action.UP: 1.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0},
        (0, 1): {Action.UP: 1.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0},
        (0, 2): {Action.UP: 0.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 1.0},
    }

    assert agent.action_value == {
        ((2, 0), Action.LEFT): 0.0,
        ((2, 0), Action.UP): 0.0,
        ((2, 0), Action.DOWN): 0.0,
        ((2, 0), Action.RIGHT): 0.0,
        ((1, 0), Action.UP): 0.0,
        ((0, 0), Action.LEFT): 0.0,
        ((1, 0), Action.DOWN): 0.0,
        ((1, 0), Action.LEFT): 0.0,
        ((1, 0), Action.RIGHT): 0.0,
        ((0, 0), Action.RIGHT): 0.0,
        ((0, 0), Action.UP): 0.0,
        ((0, 0), Action.DOWN): 0.0,
        ((0, 1), Action.LEFT): 0.0,
        ((0, 1), Action.UP): 0.0,
        ((0, 1), Action.DOWN): 0.0,
        ((0, 1), Action.RIGHT): 0.0,
        ((0, 2), Action.UP): 0.0,
        ((0, 2), Action.RIGHT): 0.96,
        ((0, 2), Action.DOWN): 0.0,
        ((0, 2), Action.LEFT): 0.0,
    }


if __name__ == "__main__":
    pytest.main()
