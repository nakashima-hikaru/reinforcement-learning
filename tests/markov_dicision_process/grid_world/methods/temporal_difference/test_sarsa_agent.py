import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import Action, GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.agent_episodes import (
    run_sarsa_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.sarsa_agent import (
    SarsaAgent,
)


def test_sarsa() -> None:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = SarsaAgent(seed=412)
    episodes = 2

    for _ in range(episodes):
        run_sarsa_episode(env=env, agent=agent)

    assert agent.behavior_policy == {
        (0, 0): {Action.UP: 0.925, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.025},
        (0, 1): {Action.UP: 0.025, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.925},
        (0, 2): {Action.UP: 0.025, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.925},
        (1, 0): {Action.UP: 0.925, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.025},
        (2, 0): {Action.UP: 0.925, Action.DOWN: 0.025, Action.LEFT: 0.025, Action.RIGHT: 0.025},
    }
    assert agent.q == {
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
        ((0, 1), Action.RIGHT): 0.5760000000000001,
        ((0, 2), Action.UP): 0.0,
        ((0, 2), Action.RIGHT): 0.96,
        ((0, 2), Action.DOWN): 0.0,
        ((0, 2), Action.LEFT): 0.0,
    }
