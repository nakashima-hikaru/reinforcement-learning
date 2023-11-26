import numpy as np
from pytest_mock import MockerFixture

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.agent_episodes import (
    run_sarsa_episode,
    run_td_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.sarsa_on_policy import (
    SarsaOnPolicyAgent,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.td_eval import TdAgent


def test_run_sarsa_episode_already_on_goal(mocker: MockerFixture) -> None:
    m = mocker.patch(
        "reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.sarsa_agent.SarsaAgentBase.update"
    )
    agent = SarsaOnPolicyAgent()
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(2, 0), start_state=(2, 0))
    run_sarsa_episode(env=env, agent=agent)
    m.assert_not_called()


def test_run_td_episode_already_on_goal(mocker: MockerFixture) -> None:
    m = mocker.patch(
        "reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.sarsa_agent.SarsaAgentBase.update"
    )
    agent = TdAgent(gamma=0.9, alpha=0.01)
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(2, 0), start_state=(2, 0))
    run_td_episode(env=env, agent=agent)
    m.assert_not_called()
