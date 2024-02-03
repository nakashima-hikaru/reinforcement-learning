import numpy as np
from pytest_mock import MockerFixture

from reinforcement_learning.markov_decision_process.cart_pole.main import run_episode


def test_run_episode(mocker: MockerFixture) -> None:
    # Mock the CartPoleEnv and DQNAgent
    env = mocker.Mock()
    env.reset.return_value = (np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), None)
    env.step.return_value = (np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32), 1.0, True, None, None)

    agent = mocker.Mock()
    agent.get_action.return_value = 0

    episode = 1
    sync_interval = 1

    # run the episode
    total_reward = run_episode(env, agent, episode, sync_interval)

    # assertions
    expected: float = 1.0
    assert total_reward == expected
    agent.get_action.assert_called_once()
    agent.update.assert_called_once()
    env.step.assert_called_once()
    env.close.assert_called_once()
