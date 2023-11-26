from typing import cast

import numpy as np
import pytest
from _pytest.fixtures import SubRequest

from reinforcement_learning.markov_decision_process.grid_world.environment import Action, GridWorld


@pytest.fixture()
def mock_env() -> GridWorld:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return GridWorld(reward_map=test_map, goal_state=(2, 0), start_state=(0, 0))


def test_initialization(mock_env: GridWorld) -> None:
    assert mock_env.shape == (3, 4)
    assert mock_env.goal_state == (2, 0)


def test_states(mock_env: GridWorld) -> None:
    num_states = sum(1 for _ in mock_env.states())
    assert num_states == mock_env.shape[0] * mock_env.shape[1]

    all_states = set(mock_env.states())
    assert all_states == {(h, w) for h in range(mock_env.shape[0]) for w in range(mock_env.shape[1])}

    assert mock_env.wall_states == {(1, 1)}


@pytest.fixture(params=Action)
def action(request: SubRequest) -> Action:
    return cast(Action, request.param)


def test_str(action: Action) -> None:
    assert str(action) == f"Action.{action.name}"


@pytest.mark.parametrize(
    ("action", "direction"),
    [(Action.UP, (-1, 0)), (Action.DOWN, (1, 0)), (Action.LEFT, (0, -1)), (Action.RIGHT, (0, 1))],
)
def test_direction(action: Action, direction: tuple[int, int]) -> None:
    assert action.direction == direction
