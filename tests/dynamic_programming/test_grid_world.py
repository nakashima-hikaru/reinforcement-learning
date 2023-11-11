import numpy as np
import pytest

from reinforcement_learning.errors import NumpyDimError
from reinforcement_learning.markov_decision_process.grid_world.environment import MAP_DIM, GridWorld


def test_shape_validation() -> None:
    reward_map = np.zeros(shape=(2, 3, 4), dtype=np.float64)
    with pytest.raises(NumpyDimError) as e:
        GridWorld(reward_map=reward_map, goal_state=(0, 1), start_state=(0, 0))
    assert e.value.actual_dim == reward_map.ndim
    assert e.value.expected_dim == MAP_DIM
