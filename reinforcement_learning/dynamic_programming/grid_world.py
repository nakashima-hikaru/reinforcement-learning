import logging
from collections.abc import Iterator
from enum import IntEnum, unique
from typing import Final, Self, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from reinforcement_learning.errors import NumpyDimError

State: TypeAlias = tuple[int, int]
Map: TypeAlias = npt.NDArray[np.float64]
MAP_DIM: Final[int] = 2


@unique
class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def direction(self: Self) -> State:
        ret: State
        match self:
            case Action.UP:
                ret = -1, 0
            case Action.DOWN:
                ret = 1, 0
            case Action.LEFT:
                ret = 0, -1
            case Action.RIGHT:
                ret = 0, 1
        return ret


class GridWorld:
    def __init__(
        self: Self,
        reward_map: Map,
        goal_state: State,
        start_state: State,
    ) -> None:
        if reward_map.ndim != MAP_DIM:
            raise NumpyDimError(expected_dim=2, actual_dim=reward_map.ndim)
        self.__reward_map: Final[Map] = reward_map

        def collect_walls(mp: Map) -> set[State]:
            se: set[State] = set()
            for x, row in enumerate(mp):
                for y, coordinate in enumerate(row):
                    if np.isnan(coordinate):
                        se.add((x, y))
            return se

        self.__wall_states: frozenset[State] = frozenset(collect_walls(reward_map))
        self.__goal_state: State = goal_state
        self.__start_state: State = start_state
        self.__agent_state: State = self.__start_state

    @property
    def goal_state(self: Self) -> State:
        return self.__goal_state

    @property
    def agent_state(self: Self) -> State:
        return self.__agent_state

    @property
    def height(self: Self) -> int:
        return len(self.__reward_map)

    @property
    def width(self: Self) -> int:
        return len(self.__reward_map[0])

    @property
    def shape(self: Self) -> tuple[int, int]:
        return cast(tuple[int, int], self.__reward_map.shape)

    def states(self: Self) -> Iterator[State]:
        for h in range(self.height):
            for w in range(self.width):
                yield h, w

    def next_state(self: Self, state: State, action: Action) -> State:
        next_state: State = (
            state[0] + action.direction[0],
            state[1] + action.direction[1],
        )
        ny, nx = next_state
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height or next_state == (1, 1):
            next_state = state
        return next_state

    def reward(self: Self, next_state: State) -> float:
        return cast(float, self.__reward_map[next_state])

    def step(self: Self, action: Action) -> tuple[State, float, bool]:
        """Applies a given action to itself and returns the result."""
        next_state = self.next_state(state=self.__agent_state, action=action)
        reward = self.reward(next_state)
        done = next_state == self.__goal_state
        self.__agent_state = next_state
        return next_state, reward, done

    def reset(self: Self) -> None:
        self.__agent_state = self.__start_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    logging.info(env.reward(next_state=(0, 3)))
