import logging
from collections.abc import Iterator
from enum import IntEnum, unique
from typing import Self, TypeAlias, cast

import numpy as np
import numpy.typing as npt

State: TypeAlias = tuple[int, int]
Map: TypeAlias = npt.NDArray[np.float64]


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
        if not np.issubdtype(reward_map.dtype, np.floating):
            msg = "dtype of reward_map must be floating to support `nan`"
            raise TypeError(msg)
        self.reward_map: Map = reward_map

        def _collect_walls(mp: Map) -> set[State]:
            se: set[State] = set()
            for x, row in enumerate(mp):
                for y, coordinate in enumerate(row):
                    if np.isnan(coordinate):
                        se.add((x, y))
            return se

        self.wall_states: frozenset[State] = frozenset(_collect_walls(self.reward_map))
        self.goal_state: State = goal_state
        self.start_state: State = start_state
        self.agent_state: State = self.start_state

    @property
    def height(self: Self) -> int:
        return len(self.reward_map)

    @property
    def width(self: Self) -> int:
        return len(self.reward_map[0])

    @property
    def shape(self: Self) -> tuple[int, int]:
        return cast(tuple[int, int], self.reward_map.shape)

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
        return float(cast(np.float64, self.reward_map[next_state]))

    def step(self: Self, action: Action) -> tuple[State, float, bool]:
        """Applies a given action to itself and returns the result."""
        next_state = self.next_state(state=self.agent_state, action=action)
        reward: float = self.reward(next_state)
        done: bool = next_state == self.goal_state
        self.agent_state = next_state
        return next_state, reward, done

    def reset(self: Self) -> None:
        self.agent_state = self.start_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    logging.info(env.reward(next_state=(0, 3)))
