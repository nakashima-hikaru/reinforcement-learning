# coding=utf-8
from enum import Enum, unique, auto
from typing import TypeAlias, Iterator
import numpy as np
import numpy.typing as npt

Coordinate: TypeAlias = tuple[int, int]
Map: TypeAlias = npt.NDArray[npt.NDArray[np.float64]]


@unique
class Action(Enum):
    UP = auto(),
    DOWN = auto(),
    LEFT = auto(),
    RIGHT = auto(),


@property
def direction(action: Action) -> Coordinate:
    match action:
        case Action.UP:
            return 0, 1
        case Action.DOWN:
            return 0, -1
        case Action.LEFT:
            return -1, 0
        case Action.RIGHT:
            return 1, 0


class GridWorld:

    def __init__(self,
                 reward_map: Map,
                 goal_state: Coordinate,
                 start_state: Coordinate,
                 ):
        if not np.issubdtype(reward_map.dtype, np.floating):
            raise TypeError('dtype of reward_map must be floating to support `n'
                            'an`')
        self.reward_map: Map = reward_map

        def _collect_walls(mp: Map) -> set[Coordinate]:
            se: set[Coordinate] = set()
            for x, row in enumerate(mp):
                for y, coordinate in enumerate(row):
                    if np.isnan(coordinate):
                        se.add((x, y))
            return se

        self.wall_states: frozenset[Coordinate] = frozenset(_collect_walls(self.reward_map))
        self.goal_state: Coordinate = goal_state
        self.start_state: Coordinate = start_state
        self.agent_state: Coordinate = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def state(self) -> Iterator[Coordinate]:
        for h in range(self.height):
            for w in range(self.width):
                yield h, w

    def next_state(self, state: Coordinate, action: Action) -> Coordinate:
        next_state: Coordinate = state[0] + direction(action)[0], state[1] + direction(action)[1]
        nx, ny = next_state
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == (1, 1):
            next_state = state
        return next_state

    def reward(self, next_state: Coordinate) -> float:
        return self.reward_map[next_state]


if __name__ == "__main__":
    test_map = np.array([[0.0, 0.0, 0.0, 1.0],
                         [0.0, None, 0.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0]], dtype=float)
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    print(env.reward(next_state=(0, 3)))
