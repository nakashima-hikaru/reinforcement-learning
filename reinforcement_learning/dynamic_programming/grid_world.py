# coding=utf-8
from enum import Enum, unique, auto
from typing import TypeAlias, Iterator, cast
import numpy as np
import numpy.typing as npt

State: TypeAlias = tuple[int, int]
Map: TypeAlias = npt.NDArray[np.float64]  # todo: add shape information after numpy introducing variadic generics


@unique
class Action(Enum):
    UP = auto(),
    DOWN = auto(),
    LEFT = auto(),
    RIGHT = auto(),

    @property
    def direction(self) -> State:
        match self:
            case Action.UP:
                return -1, 0
            case Action.DOWN:
                return 1, 0
            case Action.LEFT:
                return 0, -1
            case Action.RIGHT:
                return 0, 1


class GridWorld:

    def __init__(self,
                 reward_map: Map,
                 goal_state: State,
                 start_state: State,
                 ):
        if not np.issubdtype(reward_map.dtype, np.floating):
            raise TypeError('dtype of reward_map must be floating to support `n'
                            'an`')
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
    def height(self) -> int:
        return len(self.reward_map)

    @property
    def width(self) -> int:
        return len(self.reward_map[0])

    @property
    def shape(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.reward_map.shape)

    def states(self) -> Iterator[State]:
        for h in range(self.height):
            for w in range(self.width):
                yield h, w

    def next_state(self, state: State, action: Action) -> State:
        next_state: State = state[0] + action.direction[0], state[1] + action.direction[1]
        ny, nx = next_state
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == (1, 1):
            next_state = state
        return next_state

    def reward(self, next_state: State) -> float:
        return float(cast(np.float64, self.reward_map[next_state]))

    def step(self, action: Action) -> tuple[State, float, bool]:
        next_state = self.next_state(state=self.agent_state, action=action)
        reward: float = self.reward(next_state)
        done: bool = next_state == self.goal_state
        self.agent_state = next_state
        return next_state, reward, done

    def reset(self) -> None:
        self.agent_state = self.start_state


if __name__ == "__main__":
    test_map = np.array([[0.0, 0.0, 0.0, 1.0],
                         [0.0, None, 0.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0]], dtype=float)
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    print(env.reward(next_state=(0, 3)))
