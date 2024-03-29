"""A grid world environment.

A 'State' is a tuple of two integers representing the coordinates in the grid.
'Map' is a 2D numpy array that represents the reward map for the grid world,
with 'MAP_DIM' being the dimension of the map (currently set to 2).

The 'Action' class defines possible actions (up, down, left, right) that an agent can take in the grid world.
Each action is represented as an integer value and has an associated direction,
which is used to compute the next state of the agent in the grid world environment.

The 'GridWorld' class represents a grid world environment for the agent.
It includes a reward map and the coordinates for the start and goal states.
It also defines a series of methods for retrieving information about the environment,
executing environment steps based on agent actions and computing rewards for each state transition.
These methods are essential for implementing and running reinforcement learning algorithms on the grid world environment.
"""
from collections import defaultdict
from collections.abc import Iterator
from enum import IntEnum, unique
from types import MappingProxyType
from typing import Final, TypeAlias, cast, final

import numpy as np
import numpy.typing as npt
import torch
from pydantic import StrictBool, StrictFloat
from pydantic.dataclasses import dataclass
from torch import Tensor

from reinforcement_learning.errors import NumpyDimError

State: TypeAlias = tuple[int, int]
Map: TypeAlias = npt.NDArray[np.float64]
MAP_DIM: Final[int] = 2


@final
@unique
class Action(IntEnum):
    """The Action class represents the set of possible actions in a game or simulation.

    Each action is represented as an integer value.
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __str__(self) -> str:
        """Return a string representation of the Action enumeration value."""
        ret: str
        match self:
            case Action.UP:
                ret = "Action.UP"
            case Action.DOWN:
                ret = "Action.DOWN"
            case Action.LEFT:
                ret = "Action.LEFT"
            case _:
                ret = "Action.RIGHT"
        return ret

    def __repr__(self) -> str:
        """Return a string representation of the Action enumeration value."""
        return f"Action.{self.name}"

    @property
    def direction(self) -> State:
        """Gets the direction of an action.

        Returns:
            ret (State): a tuple representing the direction of the action.
        """
        ret: State
        match self:
            case Action.UP:
                ret = -1, 0
            case Action.DOWN:
                ret = 1, 0
            case Action.LEFT:
                ret = 0, -1
            case _:
                ret = 0, 1
        return ret


Policy: TypeAlias = defaultdict[State, dict[Action, float]]
StateValue: TypeAlias = defaultdict[State, float]
ActionValue: TypeAlias = defaultdict[tuple[State, Action], float]

ReadOnlyPolicy: TypeAlias = MappingProxyType[State, dict[Action, float]]
ReadOnlyStateValue: TypeAlias = MappingProxyType[State, float]
ReadOnlyActionValue: TypeAlias = MappingProxyType[tuple[State, Action], float]


@final
@dataclass(frozen=True)
class ActionResult:
    """Represent the result of an action in the context of a reinforcement learning system.

    Args:
        next_state: The next state after taking the action.
        reward: The reward received for taking the action.
        done: A flag indicating whether the task or episode is completed after taking the action.
    """

    next_state: State
    reward: StrictFloat
    done: StrictBool


RANDOM_ACTIONS: Final[dict[Action, float]] = {
    Action.UP: 0.25,
    Action.DOWN: 0.25,
    Action.LEFT: 0.25,
    Action.RIGHT: 0.25,
}


@final
class GridWorld:
    """Class representing a GridWorld environment."""

    def __init__(
        self,
        reward_map: Map,
        goal_state: State,
        start_state: State,
    ) -> None:
        """Initialize a GridWorld object with the given reward map, goal state, and start state.

        Args:
            reward_map (Map): A 2D numpy array representing the reward map of the grid world.
            goal_state (State): The coordinates of the goal state in the grid world.
            start_state (State): The coordinates of the start state in the grid world.

        Raises:
            NumpyDimError: If the reward map has a dimension other than 2.
        """
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
        self.__goal_state: Final[State] = goal_state
        self.__start_state: Final[State] = start_state
        self.__agent_state: State = self.__start_state

    @property
    def goal_state(self) -> State:
        """Return the goal state of the GridWorld.

        Returns:
                State: The goal state of the GridWorld.

        """
        return self.__goal_state

    @property
    def agent_state(self) -> State:
        """Return the current state of the agent."""
        return self.__agent_state

    @property
    def wall_states(self) -> frozenset[State]:
        """Return the wall states of the GridWorld."""
        return self.__wall_states

    @property
    def height(self) -> int:
        """Return the height of the grid in the GridWorld object.

        Returns:
            int: The height of the grid.
        """
        return len(self.__reward_map)

    @property
    def width(self) -> int:
        """Return the width of the reward map.

        Returns:
            the length of the first element of the private class attribute __reward_map,
            which represents the width of the reward map.
        """
        return len(self.__reward_map[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Obtain the shape of the reward map as a tuple.

        Returns: the shape of the reward map.
        """
        return cast(tuple[int, int], self.__reward_map.shape)

    def state(self) -> Iterator[State]:
        """Iterate over the state of the GridWorld."""
        for h in range(self.height):
            for w in range(self.width):
                yield h, w

    def states(self) -> Iterator[State]:
        """Execute and yield all possible states in the two-dimensional grid."""
        for h in range(self.height):
            for w in range(self.width):
                yield h, w

    def next_state(self, state: State, action: Action) -> State:
        """Move to the next state based on the provided action.

        Args:
            self (Self): An instance of the current object.
            state (State): A tuple representing the current state (y_coordinate, x_coordinate).
            action (Action): An object representing the action to be taken.

        Returns:
            State: A tuple representing the next state after performing the action.
        """
        next_state: State = (
            state[0] + action.direction[0],
            state[1] + action.direction[1],
        )
        ny, nx = next_state
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height or next_state in self.__wall_states:
            next_state = state
        return next_state

    def reward(self, *, next_state: State) -> float:
        """Compute the reward for a given state transition.

        Args:
            next_state (State): The state to which transition is made.

        Returns:
            float: The reward for transitioning to the provided state.
        """
        return cast(float, self.__reward_map[next_state])

    def step(self, *, action: Action) -> ActionResult:
        """Perform an environment step based on the provided action.

        Args:
        action (Action): The action taken by the agent in the current state of the environment.

        Returns:
        tuple(State, float, bool): The next state, reward from the current action and whether the goal state is reached.
        """
        next_state = self.next_state(state=self.__agent_state, action=action)
        reward = self.reward(next_state=next_state)
        done = next_state == self.__goal_state
        self.__agent_state = next_state
        return ActionResult(next_state=next_state, reward=reward, done=done)

    def reset_agent_state(self) -> None:
        """Reset the agent's state to the start state."""
        self.__agent_state = self.__start_state

    def convert_to_one_hot(self, *, state: State) -> Tensor:
        """Convert the given state into a one-hot encoded tensor.

        Args:
            state: The state to be converted into a one-hot encoded tensor.

        Returns:
            A one-hot encoded tensor representing the given state.
        """
        vec = torch.zeros(self.height * self.width)
        y, x = state
        idx = self.width * y + x
        vec[idx] = 1.0
        return vec.unsqueeze(dim=0)
