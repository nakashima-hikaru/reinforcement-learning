"""Interface."""
from abc import abstractmethod
from collections import deque
from typing import Self

from pydantic import StrictBool, StrictFloat
from pydantic.dataclasses import dataclass

from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, State


@dataclass
class Memory:
    """The Memory class represents a piece of memory in a reinforcement learning algorithm.

    Attributes
    ----------
        state (State): The current state of the environment.
        action (Action): The action taken by the agent.
        reward (StrictFloat): The reward received for the action taken.
        done (StrictBool): Indicates whether the episode is done or not.
    """

    state: State
    action: Action
    reward: StrictFloat
    done: StrictBool


class TdAgentBase(AgentBase):
    """A base class for an agent of temporary difference method."""

    def __init__(self: Self) -> None:
        """Initialize the TdAgentBase class."""
        super().__init__()
        self._memories: deque[Memory] = deque(maxlen=2)

    def reset(self: Self) -> None:
        """Reset the agent's memory."""
        self._memories.clear()

    @abstractmethod
    def update(self: Self, *, state: State, reward: float, next_state: State, done: bool) -> None:
        """Update the value of the current state using the temporal difference (TD) algorithm.

        Args:
        ----
            state (State): The current state.
            reward (float): The reward for taking the current action from the current state.
            next_state (State): The next state.
            done (bool): Indicates if the episode is done or not.

        Returns:
        -------
            None
        """
