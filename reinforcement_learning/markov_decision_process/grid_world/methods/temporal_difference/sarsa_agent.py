"""Sarsa agent."""
from abc import ABC
from collections import deque
from typing import Final, Self, final

from pydantic import StrictBool, StrictFloat
from pydantic.dataclasses import dataclass

from reinforcement_learning.markov_decision_process.grid_world.agent_base import DistributionModelAgent
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionResult,
    State,
)


@final
@dataclass(frozen=True)
class SarsaMemory:
    """The Memory class represents a piece of memory in a reinforcement learning algorithm.

    Attributes:
        state (State): The current state of the environment.
        action (Action): The action taken by the agent.
        reward (StrictFloat): The reward received for the action taken.
        done (StrictBool): Indicates whether the episode is done or not.
    """

    state: State
    action: Action
    reward: StrictFloat
    done: StrictBool


class SarsaAgentBase(DistributionModelAgent, ABC):
    """SARSA agent."""

    max_memory_length: Final[int] = 2

    def __init__(self: Self, *, seed: int | None) -> None:
        """Initialize an instance of the SarsaAgent class."""
        super().__init__(seed=seed)
        self.__gamma: float = 0.9
        self.__alpha: float = 0.8
        self.__epsilon: float = 0.1
        self.__memories: deque[SarsaMemory | State] = deque(maxlen=SarsaAgentBase.max_memory_length)

    @property
    def gamma(self: Self) -> float:
        """Return the gamma value of the SarsaAgentBase.

        Returns:
            float: The gamma value of the SarsaAgentBase.
        """
        return self.__gamma

    @property
    def alpha(self: Self) -> float:
        """Return the alpha value of the SarsaAgentBase.

        Returns:
            float: The alpha value of the SarsaAgentBase.
        """
        return self.__alpha

    @property
    def epsilon(self: Self) -> float:
        """Return the epsilon value of the SarsaAgentBase.

        Returns:
            float: The epsilon value of the SarsaAgentBase.
        """
        return self.__epsilon

    @property
    def memories(self: Self) -> tuple[SarsaMemory | State, ...]:
        """Return a tuple of memories."""
        return tuple(self.__memories)

    def add_memory(self: Self, *, state: State, action: Action, result: ActionResult) -> None:
        """Add a new experience into the memory.

        Args:
            state: The current state of the agent.
            action: The action taken by the agent.
            result: The result of the action taken by the agent.
        """
        memory = SarsaMemory(
            state=state,
            action=action,
            reward=result.reward,
            done=result.done,
        )
        self.__memories.append(memory)

    def add_state_as_memory(self: Self, *, state: State) -> None:
        """Add a state to the agent's memory.

        Args:
            state: The state to be added to the agent's memory.
        """
        self.__memories.append(state)

    def reset_memory(self: Self) -> None:
        """Reset the agent's memory."""
        self.__memories.clear()
