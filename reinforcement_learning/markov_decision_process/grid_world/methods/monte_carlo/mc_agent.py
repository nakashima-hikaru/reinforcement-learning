"""Interface."""
from abc import ABC, abstractmethod
from typing import Self, final

from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, State


class McAgentBase(AgentBase, ABC):
    """A base class for reinforcement learning agents using Monte Carlo methods."""

    def __init__(self: Self) -> None:
        """Initialize the instance."""
        super().__init__()
        self._memory: list[tuple[State, Action, float]] = []

    @final
    def add(self: Self, state: State, action: Action, reward: float) -> None:
        """Add a new experience into the memory.

        Args:
        ----
           state: The current state of the environment.
           action: The action taken in the current state.
           reward: The reward received after taking the action.
        """
        data = (state, action, reward)
        self._memory.append(data)

    @final
    def reset(self: Self) -> None:
        """Clear the memory of the reinforcement learning agent."""
        self._memory.clear()

    @abstractmethod
    def update(self: Self) -> None:
        """Update the value function and/or the policy."""
