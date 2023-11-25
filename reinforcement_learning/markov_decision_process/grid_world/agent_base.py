"""The base agent class for reinforcement learning models."""
from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionResult,
    ReadOnlyPolicy,
    State,
)


class AgentBase(ABC):
    """Abstract base class for reinforcement learning agents."""

    def __init__(self: Self, *, seed: int | None) -> None:
        """Initialize the AgentBase with the given seed.

        Args:
        ----
            seed (int): The seed value for random number generation.
        """
        self.__rng: np.random.Generator = np.random.default_rng(seed=seed)

    @property
    @abstractmethod
    def behavior_policy(self: Self) -> ReadOnlyPolicy:
        """Return the behavior policy."""

    def get_action(self: Self, *, state: State) -> Action:
        """Select an action based on policy `self.__b`.

        Args:
        ----
            state: the state of the environment.
            policy: the policy of the agent.

        Returns:
        -------
            the chosen action based on the action probabilities for the given state.
        """
        action_probs = self.behavior_policy[state]
        probs = list(action_probs.values())
        return Action(self.__rng.choice(list(Action), p=probs))

    @abstractmethod
    def add_memory(self: Self, *, state: State, action: Action | None, result: ActionResult | None) -> None:
        """Add a new experience into the memory.

        Args:
        ----
            state: The current state of the agent.
            action: The action taken by the agent.
            result: The result of the action taken by the agent.
        """

    @abstractmethod
    def reset_memory(self: Self) -> None:
        """Clear the memory of the reinforcement learning agent."""

    @abstractmethod
    def update(self: Self) -> None:
        """Update the agent's internal state based on the current conditions and any new information."""
