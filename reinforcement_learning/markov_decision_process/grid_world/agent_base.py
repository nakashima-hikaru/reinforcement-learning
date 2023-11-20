"""The base agent class for reinforcement learning models."""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Self, final

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import (
    RANDOM_ACTIONS,
    Action,
    ActionResult,
    Policy,
    State,
)


@final
class ActionSelector:
    """The interface for agent class."""

    def __init__(self: Self, *, seed: int | None) -> None:
        """Initialize the instance."""
        self.__rng: np.random.Generator = np.random.default_rng(seed=seed)

    def get_action(self: Self, *, state: State, policy: Policy) -> Action:
        """Select an action based on policy `self.__b`.

        Args:
        ----
            state: the state of the environment.
            policy: the policy of the agent.

        Returns:
        -------
            the chosen action based on the action probabilities for the given state.
        """
        action_probs = policy[state]
        probs = list(action_probs.values())
        return Action(self.__rng.choice(list(Action), p=probs))


class AgentBase(ABC):
    """Abstract base class for reinforcement learning agents."""

    def __init__(self: Self, *, seed: int | None) -> None:
        """Initialize the AgentBase with the given seed.

        Args:
        ----
            seed (int): The seed value for random number generation.
        """
        self.__action_selector: ActionSelector = ActionSelector(seed=seed)
        self._behavior_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)

    @final
    def get_action(self: Self, *, state: State) -> Action:
        """Select an action based on policy `self.__b`.

        Args:
        ----
            state (State): the state of the environment.

        Returns:
        -------
            the chosen action based on the action probabilities for the given state.
        """
        return self.__action_selector.get_action(state=state, policy=self._behavior_policy)

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
