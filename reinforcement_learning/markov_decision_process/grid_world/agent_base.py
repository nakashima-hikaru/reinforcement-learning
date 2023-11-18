"""The base agent class for reinforcement learning models."""
from collections import defaultdict
from typing import Self, final

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import RANDOM_ACTIONS, Action, Policy, State


class ActionSelector:
    """The interface for agent class."""

    def __init__(self: Self, *, seed: int) -> None:
        """Initialize the instance."""
        self.__rng: np.random.Generator = np.random.default_rng(seed=seed)
        self._behavior_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)

    @property
    def behavior(self: Self) -> Policy:
        """Return the action selector policy."""
        return self._behavior_policy

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
        action_probs = self._behavior_policy[state]
        probs = list(action_probs.values())
        return Action(self.__rng.choice(list(Action), p=probs))
