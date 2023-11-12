"""The base agent class for reinforcement learning models."""
from abc import ABC
from collections import defaultdict
from typing import ClassVar, Final, Self, final

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import RANDOM_ACTIONS, Action, Policy, State

SEED: Final[int] = 0


class AgentBase(ABC):
    """The interface for agent class."""

    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self) -> None:
        """Initialize the instance."""
        self._b: Policy = defaultdict(lambda: RANDOM_ACTIONS)

    @final
    def get_action(self: Self, state: State) -> Action:
        """Select an action based on policy `self.__b`.

        Args:
        ----
            state (State): the state of the environment.

        Returns:
        -------
            the chosen action based on the action probabilities for the given state.
        """
        action_probs = self._b[state]
        probs = list(action_probs.values())
        return Action(self._rng.choice(list(Action), p=probs))
