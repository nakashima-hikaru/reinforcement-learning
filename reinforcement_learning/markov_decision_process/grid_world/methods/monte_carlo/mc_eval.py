"""The implementation of two reinforcement learning agent classes: `RandomAgent` and `McAgent`.

The `RandomAgent` class follows a randomized policy for taking actions, and learns from its experiences
to improve the value function.
The `McAgent` class implements a reinforcement learning agent using Monte Carlo methods, which are learning methods
based on averaging sample returns.
Several utility functions for agents are also included, such as `greedy_probs` which computes
epsilon-greedy action probabilities for a given state.
"""
from collections import defaultdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, final

from reinforcement_learning.markov_decision_process.grid_world.environment import (
    RANDOM_ACTIONS,
    ReadOnlyPolicy,
    ReadOnlyStateValue,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import McAgentBase

if TYPE_CHECKING:
    from reinforcement_learning.markov_decision_process.grid_world.environment import Policy, State, StateValue


class RandomAgent(McAgentBase):
    """An agent that makes decisions using a randomized policy, and learns from its experiences."""

    def __init__(self: Self, *, gamma: float, seed: int | None = None) -> None:
        """Initialize the instance of the RandomAgent class.

        Args:
        ----
            gamma: A float representing the discount factor for future rewards.
            seed: An integer representing a seed value for random number generation.

        """
        super().__init__(seed=seed)
        self.__gamma: float = gamma
        self.__state_value: StateValue = defaultdict(lambda: 0.0)
        self.__counts: defaultdict[State, int] = defaultdict(lambda: 0)
        self.__behavior_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)

    @property
    def behavior_policy(self: Self) -> ReadOnlyPolicy:
        """Return the behavior policy."""
        return MappingProxyType(self.__behavior_policy)

    @property
    def state_value(self: Self) -> ReadOnlyStateValue:
        """Return the current state value."""
        return MappingProxyType(self.__state_value)

    @final
    def update(self: Self) -> None:
        """Evaluate the value function for the current policy.

        Returns:
        -------
             None
        """
        g: float = 0.0
        for memory in reversed(self.memories):
            g = self.__gamma * g + memory.reward
            self.__counts[memory.state] += 1
            self.__state_value[memory.state] += (g - self.__state_value[memory.state]) / self.__counts[memory.state]
