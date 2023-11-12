"""The implementation of two reinforcement learning agent classes: `RandomAgent` and `McAgent`.

The `RandomAgent` class follows a randomized policy for taking actions, and learns from its experiences
to improve the value function.
The `McAgent` class implements a reinforcement learning agent using Monte Carlo methods, which are learning methods
based on averaging sample returns.
Several utility functions for agents are also included, such as `greedy_probs` which computes
epsilon-greedy action probabilities for a given state.
"""
from collections import defaultdict
from typing import Self, final

from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionValue,
    State,
    StateValue,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import McAgentBase
from reinforcement_learning.util import argmax


class RandomAgent(McAgentBase):
    """An agent that makes decisions using a randomized policy, and learns from its experiences."""

    def __init__(self: Self, *, gamma: float, seed: int = 0) -> None:
        """Initialize the instance of the RandomAgent class.

        Args:
        ----
            gamma: A float representing the discount factor for future rewards.
            seed: An integer representing a seed value for random number generation.

        """
        super().__init__(seed=seed)
        self.__gamma: float = gamma
        self.__v: StateValue = defaultdict(lambda: 0.0)
        self.__counts: defaultdict[State, int] = defaultdict(lambda: 0)

    @property
    def v(self: Self) -> StateValue:
        """Return the current state value."""
        return self.__v

    @final
    def update(self: Self) -> None:
        """Evaluate the value function for the current policy.

        Returns
        -------
             None
        """
        g: float = 0.0
        for memory in reversed(self._memories):
            g = self.__gamma * g + memory.reward
            self.__counts[memory.state] += 1
            self.__v[memory.state] += (g - self.__v[memory.state]) / self.__counts[memory.state]


def greedy_probs(
    *,
    q: dict[tuple[State, Action], float],
    state: State,
    epsilon: float,
) -> dict[Action, float]:
    """Compute the epsilon-greedy action probabilities for the given state.

    Args:
    ----
        q: A dictionary mapping (state, action) pairs to their respective value.
        state: The current state of the system.
        epsilon: The factor determining the trade-off between exploration and exploitation.

    Returns:
    -------
        A dictionary mapping actions to their epsilon-greedy probability.
    """
    qs = {}
    for action in Action:
        qs[action] = q[(state, action)]
    max_action = argmax(qs)
    base_prob = epsilon / len(Action)
    action_probs: dict[Action, float] = {action: base_prob for action in Action}
    action_probs[max_action] += 1.0 - epsilon
    return action_probs


class McAgent(McAgentBase):
    """The McAgent class implements a reinforcement learning agent using Monte Carlo methods."""

    def __init__(self: Self, *, gamma: float, epsilon: float, alpha: float, seed: int = 0) -> None:
        """Initialize a reinforcement learning agent with given parameters.

        Args:
        ----
            gamma: Discount factor for future rewards.
            epsilon: Exploration factor for epsilon-greedy action selection.
            alpha: Learning rate for updating action values.
            seed: Seed for random number generation.

        """
        super().__init__(seed=seed)
        self.__gamma: float = gamma
        self.__epsilon: float = epsilon
        self.__alpha: float = alpha
        self.__q: ActionValue = defaultdict(lambda: 0.0)

    @property
    def q(self: Self) -> ActionValue:
        """Get the current value of the action-value function.

        Returns
        -------
            ActionValue: The instance's internal action-value function.
        """
        return self.__q

    @final
    def update(self: Self) -> None:
        """Compute the epsilon-greedy action probabilities for the given state.

        Args:
        ----
            q: A dictionary mapping (state, action) pairs to their respective value.
            state: The current state of the system.
            epsilon: The factor determining the trade-off between exploration and exploitation.

        Returns:
        -------
            A dictionary mapping actions to their epsilon-greedy probability.
        """
        g: float = 0.0
        for memory in reversed(self._memories):
            g = self.__gamma * g + memory.reward
            self.__q[memory.state, memory.action] += (g - self.__q[memory.state, memory.action]) * self.__alpha
            self._b[memory.state] = greedy_probs(q=self.__q, state=memory.state, epsilon=self.__epsilon)
