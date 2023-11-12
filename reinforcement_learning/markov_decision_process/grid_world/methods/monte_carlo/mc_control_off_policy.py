"""A class `McOffPolicyAgent` which epitomizes an agent that uses Monte Carlo Off-Policy learning.

`McOffPolicyAgent` class handles the key processes of the agent during the reinforcement learning simulation.
It's constructed with several parameters
including gamma (decay factor), epsilon (for epsilon-greedy policy) and alpha (learning rate).
"""
from collections import defaultdict
from typing import Self

from reinforcement_learning.markov_decision_process.grid_world.environment import (
    RANDOM_ACTIONS,
    ActionValue,
    Policy,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import McAgentBase
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_eval import greedy_probs


class McOffPolicyAgent(McAgentBase):
    """A class that represents an agent that uses Monte Carlo Off-Policy learning."""

    def __init__(self: Self, gamma: float, epsilon: float, alpha: float) -> None:
        """Initialize the instance with the provided parameters.

        Args:
        ----
            gamma (float): Decay factor, should be a positive real number less than or equal to 1.
            epsilon (float): Epsilon for epsilon-greedy policy, should be a positive real number less than or equal to 1.
            alpha (float): Learning rate, should be a positive real number less than or equal to 1.
        """
        super().__init__()
        self.__gamma: float = gamma
        self.__epsilon: float = epsilon
        self.__alpha: float = alpha
        self.__pi: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.__q: ActionValue = defaultdict(lambda: 0.0)

    def update(self: Self) -> None:
        """Update the action-value function and policies in reinforcement learning."""
        g: float = 0.0
        rho: float = 1.0
        for state, action, reward in reversed(self._memory):
            rho *= self.__pi[state][action] / self._b[state][action]
            g = self.__gamma * g + reward
            key = state, action
            self.__q[key] += (g - self.__q[key]) * self.__alpha * rho
            self.__pi[state] = greedy_probs(q=self.__q, state=state, epsilon=0.0)
            self._b[state] = greedy_probs(q=self.__q, state=state, epsilon=self.__epsilon)
