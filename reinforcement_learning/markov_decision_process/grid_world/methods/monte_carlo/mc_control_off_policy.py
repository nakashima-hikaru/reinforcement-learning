"""A class `McOffPolicyAgent` which epitomizes an agent that uses Monte Carlo Off-Policy learning.

`McOffPolicyAgent` class handles the key processes of the agent during the reinforcement learning simulation.
It's constructed with several parameters
including gamma (decay factor), epsilon (for epsilon-greedy policy) and alpha (learning rate).
"""
from collections import defaultdict
from types import MappingProxyType
from typing import Self, final

from reinforcement_learning.markov_decision_process.grid_world.environment import (
    RANDOM_ACTIONS,
    ActionValue,
    ActionValueView,
    Policy,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import McAgentBase
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_eval import greedy_probs


@final
class McOffPolicyAgent(McAgentBase):
    """A class that represents an agent that uses Monte Carlo Off-Policy learning."""

    def __init__(self: Self, *, gamma: float, epsilon: float, alpha: float, seed: int = 0) -> None:
        """Initialize the instance with the provided parameters.

        Args:
        ----
            gamma (float): Decay factor, should be a positive real number less than or equal to 1.
            epsilon (float): Epsilon for epsilon-greedy policy, should be a positive real number less than or equal to 1.
            alpha (float): Learning rate, should be a positive real number less than or equal to 1.
            seed (int): Seed for action selector.
        """
        super().__init__(seed=seed)
        self.__gamma: float = gamma
        self.__epsilon: float = epsilon
        self.__alpha: float = alpha
        self.__evaluation_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.__q: ActionValue = defaultdict(lambda: 0.0)

    @property
    def q(self: Self) -> ActionValueView:
        """Get the current value of the action-value function.

        Returns
        -------
            ActionValue: The instance's internal action-value function.
        """
        return MappingProxyType(self.__q)

    def update(self: Self) -> None:
        """Update the action-value function and policies in reinforcement learning."""
        g: float = 0.0
        rho: float = 1.0
        for memory in reversed(self.memories):
            g = self.__gamma * g * rho + memory.reward
            rho *= (
                    self.__evaluation_policy[memory.state][memory.action]
                    / self.behavior_policy[memory.state][memory.action]
            )
            self.__q[memory.state, memory.action] += (g - self.__q[memory.state, memory.action]) * self.__alpha
            self.__evaluation_policy[memory.state] = greedy_probs(q=self.__q, state=memory.state, epsilon=0.0)
            self.behavior_policy[memory.state] = greedy_probs(q=self.__q, state=memory.state, epsilon=self.__epsilon)
