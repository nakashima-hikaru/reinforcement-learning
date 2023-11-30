"""A class `McOffPolicyAgent` which epitomizes an agent that uses Monte Carlo Off-Policy learning.

`McOffPolicyAgent` class handles the key processes of the agent during the reinforcement learning simulation.
It's constructed with several parameters
including gamma (decay factor), epsilon (for epsilon-greedy policy) and alpha (learning rate).
"""
from collections import defaultdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, final

from reinforcement_learning.markov_decision_process.grid_world.environment import (
    RANDOM_ACTIONS,
    ReadOnlyActionValue,
    ReadOnlyPolicy,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import McAgentBase
from reinforcement_learning.markov_decision_process.grid_world.util import greedy_probs

if TYPE_CHECKING:
    from reinforcement_learning.markov_decision_process.grid_world.environment import (
        ActionValue,
        Policy,
    )


@final
class McOffPolicyAgent(McAgentBase):
    """A class that represents an agent that uses Monte Carlo Off-Policy learning."""

    def __init__(self: Self, *, gamma: float, epsilon: float, alpha: float, seed: int | None = None) -> None:
        """Initialize the instance with the provided parameters.

        Args:
            gamma: Decay factor, should be a positive real number less than or equal to 1.
            epsilon: Epsilon for epsilon-greedy policy, should be a positive real number less than or equal to 1.
            alpha: Learning rate, should be a positive real number less than or equal to 1.
            seed: Seed for action selector.
        """
        super().__init__(seed=seed)
        self.__gamma: float = gamma
        self.__epsilon: float = epsilon
        self.__alpha: float = alpha
        self.__evaluation_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.__action_value: ActionValue = defaultdict(lambda: 0.0)
        self.__behavior_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)

    @property
    def behavior_policy(self: Self) -> ReadOnlyPolicy:
        """Return the behavior policy."""
        return MappingProxyType(self.__behavior_policy)

    @property
    def evaluation_policy(self: Self) -> ReadOnlyPolicy:
        """Return the evaluation policy."""
        return MappingProxyType(self.__evaluation_policy)

    @property
    def action_value(self: Self) -> ReadOnlyActionValue:
        """Get the current value of the action-value function.

        Returns:
            ActionValue: The instance's internal action-value function.
        """
        return MappingProxyType(self.__action_value)

    def update(self: Self) -> None:
        """Update the action-value function and policies in reinforcement learning."""
        g: float = 0.0
        rho: float = 1.0
        for memory in reversed(self.memories):
            g = self.__gamma * g * rho + memory.reward
            rho *= (
                self.__evaluation_policy[memory.state][memory.action]
                / self.__behavior_policy[memory.state][memory.action]
            )
            self.__action_value[memory.state, memory.action] += (
                g - self.__action_value[memory.state, memory.action]
            ) * self.__alpha
            self.__evaluation_policy[memory.state] = greedy_probs(
                q=self.__action_value, state=memory.state, epsilon=0.0
            )
            self.__behavior_policy[memory.state] = greedy_probs(
                q=self.__action_value, state=memory.state, epsilon=self.__epsilon
            )
