"""An off-policy Monte Carlo agent for reinforcement learning in a Markov Decision Process.

This implementation includes exploration using epsilon-greedy action selection and learning with
specified learning rate. The agent operates in a grid-world environment and uses a greedy policy based
on action-value estimates.
"""
from collections import defaultdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, final

from reinforcement_learning.errors import NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    RANDOM_ACTIONS,
    ReadOnlyActionValue,
    ReadOnlyPolicy,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.sarsa_agent import (
    SarsaAgentBase,
)
from reinforcement_learning.markov_decision_process.grid_world.util import greedy_probs

if TYPE_CHECKING:
    from reinforcement_learning.markov_decision_process.grid_world.environment import ActionValue, Policy


@final
class SarsaOffPolicyAgent(SarsaAgentBase):
    """An agent that uses the Sarsa algorithm for off-policy learning in a grid world environment."""

    def __init__(self: Self, *, seed: int | None = None) -> None:
        """Initialize an instance of the SarsaOffPolicyAgent class.

        Args:
            seed: An optional integer representing the seed value for random number generation.
        """
        super().__init__(seed=seed)
        self.__evaluation_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.__action_value: ActionValue = defaultdict(lambda: 0.0)
        self.__behavior_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)

    @property
    def behavior_policy(self: Self) -> ReadOnlyPolicy:
        """Return the behavior policy."""
        return MappingProxyType(self.__behavior_policy)

    @property
    def action_value(self: Self) -> ReadOnlyActionValue:
        """Get the current value of the action-value function.

        Returns:
        -------
            ActionValue: The instance's internal action-value function.
        """
        return MappingProxyType(self.__action_value)

    @property
    def evaluation_policy(self: Self) -> ReadOnlyPolicy:
        """Return the evaluation policy."""
        return MappingProxyType(self.__evaluation_policy)

    def update(self: Self) -> None:
        """Update the Q-values in the Sarsa agent."""
        if len(self.memories) < SarsaAgentBase.max_memory_length:
            return
        current_memory = self.memories[0]
        if current_memory.action is None:
            raise NotInitializedError(instance_name=str(current_memory), attribute_name="action")
        next_memory = self.memories[1]
        if current_memory.done:
            next_q = 0.0
            rho = 1.0
        else:
            if next_memory.action is None:
                raise NotInitializedError(instance_name=str(next_memory), attribute_name="action")
            next_q = self.__action_value[next_memory.state, next_memory.action]
            rho = (
                self.__evaluation_policy[current_memory.state][current_memory.action]
                / self.behavior_policy[current_memory.state][current_memory.action]
            )
        if current_memory.reward is None:
            raise NotInitializedError(instance_name=str(current_memory), attribute_name="reward")
        target = rho * (current_memory.reward + self.gamma * next_q)
        key = current_memory.state, current_memory.action
        self.__action_value[key] += (target - self.__action_value[key]) * self.alpha
        self.__behavior_policy[current_memory.state] = greedy_probs(
            q=self.__action_value, state=current_memory.state, epsilon=self.epsilon
        )
        self.__evaluation_policy[current_memory.state] = greedy_probs(
            q=self.__action_value, state=current_memory.state, epsilon=0.0
        )
