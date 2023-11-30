"""A Sarsa on-policy agent class as part of the reinforcement learning project.

The `SarsaOnPolicyAgent` class extends the base `SarsaAgentBase` class, and overwrites its action-value and behavior policy
properties, and the update method. It uses an on-policy update rule to update its Q-values, and therefore chooses the same action
used for learning to also determine its next action.

This method allows the agent to take into account the current policy while determining its future action. It allows the agent
to learn directly from its experiences, and balance exploration and exploitation based on the current understanding of the environment.

This module also handles edge cases such as lack of initialized error in the action and reward of the current and next memory,
and keeps track of the action-value and behavior policy on each state as it continues to gather experience.
"""
from collections import defaultdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, final

from reinforcement_learning.errors import InvalidMemoryError
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
    from reinforcement_learning.markov_decision_process.grid_world.environment import (
        ActionValue,
        Policy,
    )


@final
class SarsaOnPolicyAgent(SarsaAgentBase):
    """A Sarsa agent that uses an on-policy update rule to update its Q-values."""

    def __init__(self: Self, *, seed: int | None = None):
        """Initialize the instance.

        Args:
            seed: a seed for action selection.
        """
        super().__init__(seed=seed)
        self.__action_value: ActionValue = defaultdict(lambda: 0.0)
        self.__behavior_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)

    @property
    def action_value(self: Self) -> ReadOnlyActionValue:
        """Get the current value of the action-value function.

        Returns:
            ActionValue: The instance's internal action-value function.
        """
        return MappingProxyType(self.__action_value)

    @property
    def behavior_policy(self: Self) -> ReadOnlyPolicy:
        """Return the behavior policy."""
        return MappingProxyType(self.__behavior_policy)

    def update(self: Self) -> None:
        """Update the Q-values in the Sarsa agent."""
        if len(self.memories) < SarsaAgentBase.max_memory_length:
            return
        current_memory = self.memories[0]
        if isinstance(current_memory, tuple):
            message = "Memory must be cleared after state-only memory added to memories"
            raise InvalidMemoryError(message)
        if current_memory.done:
            next_q = 0.0
        else:
            next_memory = self.memories[1]
            if isinstance(next_memory, tuple):
                message = "State-only memory must be added after an episode is done"
                raise InvalidMemoryError(message)
            next_q = self.__action_value[next_memory.state, next_memory.action]
        target = current_memory.reward + self.gamma * next_q
        key = current_memory.state, current_memory.action
        self.__action_value[key] += (target - self.__action_value[key]) * self.alpha
        self.__behavior_policy[current_memory.state] = greedy_probs(
            q=self.__action_value, state=current_memory.state, epsilon=self.epsilon
        )
