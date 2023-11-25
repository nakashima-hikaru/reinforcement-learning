from typing import Self

from reinforcement_learning.errors import NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.sarsa_agent import (
    SarsaAgentBase,
)
from reinforcement_learning.markov_decision_process.grid_world.util import greedy_probs


class SarsaOnPolicyAgent(SarsaAgentBase):
    """A Sarsa agent that uses an on-policy update rule to update its Q-values."""

    def update(self: Self) -> None:
        """Update the Q-values in the Sarsa agent."""
        if len(self._memories) < SarsaAgentBase.max_memory_length:
            return
        current_memory = self._memories[0]
        if current_memory.action is None:
            raise NotInitializedError(instance_name=str(current_memory), attribute_name="action")
        next_memory = self._memories[1]
        if current_memory.done:
            next_q = 0.0
        else:
            if next_memory.action is None:
                raise NotInitializedError(instance_name=str(next_memory), attribute_name="action")
            next_q = self._q[next_memory.state, next_memory.action]
        if self._memories[0].reward is None:
            raise NotInitializedError(instance_name=str(self._memories[0]), attribute_name="reward")
        target = self._memories[0].reward + self._gamma * next_q
        key = current_memory.state, current_memory.action
        self._q[key] += (target - self._q[key]) * self._alpha
        self._behavior_policy[current_memory.state] = greedy_probs(
            q=self._q, state=current_memory.state, epsilon=self._epsilon
        )
