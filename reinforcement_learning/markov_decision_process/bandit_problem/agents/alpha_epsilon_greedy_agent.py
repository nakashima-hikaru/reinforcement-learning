"""Provide an implementation of the Alpha-Epsilon-Greedy Agent, an algorithm for the multi-armed bandit problem in reinforcement learning.

The agent employs an epsilon-greedy strategy (with a step-size) to balance exploration and exploitation, and it uses a constant learning rate 'alpha' for value estimation updates.
"""
from typing import Self, final

from reinforcement_learning.markov_decision_process.bandit_problem.agents.base import EpsilonGreedyAgentBase


@final
class AlphaEpsilonGreedyAgent(EpsilonGreedyAgentBase):
    """Implement an Alpha Epsilon Greedy Agent for multi-armed bandit problems."""

    def __init__(self: Self, *, epsilon: float, action_size: int, alpha: float) -> None:
        """Initialize AlphaEpsilonGreedyAgent.

        Args:
        ----
            epsilon (float): The value of epsilon for epsilon-greedy action selection.
            action_size (int): The number of possible actions.
            alpha (float): The learning rate for updating action values.
        """
        super().__init__(epsilon, action_size)
        self.alpha: float = alpha

    def update(self: Self, i_action: int, reward: float) -> None:
        """Update the action-value estimation for the specified action using the given reward.

        Args:
        ----
            i_action (int): The index of the action to update the estimation for.
            reward (float): The reward received after taking the action.

        Returns:
        -------
            None
        """
        self._qs[i_action] += (reward - self._qs[i_action]) * self.alpha
