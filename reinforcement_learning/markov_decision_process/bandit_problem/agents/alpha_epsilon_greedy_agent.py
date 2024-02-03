"""Provide an implementation of the Alpha-Epsilon-Greedy Agent, an algorithm for the multi-armed bandit problem in reinforcement learning.

The agent employs an epsilon-greedy strategy (with a step-size) to balance exploration and exploitation, and it uses a constant learning rate 'alpha' for value estimation updates.
"""
from typing import final

import numpy as np
import numpy.typing as npt

from reinforcement_learning.markov_decision_process.bandit_problem.agents.base import EpsilonGreedyAgentBase


@final
class AlphaEpsilonGreedyAgent(EpsilonGreedyAgentBase):
    """Implement an Alpha Epsilon Greedy Agent for multi-armed bandit problems."""

    @property
    def action_values(self) -> npt.NDArray[np.float64]:
        """Return the array of action values for the current agent."""
        return self.__action_values

    def __init__(self, *, epsilon: float, action_size: int, alpha: float, seed: int | None = None) -> None:
        """Initialize AlphaEpsilonGreedyAgent.

        Args:
            epsilon: The value of epsilon for epsilon-greedy action selection.
            action_size: The number of possible actions.
            alpha: The learning rate for updating action values.
            seed: The seed value for random number generation. Must be an integer or None.
        """
        super().__init__(epsilon=epsilon, seed=seed)
        self.__alpha: float = alpha
        self.__action_values: npt.NDArray[np.float64] = np.zeros(action_size, dtype=np.float64)

    def update(self, i_action: int, reward: float) -> None:
        """Update the action-value estimation for the specified action using the given reward.

        Args:
            i_action (int): The index of the action to update the estimation for.
            reward (float): The reward received after taking the action.

        Returns:
            None
        """
        self.__action_values[i_action] += (reward - self.__action_values[i_action]) * self.__alpha
