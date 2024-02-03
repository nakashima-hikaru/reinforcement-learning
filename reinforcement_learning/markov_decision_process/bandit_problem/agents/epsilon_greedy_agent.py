"""An implementation of the Epsilon-Greedy Agent.

An algorithm for multi-armed bandit problem in reinforcement learning.
The agent uses an epsilon-greedy strategy to balance exploration and exploitation during the learning process.
The policy, named 'epsilon-greedy', randomly explores with epsilon probability,
otherwise exploits its current knowledge.
The agent's state and action-value estimations are updated based on the rewards received after choosing actions.
"""
from typing import final

import numpy as np
import numpy.typing as npt

from reinforcement_learning.markov_decision_process.bandit_problem.agents.base import EpsilonGreedyAgentBase


@final
class EpsilonGreedyAgent(EpsilonGreedyAgentBase):
    """An agent for Epsilon-greedy exploration strategy for the multi-armed bandit problem."""

    @property
    def action_values(self) -> npt.NDArray[np.float64]:
        """Return the array of action values for the current agent."""
        return self.__action_values

    def __init__(self, epsilon: float, action_size: int, seed: int | None = None) -> None:
        """Initialize an EpsilonGreedyAgent instance.

        Args:
            epsilon: The exploration rate, between 0.0 and 1.0.
            action_size: The number of possible actions.
            seed: An optional seed value for random number generation.

        Returns:
            None
        """
        super().__init__(epsilon=epsilon, seed=seed)
        self.__ns: npt.NDArray[np.int64] = np.zeros(action_size, dtype=np.int64)
        self.__action_values: npt.NDArray[np.float64] = np.zeros(action_size, dtype=np.float64)

    def update(self, *, i_action: int, reward: float) -> None:
        """Update the agent's estimate of the action value based on the received reward.

        Args:
            i_action (int): The index of the chosen action.
            reward (float): The reward received after taking the action.

        Returns:
            None
        """
        self.__ns[i_action] += 1
        self.__action_values[i_action] += (reward - self.__action_values[i_action]) / self.__ns[i_action]
