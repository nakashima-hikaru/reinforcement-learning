"""An implementation of the Epsilon-Greedy Agent.

An algorithm for multi-armed bandit problem in reinforcement learning.
The agent uses an epsilon-greedy strategy to balance exploration and exploitation during the learning process.
The policy, named 'epsilon-greedy', randomly explores with epsilon probability,
otherwise exploits its current knowledge.
The agent's state and action-value estimations are updated based on the rewards received after choosing actions.
"""
from typing import ClassVar, Final, Self, final

import numpy as np

from reinforcement_learning.markov_decision_process.bandit_problem.agents.base import EpsilonGreedyAgentBase

SEED: Final[int] = 0


@final
class EpsilonGreedyAgent(EpsilonGreedyAgentBase):
    """An agent for Epsilon-greedy exploration strategy for the multi-armed bandit problem."""

    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, epsilon: float, action_size: int) -> None:
        """Initialize an EpsilonGreedyAgent instance.

        Args:
        ----
            epsilon (float): The exploration rate, between 0.0 and 1.0.
            action_size (int): The number of possible actions.

        Returns:
        -------
            None
        """
        super().__init__(epsilon, action_size)

    def update(self: Self, i_action: int, reward: float) -> None:
        """Update the agent's estimate of the action value based on the received reward.

        Args:
        ----
            i_action (int): The index of the chosen action.
            reward (float): The reward received after taking the action.

        Returns:
        -------
            None
        """
        self._ns[i_action] += 1
        self._qs[i_action] += (reward - self._qs[i_action]) / self._ns[i_action]
