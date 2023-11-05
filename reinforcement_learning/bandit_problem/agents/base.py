"""Define the abstract base class for an epsilon-greedy strategy agent in a reinforcement learning system.

The class provides the core structure for an agent implementing an epsilon-greedy policy,
which provides a balance between exploration and exploitation while learning a policy.
The class includes essential methods that each epsilon-greedy agent should implement such as 'update' for updating value
estimation and 'get_action' to get the next action following the epsilon-greedy policy.
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Final, Self

import numpy as np
import numpy.typing as npt

SEED: Final[int] = 0


class EpsilonGreedyAgentBase(ABC):
    """Abstract base class for implementing the Epsilon-Greedy agent.

    This class provides a base implementation for an Epsilon-Greedy agent,
    which is an exploration-exploitation algorithm commonly used in Reinforcement Learning.
    """

    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, epsilon: float, action_size: int) -> None:
        """Initialize EpsilonGreedyAgentBase.

        Args:
        ----
            epsilon (float): The value of the exploration rate. Must be between 0 and 1, inclusive.
            action_size (int): The number of possible actions.

        """
        self._epsilon: float = epsilon
        self._qs: npt.NDArray[np.float64] = np.zeros(action_size, dtype=np.float64)

    @abstractmethod
    def update(self: Self, i_action: int, reward: float) -> None:
        """Update the agent's internal state based on the given action and reward.

        Args:
        ----
            i_action: An integer representing the chosen action.
            reward: A floating-point number representing the reward received.

        """

    def get_action(self: Self) -> int:
        """Determine an action according to its policy."""
        if self._rng.random() < self._epsilon:
            return int(self._rng.integers(0, len(self._qs)))
        return int(np.argmax(self._qs))
