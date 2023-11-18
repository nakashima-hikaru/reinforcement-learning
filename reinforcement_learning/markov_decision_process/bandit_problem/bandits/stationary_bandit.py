"""The stationary bandit problem - a common problem in reinforcement learning.

The implementation is based on the BanditBase class and represents a bandit problem
where the reward probabilities remain constant over time. This is referred to as a 'stationary' bandit problem.
The class defines a method for maintaining the same reward rates ('_change_rates'),
since in a stationary bandit problem, the rates do not change.
"""
from typing import Self, final

import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.markov_decision_process.bandit_problem.bandits.base import BanditBase


@final
class StationaryBandit(BanditBase):
    """A class representing a stationary bandit problem."""

    def __init__(self: Self, *, n_arms: int) -> None:
        """Initialize StationaryBandit.

        Args:
        ----
            n_arms: An integer representing the number of arms in the stationary bandit problem.

        """
        super().__init__(n_arms=n_arms)

    def _change_rates(self: Self, *, rates: NDArray[np.float64]) -> NDArray[np.float64]:
        return rates
