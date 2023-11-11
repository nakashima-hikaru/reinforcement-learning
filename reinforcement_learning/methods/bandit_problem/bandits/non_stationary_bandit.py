"""The NonStationaryBandit class, a variant of multi-armed bandit problem.

The NonStationaryBandit class represents a non-stationary bandit problem where the probabilities of rewards
for each arm evolve over time.
In other words, the bandit's arms have dynamically changing payoff rates, unlike the stationary bandit
where rates remain stable.
These changes in rates are dictated by a modification
in the NonStationaryBandit's parent BanditBase class's _change_rates method,
where Gaussian noise is added to the existing rates.
"""
from typing import Self

import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.methods.bandit_problem.bandits.base import BanditBase


class NonStationaryBandit(BanditBase):
    """A class representing a non-stationary bandit problem."""

    def __init__(self: Self, *, n_arms: int) -> None:
        """Initialize NonStationaryBandit.

        Args:
        ----
            n_arms: An integer representing the number of arms in the stationary bandit problem.

        """
        super().__init__(n_arms=n_arms)

    def _change_rates(self: Self, *, rates: NDArray[np.float64]) -> NDArray[np.float64]:
        return rates + 0.1 * self._rng.standard_normal(size=self._n_arms)
