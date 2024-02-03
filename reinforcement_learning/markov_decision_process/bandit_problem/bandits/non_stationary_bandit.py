"""The NonStationaryBandit class, a variant of multi-armed bandit problem.

The NonStationaryBandit class represents a non-stationary bandit problem where the probabilities of rewards
for each arm evolve over time.
In other words, the bandit's arms have dynamically changing payoff rates, unlike the stationary bandit
where rates remain stable.
These changes in rates are dictated by a modification
in the NonStationaryBandit's parent BanditBase class's _change_rates method,
where Gaussian noise is added to the existing rates.
"""
from typing import final

import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.markov_decision_process.bandit_problem.bandits.base import BanditBase


@final
class NonStationaryBandit(BanditBase):
    """A class representing a non-stationary bandit problem."""

    def __init__(self, *, n_arms: int, seed: int | None = None) -> None:
        """Initialize NonStationaryBandit.

        Args:
            n_arms: An integer representing the number of arms in the stationary bandit problem.
            seed: An optional seed value for random number generation.
        """
        super().__init__(n_arms=n_arms, seed=seed)

    def _next_rates(self, *, rates: NDArray[np.float64]) -> NDArray[np.float64]:
        return rates + 0.1 * self.rng.standard_normal(size=self.n_arms)
