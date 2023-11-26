"""An abstract base class 'BanditBase' for implementing bandit algorithms.

A bandit algorithm is a type of machine learning algorithm used to make decisions under uncertainty.
The class includes core methods used in a bandit algorithm such as playing a single round ('play') of the bandit game
and changing the reward rates of the arms ('_change_rates').
An important concept in bandit algorithms is that of 'arms'
which refer to the options or decisions that the agent can make.
"""
from abc import ABC, abstractmethod
from typing import Final, Self, cast

import numpy as np
from numpy.typing import NDArray

SEED: Final[int] = 0


class BanditBase(ABC):
    """Abstract base class for implementing bandit algorithms."""

    def __init__(self: Self, *, n_arms: int, seed: int | None = None) -> None:
        """Initialize BanditBase.

        Args:
        ----
            n_arms: The number of arms in the bandit.
            seed: An optional seed value for random number generation.

        """
        self.__n_arms: int = n_arms
        self.__rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.rates: NDArray[np.float64] = self.__rng.random(n_arms)

    @property
    def rng(self: Self) -> np.random.Generator:
        """Return the random number generator."""
        return self.__rng

    @property
    def n_arms(self: Self) -> int:
        """Return the number of the arms."""
        return self.__n_arms

    @abstractmethod
    def _next_rates(self: Self, *, rates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return next rates.

        Args:
        ----
            rates: An NDArray containing rates of a bandit machine.
        """

    def play(self: Self, *, i_arm: int) -> float:
        """Play a single round of the bandit game.

        Args:
        ----
            i_arm: An integer representing the index of the arm to play.

        Returns:
        -------
            A float indicating the reward obtained from playing the arm.
        """
        rate: np.float64 = cast(np.float64, self.rates[i_arm])
        self.rates = self._next_rates(rates=self.rates)
        if rate > float(self.__rng.random()):
            return 1.0
        return 0.0
