"""An abstract base class 'BanditBase' for implementing bandit algorithms.

A bandit algorithm is a type of machine learning algorithm used to make decisions under uncertainty.
The class includes core methods used in a bandit algorithm such as playing a single round ('play') of the bandit game
and changing the reward rates of the arms ('_change_rates').
An important concept in bandit algorithms is that of 'arms'
which refer to the options or decisions that the agent can make.
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Final, Self, cast

import numpy as np
from numpy.typing import NDArray

SEED: Final[int] = 0


class BanditBase(ABC):
    """Abstract base class for implementing bandit algorithms."""

    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, *, n_arms: int) -> None:
        """Initialize BanditBase.

        Args:
        ----
            n_arms: The number of arms in the bandit.

        """
        self._n_arms: int = n_arms
        self.rates: NDArray[np.float64] = self._rng.random(n_arms)

    @abstractmethod
    def _change_rates(self: Self, *, rates: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

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
        self.rates = self._change_rates(rates=self.rates)
        if rate > float(self._rng.random()):
            return 1.0
        return 0.0