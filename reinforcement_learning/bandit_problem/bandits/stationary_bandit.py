from typing import Self

import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.bandit_problem.bandits.base import BanditBase


class StationaryBandit(BanditBase):
    def __init__(self: Self, *, n_arms: int) -> None:
        super().__init__(n_arms=n_arms)

    def _change_rates(self: Self, *, rates: NDArray[np.float64]) -> NDArray[np.float64]:
        return rates
