# coding=utf-8
import numpy as np
import numpy.typing as npt


class Bandit:
    def __init__(self, n_arms: int):
        self.n_arms: int = n_arms
        self.rates: npt.NDArray[np.float64] = np.random.rand(n_arms)

    def play(self, i_arm: int) -> float:
        rate: np.float64 = self.rates[i_arm]
        if rate > np.random.rand():
            return 1.0
        else:
            return 0.0


class NonStationaryBandit(Bandit):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def play(self, i_arm: int) -> float:
        rate: np.float64 = self.rates[i_arm]
        self.rates += 0.1 * np.random.randn(self.n_arms)
        if rate > np.random.rand():
            return 1.0
        else:
            return 0.0
