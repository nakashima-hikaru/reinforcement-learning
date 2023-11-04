from typing import TYPE_CHECKING, ClassVar, Final, Self

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

SEED: Final[int] = 0


class Bandit:
    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, *, n_arms: int) -> None:
        self.n_arms: int = n_arms
        self.rates: NDArray[np.float64] = self._rng.random(n_arms)

    def play(self: Self, *, i_arm: int) -> float:
        rate: np.float64 = self.rates[i_arm]
        if rate > float(self._rng.random()):
            return 1.0
        return 0.0


class NonStationaryBandit(Bandit):
    def __init__(self: Self, n_arms: int) -> None:
        super().__init__(n_arms=n_arms)

    def play(self: Self, i_arm: int) -> float:
        rate: np.float64 = self.rates[i_arm]
        self.rates += 0.1 * self._rng.standard_normal(self.n_arms)
        if rate > float(self._rng.random()):
            return 1.0
        return 0.0
