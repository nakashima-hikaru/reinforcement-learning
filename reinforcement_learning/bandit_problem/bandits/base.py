from abc import ABC, abstractmethod
from typing import ClassVar, Final, Self, cast

import numpy as np
from numpy.typing import NDArray

SEED: Final[int] = 0


class BanditBase(ABC):
    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, *, n_arms: int) -> None:
        self._n_arms: int = n_arms
        self.rates: NDArray[np.float64] = self._rng.random(n_arms)

    @abstractmethod
    def _change_rates(self: Self, *, rates: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def play(self: Self, *, i_arm: int) -> float:
        rate: np.float64 = cast(np.float64, self.rates[i_arm])
        self.rates = self._change_rates(rates=self.rates)
        if rate > float(self._rng.random()):
            return 1.0
        return 0.0
