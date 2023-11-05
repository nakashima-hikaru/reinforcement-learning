from abc import ABC, abstractmethod
from typing import ClassVar, Final, Self

import numpy as np
import numpy.typing as npt

SEED: Final[int] = 0


class EpsilonGreedyAgentBase(ABC):
    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, epsilon: float, action_size: int) -> None:
        self._epsilon: float = epsilon
        self._qs: npt.NDArray[np.float64] = np.zeros(action_size, dtype=np.float64)

    @abstractmethod
    def update(self: Self, i_action: int, reward: float) -> None:
        """Updates the `index_of_action`-th quality."""

    def get_action(self: Self) -> int:
        """Determines an action according to its policy."""
        if self._rng.random() < self._epsilon:
            return int(self._rng.integers(0, len(self._qs)))
        return int(np.argmax(self._qs))
