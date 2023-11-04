from typing import ClassVar, Final, Self

import numpy as np
import numpy.typing as npt

SEED: Final[int] = 0


class EpsilonGreedyAgent:
    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, epsilon: float, action_size: int) -> None:
        self._epsilon: float = epsilon
        self._qs: npt.NDArray[np.float64] = np.zeros(action_size, dtype=np.float64)
        self._ns: npt.NDArray[np.int64] = np.zeros(action_size, dtype=np.int64)

    def update(self: Self, i_action: int, reward: float) -> None:
        """Updates the `index_of_action`-th quality."""
        self._ns[i_action] += 1
        self._qs[i_action] += (reward - self._qs[i_action]) / self._ns[i_action]

    def get_action(self: Self) -> int:
        if self._rng.random() < self._epsilon:
            return int(self._rng.integers(0, len(self._qs)))
        return int(np.argmax(self._qs))


class AlphaEpsilonGreedyAgent(EpsilonGreedyAgent):
    def __init__(self: Self, epsilon: float, action_size: int, alpha: float) -> None:
        super().__init__(epsilon, action_size)
        self.alpha: float = alpha

    def update(self: Self, i_action: int, reward: float) -> None:
        self._qs[i_action] += (reward - self._qs[i_action]) * self.alpha
