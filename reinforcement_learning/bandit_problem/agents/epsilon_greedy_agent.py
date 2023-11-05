from typing import ClassVar, Final, Self

import numpy as np
import numpy.typing as npt

from reinforcement_learning.bandit_problem.agents.base import EpsilonGreedyAgentBase

SEED: Final[int] = 0


class EpsilonGreedyAgent(EpsilonGreedyAgentBase):
    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, epsilon: float, action_size: int) -> None:
        super().__init__(epsilon, action_size)
        self._ns: npt.NDArray[np.int64] = np.zeros(action_size, dtype=np.int64)

    def update(self: Self, i_action: int, reward: float) -> None:
        """Updates the `index_of_action`-th quality."""
        self._ns[i_action] += 1
        self._qs[i_action] += (reward - self._qs[i_action]) / self._ns[i_action]
