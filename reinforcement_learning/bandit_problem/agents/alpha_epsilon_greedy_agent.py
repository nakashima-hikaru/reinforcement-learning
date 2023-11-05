from typing import Self

import numpy as np
import numpy.typing as npt

from reinforcement_learning.bandit_problem.agents.base import EpsilonGreedyAgentBase


class AlphaEpsilonGreedyAgent(EpsilonGreedyAgentBase):
    def __init__(self: Self, *, epsilon: float, action_size: int, alpha: float) -> None:
        super().__init__(epsilon, action_size)
        self._ns: npt.NDArray[np.int64] = np.zeros(action_size, dtype=np.int64)
        self.alpha: float = alpha

    def update(self: Self, i_action: int, reward: float) -> None:
        self._qs[i_action] += (reward - self._qs[i_action]) * self.alpha
