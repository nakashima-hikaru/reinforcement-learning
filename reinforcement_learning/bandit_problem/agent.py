# coding=utf-8
import numpy as np
import numpy.typing as npt


class Agent:
    def __init__(self, epsilon: float, action_size: int):
        self.epsilon: float = epsilon
        self.qs: npt.NDArray[np.float64] = np.zeros(action_size)
        self.ns: npt.NDArray[np.int64] = np.zeros(action_size, dtype=np.int64)

    def update(self, i_action: int, reward: float) -> None:
        """Updates the `index_of_action`-th quality."""
        self.ns[i_action] += np.int64(1)
        self.qs[i_action] += (reward - self.qs[i_action]) / self.ns[i_action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, len(self.qs)))
        return int(np.argmax(self.qs))


class AlphaAgent(Agent):
    def __init__(self, epsilon: float, action_size: int, alpha: float):
        super().__init__(epsilon, action_size)
        self.alpha: float = alpha

    def update(self, i_action: int, reward: float) -> None:
        self.qs[i_action] += (reward - self.qs[i_action]) * self.alpha
