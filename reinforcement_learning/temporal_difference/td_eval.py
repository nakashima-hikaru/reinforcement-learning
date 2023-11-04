import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, Final, Self

import numpy as np

from reinforcement_learning.dynamic_programming.grid_world import (
    Action,
    GridWorld,
    State,
)
from reinforcement_learning.monte_carlo.mc_eval import RANDOM_ACTIONS

if TYPE_CHECKING:
    from reinforcement_learning.dynamic_programming.policy_eval import Policy, StateValue

SEED: Final[int] = 0


class TdAgent:
    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, gamma: float, alpha: float) -> None:
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.pi: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.v: StateValue = defaultdict(lambda: 0.0)

    def get_action(self: Self, state: State) -> Action:
        """Gets an action according to its policy `self.pi`."""
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(self._rng.choice(list(Action), p=probs))

    def evaluate(self: Self, *, state: State, reward: float, next_state: State, done: bool) -> None:
        next_v: float = 0.0 if done else self.v[next_state]
        target: float = reward + self.gamma * next_v
        self.v[state] += (target - self.v[state]) * self.alpha


if __name__ == "__main__":
    np.random.default_rng(314)
    logging.basicConfig(level=logging.INFO)
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = TdAgent(gamma=0.9, alpha=0.01)
    n_episodes: int = 1000
    for _ in range(n_episodes):
        env.reset()
        state = env.agent_state
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action=action)
            agent.evaluate(state=state, reward=reward, next_state=next_state, done=done)
            if done:
                break
            state = next_state
    logging.info(agent.v)
