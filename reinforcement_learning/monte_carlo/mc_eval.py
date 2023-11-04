from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, Final, Self

import numpy as np

from reinforcement_learning.dynamic_programming.grid_world import Action, State
from reinforcement_learning.dynamic_programming.policy_iter import argmax

if TYPE_CHECKING:
    from reinforcement_learning.dynamic_programming.policy_eval import (
        ActionValue,
        Policy,
        StateValue,
    )

SEED: Final[int] = 0

RANDOM_ACTIONS: Final[dict[Action, float]] = {
    Action.UP: 0.25,
    Action.DOWN: 0.25,
    Action.LEFT: 0.25,
    Action.RIGHT: 0.25,
}


class RandomAgent:
    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, gamma: float) -> None:
        self.gamma: float = gamma
        self.pi: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.v: StateValue = defaultdict(lambda: 0.0)
        self.counts: defaultdict[State, int] = defaultdict(lambda: 0)
        self.memory: list[tuple[State, Action, float]] = []

    def get_action(self: Self, state: State) -> Action:
        """Gets an action according to its policy `self.pi`."""
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(self._rng.choice(list(Action), p=probs))

    def add(self: Self, state: State, action: Action, reward: float) -> None:
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self: Self) -> None:
        self.memory.clear()

    def evaluate(self: Self) -> None:
        g: float = 0.0
        for state, _action, reward in reversed(self.memory):
            g = self.gamma * g + reward
            self.counts[state] += 1
            self.v[state] += (g - self.v[state]) / self.counts[state]


def greedy_probs(
    q: dict[tuple[State, Action], float],
    state: State,
    epsilon: float,
) -> dict[Action, float]:
    """Returns the action probability at `state` that represents the epsilon greedy policy obtained from action value
    `q`.
    """
    qs = {}
    for action in Action:
        qs[action] = q[(state, action)]
    max_action = argmax(qs)
    base_prob = epsilon / len(Action)
    action_probs: dict[Action, float] = {action: base_prob for action in Action}
    action_probs[max_action] += 1.0 - epsilon
    return action_probs


class McAgent:
    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, gamma: float, epsilon: float, alpha: float) -> None:
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.pi: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.q: ActionValue = defaultdict(lambda: 0.0)
        self.memory: list[tuple[State, Action, float]] = []

    def get_action(self: Self, state: State) -> Action:
        """Gets an action according to its policy `self.pi`."""
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(self._rng.choice(list(Action), p=probs))

    def add(self: Self, state: State, action: Action, reward: float) -> None:
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self: Self) -> None:
        self.memory.clear()

    def update(self: Self) -> None:
        g: float = 0.0
        for state, action, reward in reversed(self.memory):
            g = self.gamma * g + reward
            key = state, action
            self.q[key] += (g - self.q[key]) * self.alpha
            self.pi[state] = greedy_probs(q=self.q, state=state, epsilon=self.epsilon)
