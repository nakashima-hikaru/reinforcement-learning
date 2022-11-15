from collections import defaultdict
from dataclasses import dataclass, field
from typing import Final, cast
import numpy as np

from reinforcement_learning.dynamic_programming.grid_world import Action, State
from reinforcement_learning.dynamic_programming.policy_eval import Policy, StateValue, ActionValue
from reinforcement_learning.dynamic_programming.policy_iter import argmax

random_actions: Final[dict[Action, float]] = {Action.UP: 0.25, Action.DOWN: 0.25, Action.LEFT: 0.25, Action.RIGHT: 0.25}


@dataclass
class RandomAgent:
    gamma: float
    pi: Policy = field(default_factory=lambda: defaultdict(lambda: random_actions))
    v: StateValue = field(default_factory=lambda: defaultdict(lambda: 0.0))
    counts: defaultdict[State, int] = field(default_factory=lambda: defaultdict(lambda: 0))
    memory: list[tuple[State, Action, float]] = field(default_factory=list)
    _action_index: list[int] = field(default_factory=lambda: list(map(int, Action)))

    def get_action(self, state: State) -> Action:
        """Gets an action according to its policy `self.pi`."""
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(np.random.choice(self._action_index, p=probs))

    def add(self, state: State, action: Action, reward: float) -> None:
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def eval(self) -> None:
        g: float = 0.0
        for state, action, reward in reversed(self.memory):
            g = self.gamma * g + reward
            self.counts[state] += 1
            self.v[state] += (g - self.v[state]) / self.counts[state]


def greedy_probs(q: dict[tuple[State, Action], float], state: State, epsilon: float) -> dict[Action, float]:
    """Returns the action probability at `state` that represents the epsilon greedy policy obtained from action value
    `q`. """
    qs = dict()
    for action in Action:
        qs[action] = q[(state, action)]
    max_action = argmax(qs)
    base_prob = epsilon / len(Action)
    action_probs: dict[Action, float] = {action: base_prob for action in Action}
    action_probs[max_action] += 1.0 - epsilon
    return action_probs


@dataclass
class McAgent:
    gamma: float
    epsilon: float
    alpha: float
    pi: Policy = field(default_factory=lambda: defaultdict(lambda: random_actions))
    q: ActionValue = field(default_factory=lambda: defaultdict(lambda: 0.0))
    counts: defaultdict[tuple[State, Action], int] = field(default_factory=lambda: defaultdict(lambda: 0))
    memory: list[tuple[State, Action, float]] = field(default_factory=list)
    _action_index: list[int] = field(default_factory=lambda: list(map(int, Action)))

    def get_action(self, state: State) -> Action:
        """Gets an action according to its policy `self.pi`."""
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(np.random.choice(self._action_index, p=probs))

    def add(self, state: State, action: Action, reward: float) -> None:
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def update(self) -> None:
        g: float = 0.0
        for state, action, reward in reversed(self.memory):
            g = self.gamma * g + reward
            key = state, action
            self.counts[key] += 1
            self.q[key] += (g - self.q[key]) * self.alpha
            self.pi[state] = greedy_probs(q=self.q, state=state, epsilon=self.epsilon)
