from collections import defaultdict
from dataclasses import dataclass, field
from typing import Final, cast
import numpy as np

from reinforcement_learning.dynamic_programming.grid_world import Action, State
from reinforcement_learning.dynamic_programming.policy_eval import Policy, StateValue

random_actions: Final[dict[Action, float]] = {Action.UP: 0.25, Action.DOWN: 0.25, Action.LEFT: 0.25, Action.RIGHT: 0.25}


@dataclass
class RandomAgent:
    gamma: float
    pi: Policy = field(default_factory=lambda: defaultdict(lambda: random_actions))
    v: StateValue = field(default_factory=lambda: defaultdict(lambda: 0.0))
    counts: defaultdict[State, int] = field(default_factory=lambda: defaultdict(lambda: 0))
    memory: list[tuple[State, Action, float]] = field(default_factory=list)

    def get_action(self, state: State) -> Action:
        action_probs = self.pi[state]
        actions = np.array(list(action_probs.keys()))
        probs = list(action_probs.values())
        return cast(Action, np.random.choice(actions, p=probs))

    def add(self, state: State, action: Action, reward: float) -> None:
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def eval(self) -> None:
        g: float = 0.0
        for data in reversed(self.memory):
            state, action, reward = data
            g = self.gamma * g + reward
            self.counts[state] += 1
            self.v[state] += (g - self.v[state]) / self.counts[state]
