"""A class `McOffPolicyAgent` which epitomizes an agent that uses Monte Carlo Off-Policy learning.

`McOffPolicyAgent` class handles the key processes of the agent during the reinforcement learning simulation.
It's constructed with several parameters
including gamma (decay factor), epsilon (for epsilon-greedy policy) and alpha (learning rate).
"""
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, Final, Self

import numpy as np

from reinforcement_learning.dynamic_programming.grid_world import (
    Action,
    State,
)
from reinforcement_learning.monte_carlo.mc_eval import RANDOM_ACTIONS, greedy_probs

if TYPE_CHECKING:
    from reinforcement_learning.dynamic_programming.policy_eval import ActionValue, Policy

SEED: Final[int] = 0


class McOffPolicyAgent:
    """A class that represents an agent that uses Monte Carlo Off-Policy learning."""

    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, gamma: float, epsilon: float, alpha: float) -> None:
        """Initialize the instance with the provided parameters.

        Args:
        ----
            gamma (float): Decay factor, should be a positive real number less than or equal to 1.
            epsilon (float): Epsilon for epsilon-greedy policy, should be a positive real number less than or equal to 1.
            alpha (float): Learning rate, should be a positive real number less than or equal to 1.
        """
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.pi: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.b: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.q: ActionValue = defaultdict(lambda: 0.0)
        self.__memory: list[tuple[State, Action, float]] = []

    def get_action(self: Self, state: State) -> Action:
        """Determine the next action based on given state.

        Args:
        ----
            state (State): The current state of the grid world.

        Returns:
        -------
            Action: The chosen action based on the agent's current policy.
        """
        action_probs = self.b[state]
        probs = list(action_probs.values())
        return Action(self._rng.choice(list(Action), p=probs))

    def add(self: Self, state: State, action: Action, reward: float) -> None:
        """Add a new experience into the memory.

        Args:
        ----
           state: The current state of the environment.
           action: The action taken in the current state.
           reward: The reward received after taking the action.
        """
        data = (state, action, reward)
        self.__memory.append(data)

    def reset(self: Self) -> None:
        """Reset the McOffPolicyAgent object by clearing its memory."""
        self.__memory.clear()

    def update(self: Self) -> None:
        """Update the action-value function and policies in reinforcement learning."""
        g: float = 0.0
        rho: float = 1.0
        for state, action, reward in reversed(self.__memory):
            rho *= self.pi[state][action] / self.b[state][action]
            g = self.gamma * g + reward
            key = state, action
            self.q[key] += (g - self.q[key]) * self.alpha * rho
            self.pi[state] = greedy_probs(q=self.q, state=state, epsilon=0.0)
            self.b[state] = greedy_probs(q=self.q, state=state, epsilon=self.epsilon)
