"""The implementation of two reinforcement learning agent classes: `RandomAgent` and `McAgent`.

The `RandomAgent` class follows a randomized policy for taking actions, and learns from its experiences
to improve the value function.
The `McAgent` class implements a reinforcement learning agent using Monte Carlo methods, which are learning methods
based on averaging sample returns.
Several utility functions for agents are also included, such as `greedy_probs` which computes
epsilon-greedy action probabilities for a given state.
"""
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
    """An agent that makes decisions using a randomized policy, and learns from its experiences."""

    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, gamma: float) -> None:
        """Initialize the instance of the RandomAgent class.

        Args:
        ----
            gamma (float): Discount factor for rewards.
        """
        self.gamma: float = gamma
        self.pi: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.v: StateValue = defaultdict(lambda: 0.0)
        self.counts: defaultdict[State, int] = defaultdict(lambda: 0)
        self.__memory: list[tuple[State, Action, float]] = []

    def get_action(self: Self, state: State) -> Action:
        """Select an action based on policy `self.pi`.

        Args:
        ----
            state (State): the state of the environment.

        Returns:
        -------
            the chosen action based on the action probabilities for the given state.
        """
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(self._rng.choice(list(Action), p=probs))

    def add(self: Self, state: State, action: Action, reward: float) -> None:
        """Add a new experience to the memory.

        Args:
        ----
            state: The state of the agent.
            action: The action taken by the agent.
            reward: The reward received by the agent after taking the action.
        """
        data = (state, action, reward)
        self.__memory.append(data)

    def reset(self: Self) -> None:
        """Reset the memory storage of the class instance."""
        self.__memory.clear()

    def evaluate(self: Self) -> None:
        """Evaluate the value function for the current policy.

        Returns
        -------
             None
        """
        g: float = 0.0
        for state, _action, reward in reversed(self.__memory):
            g = self.gamma * g + reward
            self.counts[state] += 1
            self.v[state] += (g - self.v[state]) / self.counts[state]


def greedy_probs(
    q: dict[tuple[State, Action], float],
    state: State,
    epsilon: float,
) -> dict[Action, float]:
    """Compute the epsilon-greedy action probabilities for the given state.

    Args:
    ----
        q: A dictionary mapping (state, action) pairs to their respective value.
        state: The current state of the system.
        epsilon: The factor determining the trade-off between exploration and exploitation.

    Returns:
    -------
        A dictionary mapping actions to their epsilon-greedy probability.
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
    """The McAgent class implements a reinforcement learning agent using Monte Carlo methods."""

    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, gamma: float, epsilon: float, alpha: float) -> None:
        """Initialize a reinforcement learning agent with given parameters.

        Args:
        ----
            gamma: Discount factor used to decide how important are the future rewards.
            epsilon: Exploration factor used to decide the tradeoff between exploration and exploitation.
            alpha: Learning rate used to decide the step size in learning process.
        """
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.pi: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.q: ActionValue = defaultdict(lambda: 0.0)
        self.__memory: list[tuple[State, Action, float]] = []

    def get_action(self: Self, state: State) -> Action:
        """Add a new experience to the agent's memory.

        Args:
        ----
            state (State): The state of the agent at a specific time step.
            action (Action): The action taken by the agent at a specific time step.
            reward (float): The reward received by the agent after taking the action in the given state.
        """
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(self._rng.choice(list(Action), p=probs))

    def add(self: Self, state: State, action: Action, reward: float) -> None:
        """Add a new experience to the agent's memory.

        Args:
        ----
            state (State): The state of the agent at a specific time step.
            action (Action): The action taken by the agent at a specific time step.
            reward (float): The reward received by the agent after taking the action in the given state.
        """
        data = (state, action, reward)
        self.__memory.append(data)

    def reset(self: Self) -> None:
        """Clear the memory of the reinforcement learning agent."""
        self.__memory.clear()

    def update(self: Self) -> None:
        """Compute the epsilon-greedy action probabilities for the given state.

        Args:
        ----
            q: A dictionary mapping (state, action) pairs to their respective value.
            state: The current state of the system.
            epsilon: The factor determining the trade-off between exploration and exploitation.

        Returns:
        -------
            A dictionary mapping actions to their epsilon-greedy probability.
        """
        g: float = 0.0
        for state, action, reward in reversed(self.__memory):
            g = self.gamma * g + reward
            key = state, action
            self.q[key] += (g - self.q[key]) * self.alpha
            self.pi[state] = greedy_probs(q=self.q, state=state, epsilon=self.epsilon)
