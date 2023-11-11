"""The TdAgent class, a reinforcement learning agent that uses Temporal Difference (TD) algorithm.

The Temporal Difference Agent (TdAgent) implements a TD learning algorithm in the reinforcement learning process. It
consists of two methods: a function to determine the next action based on the current state (`get_action`),
and a function to evaluate and update the state's value (`evaluate`). These processes use parameters such as gamma (
the discount factor) and alpha (the learning rate), provided during the initialization of the class.

The TD agent also maintains an internal policy (`pi`) and state-value function (`v`), both of which are initially
implemented as dictionaries with default values.

Moreover, it utilizes a policy to determine the probability distribution over the possible actions from a given
state, and a state-value function to hold the expected return for each state, considering the current policy.
Notably, the agent operates within a GridWorld environment, taking actions according to its policy, and updating the
policy and value function based on the results.

In the main function of the module, the TdAgent is tested within a manually defined GridWorld environment. This
includes initiating the agent and GridWorld environment, running a number of episodes, and finally logging the value
function of the agent.

This module can be run standalone to test the TdAgent in a GridWorld environment. Otherwise, the TdAgent class can be
imported to be used in a reinforcement learning process.
"""
from collections import defaultdict
from typing import ClassVar, Final, Self

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import (
    RANDOM_ACTIONS,
    Action,
    Policy,
    State,
    StateValue,
)

SEED: Final[int] = 0


class TdAgent:
    """Represent a Temporal Difference (TD) Agent for reinforcement learning."""

    _rng: ClassVar[np.random.Generator] = np.random.default_rng(seed=SEED)

    def __init__(self: Self, gamma: float, alpha: float) -> None:
        """Initialize the reinforcement learning object.

        Args:
        ----
            gamma (float): The discount factor to be used in the process, determines the importance of future rewards.
            alpha (float): The learning rate for the algorithm, determines the extent to which new information will override the old information.
        """
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.pi: Policy = defaultdict(lambda: RANDOM_ACTIONS)
        self.v: StateValue = defaultdict(lambda: 0.0)

    def get_action(self: Self, state: State) -> Action:
        """Return the action to take for the given state.

        Args:
        ----
            state (State): The current state of the agent.

        Returns:
        -------
            Action: The action to take for the given state.
        """
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(self._rng.choice(list(Action), p=probs))

    def evaluate(self: Self, *, state: State, reward: float, next_state: State, done: bool) -> None:
        """Update the value of the current state using the temporal difference (TD) algorithm.

        Args:
        ----
            state (State): The current state.
            reward (float): The reward for taking the current action from the current state.
            next_state (State): The next state.
            done (bool): Indicates if the episode is done or not.

        Returns:
        -------
            None
        """
        next_v: float = 0.0 if done else self.v[next_state]
        target: float = reward + self.gamma * next_v
        self.v[state] += (target - self.v[state]) * self.alpha
