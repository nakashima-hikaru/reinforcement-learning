"""A Q-learning agent.

The QLearningMemory class represents a single transition in the reinforcement learning environment. Each transition
consists of a current state, reward received, next state and a boolean flag indicating whether the episode is done.
"""
from collections import defaultdict
from types import MappingProxyType
from typing import TYPE_CHECKING, final

import numpy as np
from pydantic import StrictBool, StrictFloat
from pydantic.dataclasses import dataclass

from reinforcement_learning.errors import NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionResult,
    ReadOnlyActionValue,
    State,
)

if TYPE_CHECKING:
    from reinforcement_learning.markov_decision_process.grid_world.environment import ActionValue


@final
@dataclass(frozen=True)
class QLearningMemory:
    """Memory class represents a single transition in a reinforcement learning environment.

    Attributes:
        state (State): The current state in the transition.
        reward (StrictFloat): The reward received in the transition.
        next_state (State): The next state after the transition.
        done (StrictBool): Indicates whether the episode is done after the transition.
    """

    state: State
    action: Action
    reward: StrictFloat
    next_state: State
    done: StrictBool


@final
class QLearningAgent(AgentBase):
    """An agent that uses the Q-learning algorithm to learn and make decisions in a grid world environment."""

    def __init__(self, *, seed: int | None):
        """Initialize the agent with the given seed."""
        super().__init__(seed=seed)
        self.__gamma: float = 0.9
        self.__alpha: float = 0.8
        self.__epsilon: float = 0.1
        self.__action_value: ActionValue = defaultdict(lambda: 0.0)
        self.__memory: QLearningMemory | None = None

    @property
    def action_value(self) -> ReadOnlyActionValue:
        """Get the current value of the action-value function.

        Returns:
            ActionValue: The instance's internal action-value function.
        """
        return MappingProxyType(self.__action_value)

    def get_action(self, *, state: State) -> Action:
        """Select an action based on `self.rng`."""
        if self.rng.random() < self.__epsilon:
            return Action(self.rng.choice(list(Action)))
        return Action(np.argmax([self.__action_value[state, action] for action in Action]).item())

    def add_memory(self, *, state: State, action: Action, result: ActionResult) -> None:
        """Add a new experience into the memory.

        Args:
            state: The current state of the agent.
            action: The action taken by the agent.
            result: The result of the action taken by the agent.
        """
        self.__memory = QLearningMemory(state=state, action=action, reward=result.reward, next_state=result.next_state, done=result.done)

    def reset_memory(self) -> None:
        """Reset the agent's memory."""
        self.__memory = None

    def update(self) -> None:
        """Update the action-value estimates of the Q-learning agent."""
        if self.__memory is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="__memory")
        max_next_action_value = 0.0 if self.__memory.done else max(self.__action_value[self.__memory.next_state, action] for action in Action)
        target = self.__gamma * max_next_action_value + self.__memory.reward
        self.__action_value[self.__memory.state, self.__memory.action] += (target - self.__action_value[self.__memory.state, self.__memory.action]) * self.__alpha
