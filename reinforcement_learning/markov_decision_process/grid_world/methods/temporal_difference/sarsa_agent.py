"""Sarsa agent."""
from collections import defaultdict, deque
from typing import Self, final

from pydantic import StrictBool, StrictFloat
from pydantic.dataclasses import dataclass

from reinforcement_learning.errors import NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.agent_base import ActionSelector
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionValue,
    State,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_eval import greedy_probs


@dataclass
class SarsaMemory:
    """The Memory class represents a piece of memory in a reinforcement learning algorithm.

    Attributes
    ----------
        state (State): The current state of the environment.
        action (Action): The action taken by the agent.
        reward (StrictFloat): The reward received for the action taken.
        done (StrictBool): Indicates whether the episode is done or not.
    """

    state: State
    action: Action
    reward: StrictFloat
    done: StrictBool


class SarsaAgent(ActionSelector):
    """SARSA agent."""

    def __init__(self: Self, *, seed: int = 0) -> None:
        """Initialize an instance of the SarsaAgent class."""
        super().__init__(seed=seed)
        self.__gamma: float = 0.9
        self.__alpha: float = 0.8
        self.__epsilon: float = 0.1
        self.__q: ActionValue = defaultdict(lambda: 0.0)
        self._memories: deque[SarsaMemory] = deque(maxlen=2)

    @final
    def reset_memory(self: Self) -> None:
        """Reset the agent's memory."""
        self._memories.clear()

    @final
    def update(self: Self) -> None:
        """Update the Q-values in the Sarsa agent."""
        if self._memories.maxlen is None:
            raise NotInitializedError(class_name=str(self), attribute_name="_memories")
        if len(self._memories) < self._memories.maxlen:
            return
        current_memory = self._memories[0]
        next_memory = self._memories[1]
        next_q = 0 if current_memory.done else self.__q[next_memory.state, next_memory.action]
        target = self._memories[0].reward + self.__gamma * next_q
        key = current_memory.state, current_memory.action
        self.__q[key] += (target - self.__q[key]) * self.__alpha
        self._b[current_memory.state] = greedy_probs(q=self.__q, state=current_memory.state, epsilon=self.__epsilon)
