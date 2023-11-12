"""Interface."""
from abc import ABC, abstractmethod
from typing import Self, final

from pydantic import StrictFloat
from pydantic.dataclasses import dataclass

from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, State


@dataclass
class Memory:
    """Memory class represents a memory of an agent in a grid world environment.

    Attributes
    ----------
        state (State): the current state of the agent.
        action (Action): the action taken by the agent in the current state.
        reward (float): the reward received by the agent for taking the action in the current state.
    """

    state: State
    action: Action
    reward: StrictFloat


class McAgentBase(AgentBase, ABC):
    """A base class for reinforcement learning agents using Monte Carlo methods."""

    def __init__(self: Self) -> None:
        """Initialize the instance."""
        super().__init__()
        self._memories: list[Memory] = []

    @final
    def add_memory(self: Self, state: State, action: Action, reward: float) -> None:
        """Add a new experience into the memory.

        Args:
        ----
           state: The current state of the environment.
           action: The action taken in the current state.
           reward: The reward received after taking the action.
        """
        memory = Memory(state=state, action=action, reward=reward)
        self._memories.append(memory)

    @final
    def reset(self: Self) -> None:
        """Clear the memory of the reinforcement learning agent."""
        self._memories.clear()

    @abstractmethod
    def update(self: Self) -> None:
        """Update the value function and/or the policy."""
