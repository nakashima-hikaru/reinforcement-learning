"""Sarsa agent."""
from abc import ABC
from collections import defaultdict, deque
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Self, final

from pydantic import StrictBool, StrictFloat
from pydantic.dataclasses import dataclass

from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionResult,
    ReadOnlyActionValue,
    ReadOnlyPolicy,
    State,
)

if TYPE_CHECKING:
    from reinforcement_learning.markov_decision_process.grid_world.environment import (
        ActionValue,
    )


@final
@dataclass(frozen=True)
class SarsaMemory:
    """The Memory class represents a piece of memory in a reinforcement learning algorithm.

    Attributes:
    ----------
        state (State): The current state of the environment.
        action (Action | None): The action taken by the agent.
        reward (StrictFloat | None): The reward received for the action taken.
        done (StrictBool | None): Indicates whether the episode is done or not.
    """

    state: State
    action: Action | None
    reward: StrictFloat | None
    done: StrictBool | None


class SarsaAgentBase(AgentBase, ABC):
    """SARSA agent."""

    max_memory_length: Final[int] = 2

    def __init__(self: Self, *, seed: int | None) -> None:
        """Initialize an instance of the SarsaAgent class."""
        super().__init__(seed=seed)
        self._gamma: float = 0.9
        self._alpha: float = 0.8
        self._epsilon: float = 0.1
        self._q: ActionValue = defaultdict(lambda: 0.0)
        self._memories: deque[SarsaMemory] = deque(maxlen=SarsaAgentBase.max_memory_length)

    @property
    def q(self: Self) -> ReadOnlyActionValue:
        """Get the current value of the action-value function.

        Returns:
        -------
            ActionValue: The instance's internal action-value function.
        """
        return MappingProxyType(self._q)

    @property
    def behavior_policy(self: Self) -> ReadOnlyPolicy:
        """Return the action selector policy."""
        return MappingProxyType(self._behavior_policy)

    @property
    def memories(self: Self) -> tuple[SarsaMemory, ...]:
        """Return a tuple of memories."""
        return tuple(self._memories)

    def add_memory(self: Self, *, state: State, action: Action | None, result: ActionResult | None) -> None:
        """Add a new experience into the memory."""
        memory = SarsaMemory(
            state=state,
            action=action,
            reward=result.reward if result is not None else None,
            done=result.done if result is not None else None,
        )
        self._memories.append(memory)

    def reset_memory(self: Self) -> None:
        """Reset the agent's memory."""
        self._memories.clear()
