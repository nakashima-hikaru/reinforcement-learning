"""Sarsa agent."""
from collections import defaultdict, deque
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Self, final

from pydantic import StrictBool, StrictFloat
from pydantic.dataclasses import dataclass

from reinforcement_learning.errors import NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionResult,
    ActionValueView,
    PolicyView,
    State,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_eval import greedy_probs

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


@final
class SarsaAgent(AgentBase):
    """SARSA agent."""

    max_memory_length: Final[int] = 2

    def __init__(self: Self, *, seed: int | None = None) -> None:
        """Initialize an instance of the SarsaAgent class."""
        super().__init__(seed=seed)
        self.__gamma: float = 0.9
        self.__alpha: float = 0.8
        self.__epsilon: float = 0.1
        self.__q: ActionValue = defaultdict(lambda: 0.0)
        self.__memories: deque[SarsaMemory] = deque(maxlen=SarsaAgent.max_memory_length)

    @property
    def q(self: Self) -> ActionValueView:
        """Get the current value of the action-value function.

        Returns:
        -------
            ActionValue: The instance's internal action-value function.
        """
        return MappingProxyType(self.__q)

    @property
    def behavior_policy(self: Self) -> PolicyView:
        """Return the action selector policy."""
        return MappingProxyType(self._behavior_policy)

    @property
    def memories(self: Self) -> tuple[SarsaMemory, ...]:
        """Return a tuple of memories."""
        return tuple(self.__memories)

    def add_memory(self: Self, *, state: State, action: Action | None, result: ActionResult | None) -> None:
        """Add a new experience into the memory."""
        memory = SarsaMemory(
            state=state,
            action=action,
            reward=result.reward if result is not None else None,
            done=result.done if result is not None else None,
        )
        self.__memories.append(memory)

    def reset_memory(self: Self) -> None:
        """Reset the agent's memory."""
        self.__memories.clear()

    def update(self: Self) -> None:
        """Update the Q-values in the Sarsa agent."""
        if len(self.__memories) < SarsaAgent.max_memory_length:
            return
        current_memory = self.__memories[0]
        if current_memory.action is None:
            raise NotInitializedError(instance_name=str(current_memory), attribute_name="action")
        next_memory = self.__memories[1]
        if current_memory.done:
            next_q = 0.0
        else:
            if next_memory.action is None:
                raise NotInitializedError(instance_name=str(next_memory), attribute_name="action")
            next_q = self.__q[next_memory.state, next_memory.action]
        if self.__memories[0].reward is None:
            raise NotInitializedError(instance_name=str(self.__memories[0]), attribute_name="reward")
        target = self.__memories[0].reward + self.__gamma * next_q
        key = current_memory.state, current_memory.action
        self.__q[key] += (target - self.__q[key]) * self.__alpha
        self._behavior_policy[current_memory.state] = greedy_probs(
            q=self.__q, state=current_memory.state, epsilon=self.__epsilon
        )
