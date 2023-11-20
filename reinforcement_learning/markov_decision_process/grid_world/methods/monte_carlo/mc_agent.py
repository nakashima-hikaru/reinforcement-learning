"""Interface."""
from abc import ABC
from typing import Self, final

from pydantic import StrictFloat
from pydantic.dataclasses import dataclass

from reinforcement_learning.errors import InvalidMemoryError
from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, ActionResult, GridWorld, State


@final
@dataclass(frozen=True)
class McMemory:
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

    def __init__(self: Self, *, seed: int | None) -> None:
        """Initialize the instance."""
        super().__init__(seed=seed)
        self.__memories: list[McMemory] = []

    @property
    def memories(self: Self) -> tuple[McMemory, ...]:
        """Get the memories of the agent.

        Returns
        -------
            A list of `McMemory` objects representing the agent's memories.
        """
        return tuple(self.__memories)

    @final
    def add_memory(self: Self, *, state: State, action: Action | None, result: ActionResult | None) -> None:
        """Add a new experience into the memory."""
        if action is None or result is None:
            if action is None and result is None:
                message = "action or result must not be None"
            elif action is None:
                message = "action must not be None"
            else:
                message = "result must not be None"
            raise InvalidMemoryError(message)
        memory = McMemory(state=state, action=action, reward=result.reward)
        self.__memories.append(memory)

    @final
    def reset_memory(self: Self) -> None:
        """Clear the memory of the reinforcement learning agent."""
        self.__memories.clear()


def run_monte_carlo_episode(*, env: GridWorld, agent: McAgentBase) -> None:
    """Run a single episode."""
    env.reset_agent_state()
    state = env.agent_state
    agent.reset_memory()

    while True:
        action = agent.get_action(state=state)
        result = env.step(action=action)
        agent.add_memory(state=state, action=action, result=result)
        if result.done:
            break

        state = result.next_state
    agent.update()
