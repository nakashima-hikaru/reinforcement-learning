"""Interface."""
from typing import Self

from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import State


class TdAgentBase(AgentBase):
    """A base class for an agent of temporary difference method."""

    def update(self: Self, *, state: State, reward: float, next_state: State, done: bool) -> None:
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
