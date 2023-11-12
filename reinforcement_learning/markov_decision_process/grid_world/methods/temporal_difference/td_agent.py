"""Interface."""
from abc import ABC, abstractmethod
from typing import Self

from reinforcement_learning.markov_decision_process.grid_world.agent_base import ActionSelector


class TdAgentBase(ABC, ActionSelector):
    """A base class for an agent of temporary difference method."""

    def __init__(self: Self, *, seed: int = 0) -> None:
        """Initialize the TdAgentBase class."""
        super().__init__(seed=seed)

    @abstractmethod
    def update(self: Self) -> None:
        """Update the value of the current state using the temporal difference (TD) algorithm."""
