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
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, final

from pydantic import StrictBool, StrictFloat

from reinforcement_learning.errors import InvalidMemoryError, NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.agent_base import DistributionModelAgent
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    RANDOM_ACTIONS,
    Action,
    ActionResult,
    ReadOnlyPolicy,
    ReadOnlyStateValue,
    State,
)

if TYPE_CHECKING:
    from reinforcement_learning.markov_decision_process.grid_world.environment import (
        Policy,
        StateValue,
    )


@final
@dataclass(frozen=True)
class TdMemory:
    """Memory class represents a single transition in a reinforcement learning environment.

    Attributes:
    ----------
        state (State): The current state in the transition.
        reward (StrictFloat): The reward received in the transition.
        next_state (State): The next state after the transition.
        done (StrictBool): Indicates whether the episode is done after the transition.
    """

    state: State
    reward: StrictFloat
    next_state: State
    done: StrictBool


@final
class TdAgent(DistributionModelAgent):
    """Represent a Temporal Difference (TD) Agent for reinforcement learning."""

    def __init__(self: Self, *, gamma: float, alpha: float, seed: int | None = None) -> None:
        """Initialize the reinforcement learning object.

        Args:
        ----
            gamma: Discount factor for future rewards.
            alpha: Learning rate for updating the state values.
            seed: Seed used for random number generation. Defaults to 0.
        """
        super().__init__(seed=seed)
        self.__gamma: float = gamma
        self.__alpha: float = alpha
        self.__v: StateValue = defaultdict(lambda: 0.0)
        self.__memory: TdMemory | None = None
        self.__behavior_policy: Policy = defaultdict(lambda: RANDOM_ACTIONS)

    @property
    def behavior_policy(self: Self) -> ReadOnlyPolicy:
        """Return the behavior policy."""
        return MappingProxyType(self.__behavior_policy)

    @property
    def v(self: Self) -> ReadOnlyStateValue:
        """Return the state value."""
        return MappingProxyType(self.__v)

    def add_memory(self: Self, *, state: State, action: Action | None, result: ActionResult | None) -> None:  # noqa: ARG002
        """Add a new experience into the memory."""
        if result is None:
            message = "result must not be None"
            raise InvalidMemoryError(message)
        memory = TdMemory(state=state, reward=result.reward, next_state=result.next_state, done=result.done)
        self.__memory = memory

    def reset_memory(self: Self) -> None:
        """Clear the memory of the reinforcement learning agent."""
        self.__memory = None

    def update(self: Self) -> None:
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
        if self.__memory is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="__memory")
        next_v: float = 0.0 if self.__memory.done else self.__v[self.__memory.next_state]
        target: float = self.__memory.reward + self.__gamma * next_v
        self.__v[self.__memory.state] += (target - self.__v[self.__memory.state]) * self.__alpha
