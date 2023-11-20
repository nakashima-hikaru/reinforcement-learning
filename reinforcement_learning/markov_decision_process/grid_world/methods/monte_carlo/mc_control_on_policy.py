"""An on-policy Monte Carlo agent for reinforcement learning in a Markov Decision Process.

This implementation includes exploration using epsilon-greedy action selection and learning with
specified learning rate. The agent operates in a grid-world environment and uses a greedy policy based
on action-value estimates.
"""
from collections import defaultdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, final

from reinforcement_learning.markov_decision_process.grid_world.environment import ActionValueView
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import McAgentBase
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_eval import greedy_probs

if TYPE_CHECKING:
    from reinforcement_learning.markov_decision_process.grid_world.environment import ActionValue


@final
class McOnPolicyAgent(McAgentBase):
    """The McAgent class implements a reinforcement learning agent using Monte Carlo methods."""

    def __init__(self: Self, *, gamma: float, epsilon: float, alpha: float, seed: int | None = None) -> None:
        """Initialize a reinforcement learning agent with given parameters.

        Args:
        ----
            gamma: Discount factor for future rewards.
            epsilon: Exploration factor for epsilon-greedy action selection.
            alpha: Learning rate for updating action values.
            seed: Seed for random number generation.

        """
        super().__init__(seed=seed)
        self.__gamma: float = gamma
        self.__epsilon: float = epsilon
        self.__alpha: float = alpha
        self.__q: ActionValue = defaultdict(lambda: 0.0)

    @property
    def q(self: Self) -> ActionValueView:
        """Get the current value of the action-value function.

        Returns
        -------
            ActionValue: The instance's internal action-value function.
        """
        return MappingProxyType(self.__q)

    def update(self: Self) -> None:
        """Compute the epsilon-greedy action probabilities for the given state.

        Args:
        ----
            q: A dictionary mapping (state, action) pairs to their respective value.
            state: The current state of the system.
            epsilon: The factor determining the trade-off between exploration and exploitation.
        """
        g: float = 0.0
        for memory in reversed(self.memories):
            g = self.__gamma * g + memory.reward
            self.__q[memory.state, memory.action] += (g - self.__q[memory.state, memory.action]) * self.__alpha
            self._behavior_policy[memory.state] = greedy_probs(q=self.__q, state=memory.state, epsilon=self.__epsilon)
