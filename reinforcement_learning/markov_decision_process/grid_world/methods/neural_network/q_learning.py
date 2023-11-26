"""Classes and methods for reinforcement learning in a grid world environment."""
from collections import defaultdict
from types import MappingProxyType
from typing import Self, cast

import torch
import torch.nn.functional as f
from torch import Tensor, nn, optim

from reinforcement_learning.errors import InvalidMemoryError, NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionResult,
    GridWorld,
    ReadOnlyActionValue,
    State,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.q_learning import (
    QLearningMemory,
)


class Qnet(nn.Module):
    """A Q-network for reinforcement learning in a grid world environment."""

    def __init__(self: Self) -> None:
        """Initialize a Qnet object."""
        super().__init__()
        self.l1: nn.Linear = nn.Linear(in_features=12, out_features=128)
        self.l2: nn.Linear = nn.Linear(in_features=128, out_features=len(Action))

    def forward(self: Self, x: Tensor) -> Tensor:
        """Perform the forward pass of the Qnet neural network.

        Args:
            x: A torch.Tensor representing the input to the network.
        """
        return cast(Tensor, self.l2(f.relu(self.l1(x))))


class QLearningAgent(AgentBase):
    """A Q-learning algorithm for reinforcement learning in a GridWorld environment.

    Attributes:
    ----------
    - __gamma (float): Discount factor for future rewards.
    - __lr (float): Learning rate for optimization.
    - __epsilon (float): Exploration rate for epsilon-greedy policy.
    - __q_net (nn.Module): Q-network, a neural network that estimates action values.
    - __optimizer (optim.Optimizer): Optimizer for updating the Q-network.
    - __env (GridWorld): GridWorld environment.
    - __memory (QLearningMemory | None): Memory to store agent's experiences.
    - __total_loss (float): Total loss accumulated during training.
    - __count (int): Count of training steps performed.

    Methods:
    -------
    - average_loss(self: Self) -> float:
        Returns the average loss per training step.

    - action_value(self: Self) -> ReadOnlyActionValue:
        Returns a read-only mapping of state-action pairs to their estimated action values.

    - get_action(self: Self, *, state: State) -> Action:
        Selects an action to take given the current state based on the epsilon-greedy policy.

    - add_memory(self: Self, *, state: State, action: Action | None, result: ActionResult | None) -> None:
        Adds a new experience to the agent's memory.

    - reset_memory(self: Self) -> None:
        Resets the agent's memory.

    - update(self: Self) -> None:
        Performs a single update step of the Q-learning algorithm.

    """

    def __init__(self: Self, *, seed: int | None, env: GridWorld):
        """Initialize an agent.

        Args:
            seed: An integer specifying the seed value for random number generation. If None, no seed is set.
            env: A GridWorld instance representing the environment in which the agent will operate.
        """
        super().__init__(seed=seed)
        self.__gamma: float = 0.9
        self.__lr: float = 0.01
        self.__epsilon: float = 0.1
        self.__q_net: nn.Module = Qnet()
        self.__optimizer: optim.Optimizer = optim.SGD(lr=self.__lr, params=self.__q_net.parameters())
        self.__env: GridWorld = env
        self.__memory: QLearningMemory | None = None
        self.__total_loss: float = 0.0
        self.__count: int = 0

    @property
    def average_loss(self: Self) -> float:
        """Calculate the average loss of the QLearningAgent."""
        return self.__total_loss / self.__count

    @property
    def action_value(self: Self) -> ReadOnlyActionValue:
        """Return a readonly action value map for the agent."""
        ret: defaultdict[tuple[State, Action], float] = defaultdict()
        with torch.set_grad_enabled(mode=False):
            for state in self.__env.state():
                for action in Action:
                    ret[state, action] = float(
                        self.__q_net(self.__env.convert_to_one_hot(state=state))[:, action.value]
                    )
        return MappingProxyType(ret)

    def get_action(self: Self, *, state: State) -> Action:
        """Select an action based on `self.rng`."""
        if self.rng.random() < self.__epsilon:
            return Action(self.rng.choice(list(Action)))

        return Action(torch.argmax(self.__q_net(self.__env.convert_to_one_hot(state=state)), dim=1)[0].item())

    def add_memory(self: Self, *, state: State, action: Action | None, result: ActionResult | None) -> None:
        """Add a new experience into the memory.

        Args:
        ----
            state: The current state of the agent.
            action: The action taken by the agent.
            result: The result of the action taken by the agent.
        """
        if action is None or result is None:
            if action is None and result is None:
                message = "action or result must not be None"
            elif action is None:
                message = "action must not be None"
            else:
                message = "result must not be None"
            raise InvalidMemoryError(message)
        self.__memory = QLearningMemory(
            state=state, action=action, reward=result.reward, next_state=result.next_state, done=result.done
        )

    def reset_memory(self: Self) -> None:
        """Reset the agent's memory."""
        self.__memory = None
        self.__total_loss = 0.0
        self.__count = 0

    def update(self: Self) -> None:
        """Updates the Q-values in the Q-learning agent based on the current memory."""
        if self.__memory is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="__memory")
        if self.__memory.done:
            next_action_value = torch.zeros(size=[1])
        else:
            next_action_value = torch.max(  # noqa: PD011
                self.__q_net(self.__env.convert_to_one_hot(state=self.__memory.next_state)),
                dim=1,
            ).values
        target = self.__gamma * next_action_value + self.__memory.reward
        current_action_values = self.__q_net(self.__env.convert_to_one_hot(state=self.__memory.state))
        current_action_value = current_action_values[:, self.__memory.action.value]
        loss = nn.MSELoss()
        output = loss(target, current_action_value)
        self.__total_loss += float(output)
        self.__count += 1
        self.__q_net.zero_grad()
        output.backward()
        self.__optimizer.step()
