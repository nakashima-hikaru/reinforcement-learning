"""The Deep Q-Network (DQN) model for an agent.

It incorporates crucial parameters for reinforcement learning, including discount factor, learning rate, exploration rate, buffer size, batch size, and action size.
With replay buffer for storing experiences, this class defines an RL agent with capabilities for model training and action selection based on a Q-network.
"""
import copy
import random
from collections import deque
from collections.abc import Sized
from typing import Final, cast, final

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as f
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
)
from pydantic.dataclasses import dataclass
from torch import Tensor, nn, optim
from torch._dynamo import OptimizedModule

from reinforcement_learning.markov_decision_process.cart_pole.type import State, Tensor1DimFloat32, Tensor1DimInt32, Tensor1DimInt64, Tensor2DimFloat32


@final
@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Memory:
    """Memory class represents a single experience tuple in a memory buffer.

    Attributes:
        state (ObsType): The current state of the environment
        action (StrictInt): The action taken in the current state
        reward (StrictFloat): The reward received for taking the action
        next_state (ObsType): The next state observed after taking the action
        is_terminated (StrictBool): Indicates whether the episode is done after the action
    """

    state: State
    action: StrictInt
    reward: StrictFloat
    next_state: State
    is_terminated: StrictBool


@final
class BatchArray(BaseModel):
    """A batch of data used for training or prediction.

    Attributes:
        states (Tensor2DimFloat32): A 2-dimensional float32 tensor representing the states in the batch.
        actions (Tensor1DimInt64): A 1-dimensional int64 tensor representing the actions taken in the batch.
        rewards (Tensor1DimFloat32): A 1-dimensional float32 tensor representing the rewards received in the batch.
        next_states (Tensor2DimFloat32): A 2-dimensional float32 tensor representing the next states in the batch.
        dones (Tensor1DimInt32): A 1-dimensional int32 tensor representing the done status in the batch.
        model_config (ConfigDict): A configuration dictionary with frozen parameters.

    """

    states: Tensor2DimFloat32
    actions: Tensor1DimInt64
    rewards: Tensor1DimFloat32
    next_states: Tensor2DimFloat32
    dones: Tensor1DimInt32
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


@final
class ReplayBuffer(Sized):
    """A replay buffer used in reinforcement learning algorithms.

    Attributes:
        - __buffer: A deque data structure used to store the replay memories.
        - __batch_size: An integer representing the size of each batch during training.

    Methods:
        - __len__(self) -> int:
            Returns the size of the buffer.

        - __init__(self, *, buffer_size: int, batch_size: int):
            Initializes a new instance of the ReplayBuffer class.

    Args:
                - buffer_size: An integer representing the maximum size of the buffer.
                - batch_size: An integer representing the size of each batch during training.

        - add(self, *, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
            Adds a new memory to the buffer.

    Args:
                - state: A numpy array representing the current state.
                - action: An integer representing the action taken in the current state.
                - reward: A float representing the reward received for taking the action.
                - next_state: A numpy array representing the next state after taking the action.
                - done: A boolean indicating whether the episode is done after taking the action.

        - get_batch(self) -> BatchArray:
            Returns a batch of memories randomly sampled from the buffer.

    Returns:
                - A BatchArray object containing the batched data (states, actions, rewards, next_states, dones).
    """

    def __len__(self) -> int:
        """Returns the length of the buffer."""
        return len(self.__buffer)

    def __init__(self, *, buffer_size: int, batch_size: int):
        """Initializes an instance.

        Args:
        buffer_size: The maximum number of items to keep in the buffer.
        batch_size: The number of items to include in each batch.

        """
        self.__buffer: deque[Memory] = deque(maxlen=buffer_size)
        self.__batch_size: int = batch_size

    def add(self, *, memory: Memory) -> None:
        """Adds a new memory to the memory buffer."""
        self.__buffer.append(memory)

    def get_batch(self) -> BatchArray:
        """Samples a batch of data from the memory buffer.

        Returns:
            BatchArray: A batch of data containing states, actions, rewards, next states, and dones.
        """
        data: list[Memory] = random.sample(list(self.__buffer), self.__batch_size)

        states: npt.NDArray[np.float32] = np.stack([x.state for x in data])
        actions: npt.NDArray[np.int64] = np.array([x.action for x in data])
        rewards: npt.NDArray[np.float32] = np.array([x.reward for x in data]).astype(np.float32)
        next_states: npt.NDArray[np.float32] = np.stack([x.next_state for x in data])
        dones: npt.NDArray[np.int32] = np.array([x.is_terminated for x in data]).astype(np.int32)
        return BatchArray(
            states=torch.from_numpy(states),
            actions=torch.from_numpy(actions),
            rewards=torch.from_numpy(rewards),
            next_states=torch.from_numpy(next_states),
            dones=torch.from_numpy(dones),
        )


class QNet(nn.Module):
    """A neural network model for Q-learning.

    Attributes:
        l1 (torch.nn.Linear): The first linear layer of the network
        l2 (torch.nn.Linear): The second linear layer of the network
        l3 (torch.nn.Linear): The output linear layer of the network

    Methods:
        __init__(self, action_size: int)
            Initializes the QNet object.

        forward(self, x: np.ndarray[np.float32]) -> Tensor
            Performs forward propagation on the input.

    """

    def __init__(self, *, action_size: int):
        """Initializes an instance.

        Args:
            action_size (int): The number of possible actions in the environment.
        """
        super().__init__()
        self.l1: nn.Linear = nn.Linear(in_features=4, out_features=128)
        self.l2: nn.Linear = nn.Linear(in_features=128, out_features=128)
        self.l3: nn.Linear = nn.Linear(in_features=128, out_features=action_size)

    def forward(self, x: npt.NDArray[np.float32]) -> Tensor:
        """Returns the result of forward propagation.

        Args:
            x (npt.NDArray[np.float32]): The input array.
        """
        x1 = f.relu(self.l1(x))
        x2 = f.relu(self.l2(x1))
        return cast(Tensor, self.l3(x2))


class DQNAgent:
    """A Deep Q-Network (DQN) agent. It is responsible for training and selecting actions based on a Q-network.

    Attributes:
        __gamma (float): The discount factor for future rewards.
        __learning_rate (float): The learning rate for the optimizer.
        __epsilon (float): The exploration rate for selecting random actions.
        __buffer_size (int): The maximum size of the replay buffer.
        __batch_size (Final[int]): The size of each batch for training.
        __action_size (Final[int]): The number of possible actions.
        __replay_buffer (ReplayBuffer[ObsType]): The replay buffer for storing experiences.
        __qnet (OptimizedModule): The Q-network used for training and selecting actions.
        __qnet_target (OptimizedModule): A target Q-network for stability during training.
        __optimizer (optim.Optimizer): The optimizer used for updating the Q-network.
        __rng (np.random.Generator): A random number generator for exploration.

    Methods:
        __init__(self, *, seed: int | None = None)
            Initializes a new instance of the DQNAgent class.

        sync_qnet(self) -> None
    Creates a deep copy of the QNet and assigns it to the QNet target
    """

    def __init__(self, *, seed: int | None = None):
        """Initializes an instance of the class.

        Args:
            seed: An optional integer seed used for random number generation. Defaults to None.

        """
        self.__gamma: float = 0.98
        self.__learning_rate: float = 0.0005
        self.__epsilon: float = 0.1
        self.__buffer_size: int = 10000
        self.__batch_size: Final[int] = 32
        self.__action_size: Final[int] = 2
        self.__replay_buffer: ReplayBuffer = ReplayBuffer(buffer_size=self.__buffer_size, batch_size=self.__batch_size)
        self.__qnet: OptimizedModule = cast(OptimizedModule, torch.compile(QNet(action_size=self.__action_size)))
        self.__qnet_target: OptimizedModule = cast(OptimizedModule, torch.compile(QNet(action_size=self.__action_size)))
        self.__optimizer: optim.Optimizer = optim.Adam(lr=self.__learning_rate, params=self.__qnet.parameters())
        self.__rng: np.random.Generator = np.random.default_rng(seed=seed)

    def sync_qnet(self) -> None:
        """Creates a deep copy of the QNet and assigns it to the QNet target."""
        self.__qnet_target = copy.deepcopy(self.__qnet)

    def add_memory(self, *, memory: Memory) -> None:
        """Adds a memory to the replay buffer.

        Args:
            memory: The memory to be added to the replay buffer.
        """
        self.__replay_buffer.add(memory=memory)

    def get_action(self, states: Tensor) -> int:
        """Returns the action to be taken based on the given states.

        Args:
            states (Tensor): The input states.

        Returns:
            int: The action to be taken.
        """
        if self.__rng.random() < self.__epsilon:
            return self.__rng.choice(self.__action_size)
        qs: Tensor = self.__qnet(states.unsqueeze(dim=0))
        return cast(int, torch.argmax(input=qs).item())

    def update(self) -> None:
        """Updates the Q-network by performing a single optimization step on a batch of experiences from the replay buffer."""
        if len(self.__replay_buffer) < self.__batch_size:
            return
        batch = self.__replay_buffer.get_batch()
        qs = self.__qnet(batch.states)  # [N, 2]
        q = qs[np.arange(self.__batch_size), batch.actions]  # [B, 2]

        next_qs = self.__qnet_target(batch.next_states)  # [N, 2]
        next_q = next_qs.max(axis=1).values  # [N, 1]
        next_q.detach()
        target = batch.rewards + (1 - batch.dones) * self.__gamma * next_q
        loss = nn.MSELoss()
        output = loss(q, target)
        self.__qnet.zero_grad()
        output.backward()
        self.__optimizer.step()
