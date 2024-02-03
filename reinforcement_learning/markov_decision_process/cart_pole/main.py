"""Runs the Deep Q-Network Agent in the CartPole environment.

Two main functions are provided:
- `run_episode` which runs a single episode of the CartPole environment using a given DQNAgent.
- `main_process` which performs the main processing of running episodes in the CartPole environment using a DQNAgent, and returns a list of rewards obtained for each episode.
"""
from typing import cast

import gymnasium as gym
import torch
import tqdm
from gymnasium.envs.classic_control import CartPoleEnv  # type: ignore[attr-defined]

from reinforcement_learning.markov_decision_process.cart_pole.dqn import DQNAgent, Memory


def run_episode(env: CartPoleEnv, agent: DQNAgent, episode: int, sync_interval: int) -> float:
    """Runs a single episode of the environment using the given agent.

    Args:
        env: The environment to run the episode on.
        agent: The agent to use for selecting actions and updating Q-values.
        episode: The current episode number.
        sync_interval: The number of episodes before synchronizing the Q-network.

    Returns:
        The total reward obtained during the episode.
    """
    state, _ = env.reset()
    total_reward = 0.0
    next_states = []
    while True:
        action = agent.get_action(states=torch.from_numpy(state))
        next_state, reward, is_terminated, _truncated, _info = env.step(action=action)  # type: ignore[no-untyped-call]
        reward = cast(float, reward)
        memory = Memory(state=state, action=action, reward=reward, next_state=next_state, is_terminated=is_terminated)
        agent.add_memory(memory=memory)
        agent.update()
        state = next_state
        total_reward += reward
        next_states.append(next_state)
        if is_terminated:
            break
    if episode % sync_interval == 0:
        agent.sync_qnet()
    env.close()  # type: ignore[no-untyped-call]
    return total_reward


def main_process(*, n_episode: int = 300, sync_interval: int = 20, seed: int | None = None) -> list[float]:
    """Performs the main processing of running episodes in the CartPole environment using a DQNAgent.

    Args:
        n_episode (int): The number of episodes to run. Default is 300.
        sync_interval (int): The interval at which to synchronize the agent's target network with the main network. Default is 20.
        seed (int | None): The random seed to use for reproducibility. Default is None.

    Returns:
        reward_history (list[float]): A list of rewards obtained for each episode.
    """
    env = cast(CartPoleEnv, gym.make("CartPole-v1"))
    agent: DQNAgent = DQNAgent(seed=seed)
    reward_history = []
    for episode in tqdm.tqdm(range(n_episode)):
        total_reward = run_episode(env, agent, episode, sync_interval)
        reward_history.append(total_reward)
    return reward_history


if __name__ == "__main__":
    rewards = main_process()
