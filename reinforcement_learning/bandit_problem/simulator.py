"""Simulate the interaction between a bandit and an agent over a number of steps.

This function simulates a scenario wherein a bandit and an agent interact
over a specified number of steps. It uses the agent's action selection strategy
(as defined in the EpsilonGreedyAgentBase class) to decide which action to take at each step.
The chosen action is then played out on the bandit, and the agent's strategy is updated based on
a reward system (defined in the BanditBase class).
"""
from reinforcement_learning.bandit_problem.agents.base import EpsilonGreedyAgentBase
from reinforcement_learning.bandit_problem.bandits.base import BanditBase


def simulate(
    steps: int,
    bandit: BanditBase,
    agent: EpsilonGreedyAgentBase,
) -> tuple[list[float], list[float]]:
    """Simulate the interaction between a bandit and an agent over a number of steps.

    Args:
    ----
        steps (int): The number of steps to simulate.
        bandit (BanditBase): The bandit to interact with.
        agent (EpsilonGreedyAgentBase): The agent to use for selecting actions.

    Returns:
    -------
        tuple[list[float], list[float]]: A tuple containing two lists. The first list
        contains the total rewards accumulated after each step. The second list contains
        the rewards rate at each step.

    """
    total_reward: float = 0.0
    total_rewards: list[float] = []
    rates: list[float] = []
    for step in range(1, steps + 1):
        i_action = agent.get_action()
        reward = bandit.play(i_arm=i_action)
        agent.update(i_action=i_action, reward=reward)
        total_reward += reward
        total_rewards.append(total_reward)
        rates.append(total_reward / step)
    return total_rewards, rates
