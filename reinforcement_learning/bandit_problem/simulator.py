# coding=utf-8
from reinforcement_learning.bandit_problem.agent import Agent
from reinforcement_learning.bandit_problem.bandit import Bandit


def simulate(steps: int, bandit: Bandit, agent: Agent) -> tuple[list[float], list[float]]:
    total_reward: float = 0.0
    total_rewards: list[float] = []
    rates: list[float] = []
    for step in range(1, steps + 1):
        i_action = agent.get_action()
        reward = bandit.play(i_action)
        agent.update(i_action=i_action, reward=reward)
        total_reward += reward
        total_rewards.append(total_reward)
        rates.append(total_reward / step)
    return total_rewards, rates
