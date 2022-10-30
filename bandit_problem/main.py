# coding=utf-8
import matplotlib.pyplot as plt

from bandit_problem.agent import Agent
from bandit_problem.bandit import Bandit
from bandit_problem.simulator import simulate

if __name__ == '__main__':
    steps: int = 1000
    epsilon: float = 0.1
    n_arms: int = 10
    bandit: Bandit = Bandit(n_arms=n_arms)
    agent: Agent = Agent(epsilon=epsilon, action_size=n_arms)
    total_rewards, rates = simulate(steps=steps, bandit=bandit, agent=agent)
    print(total_reward := total_rewards[-1])

    plt.xlabel('Steps')
    plt.ylabel('Total Reward')
    plt.plot(range(steps), total_rewards)
    plt.show()

    plt.xlabel('Steps')
    plt.ylabel('Rates')
    plt.plot(range(steps), rates)
    plt.show()
