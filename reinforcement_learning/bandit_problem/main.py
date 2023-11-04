import logging

from matplotlib import pyplot as plt

from reinforcement_learning.bandit_problem.agent import EpsilonGreedyAgent
from reinforcement_learning.bandit_problem.bandit import Bandit
from reinforcement_learning.bandit_problem.simulator import simulate

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    steps: int = 1000
    epsilon: float = 0.1
    n_arms: int = 10
    bandit: Bandit = Bandit(n_arms=n_arms)
    agent: EpsilonGreedyAgent = EpsilonGreedyAgent(epsilon=epsilon, action_size=n_arms)
    total_rewards, rates = simulate(steps=steps, bandit=bandit, agent=agent)
    logging.info(total_reward := total_rewards[-1])

    plt.xlabel("Steps")
    plt.ylabel("Total Reward")
    plt.plot(range(steps), total_rewards)
    plt.show()

    plt.xlabel("Steps")
    plt.ylabel("Rates")
    plt.plot(range(steps), rates)
    plt.show()
