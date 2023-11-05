import logging

from matplotlib import pyplot as plt

from reinforcement_learning.bandit_problem.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from reinforcement_learning.bandit_problem.bandit import Bandit
from reinforcement_learning.bandit_problem.simulator import simulate


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    steps: int = 1000
    epsilon: float = 0.1
    n_arms: int = 10
    bandit: Bandit = Bandit(n_arms=n_arms)
    agent: EpsilonGreedyAgent = EpsilonGreedyAgent(epsilon=epsilon, action_size=n_arms)
    total_rewards, rates = simulate(steps=steps, bandit=bandit, agent=agent)
    logging.info(total_rewards[-1])

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle("Transition of learning")
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Transition of Reward")
    ax1.plot(range(steps), total_rewards)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Total Reward")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Transition of rate")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Rates")
    ax2.plot(range(steps), rates)
    plt.show()


if __name__ == "__main__":
    main()
