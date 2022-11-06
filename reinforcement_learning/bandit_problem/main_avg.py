# coding=utf-8
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from reinforcement_learning.bandit_problem.agent import Agent, AlphaAgent
from reinforcement_learning.bandit_problem.bandit import Bandit, NonStationaryBandit
from reinforcement_learning.bandit_problem.simulator import simulate


if __name__ == '__main__':
    runs: int = 200
    steps: int = 1000
    epsilon: float = 0.1
    n_arms: int = 10
    all_rates: npt.NDArray[np.float64] = np.zeros((runs, steps))

    for run in range(runs):
        bandit: Bandit = Bandit(n_arms=n_arms)
        agent: Agent = Agent(epsilon=epsilon, action_size=n_arms)
        total_rewards, rates = simulate(steps=steps, bandit=bandit, agent=agent)
        all_rates[run] = rates

    avg_rates = np.mean(all_rates, axis=0)
    plt.xlabel('Steps')
    plt.ylabel('Rates')
    plt.plot(range(steps), avg_rates, label='simple average')

    all_rates_alpha: npt.NDArray[np.float64] = np.zeros((runs, steps))

    for run in range(runs):
        non_stationary_bandit = NonStationaryBandit(n_arms=n_arms)
        alpha_agent: Agent = AlphaAgent(epsilon=epsilon, action_size=n_arms, alpha=0.8)
        total_rewards, rates = simulate(steps=steps, bandit=non_stationary_bandit, agent=alpha_agent)
        all_rates_alpha[run] = rates

    avg_rates_alpha = np.mean(all_rates_alpha, axis=0)
    plt.plot(range(steps), avg_rates_alpha, label='alpha const update')
    plt.legend()
    plt.show()
