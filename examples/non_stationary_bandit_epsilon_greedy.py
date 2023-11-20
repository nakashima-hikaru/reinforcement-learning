from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from reinforcement_learning.markov_decision_process.bandit_problem.agents.alpha_epsilon_greedy_agent import (
    AlphaEpsilonGreedyAgent,
)
from reinforcement_learning.markov_decision_process.bandit_problem.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from reinforcement_learning.markov_decision_process.bandit_problem.bandits.non_stationary_bandit import (
    NonStationaryBandit,
)
from reinforcement_learning.markov_decision_process.bandit_problem.simulator import simulate

if TYPE_CHECKING:
    import numpy.typing as npt


def main() -> None:
    runs: int = 200
    steps: int = 1000
    epsilon: float = 0.1
    n_arms: int = 10
    all_rates: npt.NDArray[np.float64] = np.zeros((runs, steps), dtype=np.float64)

    for run in range(runs):
        bandit = NonStationaryBandit(n_arms=n_arms)
        agent = EpsilonGreedyAgent(
            epsilon=epsilon,
            action_size=n_arms,
        )
        total_rewards, rates = simulate(steps=steps, bandit=bandit, agent=agent)
        all_rates[run] = rates

    avg_rates = np.mean(all_rates, axis=0)
    plt.xlabel("Steps")
    plt.ylabel("Rates")
    plt.plot(range(steps), avg_rates, label="simple average")

    all_rates_alpha = np.zeros((runs, steps), dtype=np.float64)

    for run in range(runs):
        non_stationary_bandit = NonStationaryBandit(n_arms=n_arms)
        alpha_agent = AlphaEpsilonGreedyAgent(
            epsilon=epsilon,
            action_size=n_arms,
            alpha=0.8,
        )
        _, rates = simulate(
            steps=steps,
            bandit=non_stationary_bandit,
            agent=alpha_agent,
        )
        all_rates_alpha[run] = rates

    avg_rates_alpha = np.mean(all_rates_alpha, axis=0)
    plt.plot(range(steps), avg_rates_alpha, label="alpha const update")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
