from reinforcement_learning.markov_decision_process.bandit_problem.agents.alpha_epsilon_greedy_agent import (
    AlphaEpsilonGreedyAgent,
)
from reinforcement_learning.markov_decision_process.bandit_problem.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from reinforcement_learning.markov_decision_process.bandit_problem.bandits.non_stationary_bandit import (
    NonStationaryBandit,
)
from reinforcement_learning.markov_decision_process.bandit_problem.bandits.stationary_bandit import StationaryBandit
from reinforcement_learning.markov_decision_process.bandit_problem.simulator import simulate


def test_epsilon_greedy_stationary() -> None:
    steps = 1000
    epsilon = 0.1
    n_arms = 10
    bandit = StationaryBandit(n_arms=n_arms, seed=0)
    agent = EpsilonGreedyAgent(epsilon=epsilon, action_size=n_arms, seed=0)
    total_rewards, _rates = simulate(steps=steps, bandit=bandit, agent=agent)
    expected = 887.0
    assert total_rewards[-1] == expected


def test_epsilon_greedy_non_stationary() -> None:
    steps = 1000
    epsilon = 0.1
    n_arms = 10
    bandit = NonStationaryBandit(n_arms=n_arms, seed=0)
    agent = EpsilonGreedyAgent(epsilon=epsilon, action_size=n_arms, seed=0)
    total_rewards, _rates = simulate(steps=steps, bandit=bandit, agent=agent)
    expected = 674.0
    assert total_rewards[-1] == expected


def test_alpha_epsilon_greedy_stationary() -> None:
    steps = 1000
    epsilon = 0.1
    n_arms = 10
    bandit = StationaryBandit(n_arms=n_arms, seed=0)
    agent = AlphaEpsilonGreedyAgent(epsilon=epsilon, action_size=n_arms, alpha=0.1, seed=0)
    total_rewards, _rates = simulate(steps=steps, bandit=bandit, agent=agent)
    expected = 740.0
    assert total_rewards[-1] == expected


def test_alpha_epsilon_greedy_non_stationary() -> None:
    steps = 1000
    epsilon = 0.1
    n_arms = 10
    bandit = NonStationaryBandit(n_arms=n_arms, seed=0)
    agent = AlphaEpsilonGreedyAgent(epsilon=epsilon, action_size=n_arms, alpha=0.1, seed=0)
    total_rewards, _rates = simulate(steps=steps, bandit=bandit, agent=agent)
    expected = 800.0
    assert total_rewards[-1] == expected
