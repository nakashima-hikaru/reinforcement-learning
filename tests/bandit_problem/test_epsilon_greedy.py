from reinforcement_learning.bandit_problem.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from reinforcement_learning.bandit_problem.bandits.stationary_bandit import StationaryBandit
from reinforcement_learning.bandit_problem.simulator import simulate


def test_simulate() -> None:
    steps = 1000
    epsilon = 0.1
    n_arms = 10
    bandit = StationaryBandit(n_arms=n_arms)
    agent = EpsilonGreedyAgent(epsilon=epsilon, action_size=n_arms)
    total_rewards, _rates = simulate(steps=steps, bandit=bandit, agent=agent)
    expected = 887.0
    assert total_rewards[-1] == expected
