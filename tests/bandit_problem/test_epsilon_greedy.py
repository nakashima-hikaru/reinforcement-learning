from reinforcement_learning.bandit_problem.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from reinforcement_learning.bandit_problem.bandit import Bandit
from reinforcement_learning.bandit_problem.simulator import simulate


def test_end_to_end() -> None:
    steps = 1000
    epsilon = 0.1
    n_arms = 10
    bandit = Bandit(n_arms=n_arms)
    agent = EpsilonGreedyAgent(epsilon=epsilon, action_size=n_arms)
    total_rewards, rates = simulate(steps=steps, bandit=bandit, agent=agent)
    expected = 887.0
    assert total_rewards[-1] == expected
