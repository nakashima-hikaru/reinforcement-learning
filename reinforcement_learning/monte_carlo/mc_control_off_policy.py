from collections import defaultdict

import numpy as np
from tqdm import tqdm

from reinforcement_learning.dynamic_programming.grid_world import GridWorld, Action, State
from reinforcement_learning.dynamic_programming.policy_eval import Policy, ActionValue
from reinforcement_learning.monte_carlo.mc_eval import random_actions, RandomAgent, greedy_probs


class McOffPolicyAgent:
    def __init__(self, gamma: float, epsilon: float, alpha: float):
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.pi: Policy = defaultdict(lambda: random_actions)
        self.b: Policy = defaultdict(lambda: random_actions)
        self.q: ActionValue = defaultdict(lambda: 0.0)
        self.memory: list[tuple[State, Action, float]] = list()
        self._action_index: list[int] = list(map(int, Action))

    def get_action(self, state: State) -> Action:
        """Gets an action according to its policy `self.pi`."""
        action_probs = self.b[state]
        probs = list(action_probs.values())
        return Action(np.random.choice(self._action_index, p=probs))

    def add(self, state: State, action: Action, reward: float) -> None:
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def update(self) -> None:
        g: float = 0.0
        rho: float = 1.0
        for state, action, reward in reversed(self.memory):
            rho *= self.pi[state][action] / self.b[state][action]
            g = self.gamma * g + reward
            key = state, action
            self.q[key] += (g - self.q[key]) * self.alpha * rho
            self.pi[state] = greedy_probs(q=self.q, state=state, epsilon=0.0)
            self.b[state] = greedy_probs(q=self.q, state=state, epsilon=self.epsilon)


if __name__ == '__main__':
    np.random.seed(314)
    test_map = np.array([[0.0, 0.0, 0.0, 1.0],
                         [0.0, None, 0.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0]], dtype=float)
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = McOffPolicyAgent(gamma=0.9, epsilon=0.01, alpha=0.01)
    n_episodes: int = 1000

    for i_episode in tqdm(range(n_episodes)):
        env.reset()
        agent.reset()
        state = env.agent_state
        while True:
            action = agent.get_action(state=state)
            next_state, reward, done = env.step(action=action)
            agent.add(state=next_state, action=action, reward=reward)
            if done:
                agent.update()
                break

            state = next_state
