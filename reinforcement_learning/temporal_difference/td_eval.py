import numpy as np
from collections import defaultdict

from reinforcement_learning.dynamic_programming.grid_world import State, Action, GridWorld
from reinforcement_learning.dynamic_programming.policy_eval import Policy, ActionValue, StateValue
from reinforcement_learning.monte_carlo.mc_eval import random_actions, McAgent


class TdAgent:
    def __init__(self, gamma: float, alpha: float):
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.pi: Policy = defaultdict(lambda: random_actions)
        self.v: StateValue = defaultdict(lambda: 0.0)
        self._action_index: list[int] = list(map(int, Action))

    def get_action(self, state: State) -> Action:
        """Gets an action according to its policy `self.pi`."""
        action_probs = self.pi[state]
        probs = list(action_probs.values())
        return Action(np.random.choice(self._action_index, p=probs))

    def eval(self, state: State, reward: float, next_state: State, done: bool) -> None:
        next_v: float = 0.0 if done else self.v[next_state]
        target: float = reward + self.gamma * next_v
        self.v[state] += (target - self.v[state]) * self.alpha


if __name__ == '__main__':
    np.random.seed(314)
    test_map = np.array([[0.0, 0.0, 0.0, 1.0],
                         [0.0, None, 0.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0]], dtype=float)
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = TdAgent(gamma=0.9, alpha=0.01)
    n_episodes: int = 1000
    for i_episode in range(n_episodes):
        env.reset()
        state = env.agent_state
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action=action)
            agent.eval(state=state, reward=reward, next_state=next_state, done=done)
            if done:
                break
            state = next_state
    print(agent.v)
