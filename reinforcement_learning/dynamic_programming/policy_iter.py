from collections import defaultdict
from typing import TypeVar

from reinforcement_learning.dynamic_programming.grid_world import GridWorld, Action
from reinforcement_learning.dynamic_programming.policy_eval import StateValue, Policy, policy_eval

T = TypeVar("T")


def argmax(d: dict[T, float]) -> T:
    return max(d, key=lambda key: d[key])


def greedy_policy(v: StateValue, env: GridWorld, gamma: float) -> Policy:
    pi: Policy = Policy()
    for state in env.states():
        action_values: dict[Action, float] = {}
        for action in Action:
            next_state = env.next_state(state=state, action=action)
            reward = env.reward(next_state=next_state)
            value = reward + gamma * v[next_state]
            action_values[action] = value
            max_action = argmax(action_values)
            action_probs: dict[Action, float] = {Action.UP: 0.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0}
            action_probs[max_action] = 1.0
            pi[state] = action_probs
    return pi


def policy_iter(env: GridWorld, gamma: float, threshold: float) -> Policy:
    pi: Policy = defaultdict(lambda: {Action.UP: 0.25, Action.DOWN: 0.25, Action.LEFT: 0.25, Action.RIGHT: 0.25})
    v: StateValue = defaultdict(lambda: 0)
    while True:
        v = policy_eval(pi=pi, v=v, env=env, gamma=gamma, threshold=threshold)
        new_pi: Policy = greedy_policy(v=v, env=env, gamma=gamma)
        if new_pi == pi:
            break
        pi = new_pi
    return pi


if __name__ == '__main__':
    import numpy as np

    test_map = np.array([[0.0, 0.0, 0.0, 1.0],
                         [0.0, None, 0.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0]], dtype=float)
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    pi: Policy = policy_iter(env=env, gamma=0.9, threshold=1e-3)
    for state in pi.keys():
        for action in pi[state]:
            if pi[state][action] == 1.0:
                print(f'{state, action=}')
