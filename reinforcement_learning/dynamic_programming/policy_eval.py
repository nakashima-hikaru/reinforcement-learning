from collections import defaultdict
from typing import TypeAlias, cast

from reinforcement_learning.dynamic_programming.grid_world import Action, GridWorld, State

Policy: TypeAlias = defaultdict[State, dict[Action, float]]
StateValue: TypeAlias = defaultdict[State, float]
ActionValue: TypeAlias = defaultdict[tuple[State, Action], float]


def eval_one_step(pi: Policy, v: StateValue, env: GridWorld, gamma: float) -> StateValue:
    for state in env.states():
        if state == env.goal_state:
            v[state] = 0.0
            continue
        action_probs = pi[state]
        new_v: float = 0.0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state=state, action=action)
            reward = env.reward(next_state)
            new_v += action_prob * (reward + gamma * v[next_state])
        v[state] = new_v
    return v


def policy_eval(pi: Policy, v: StateValue, env: GridWorld, gamma: float, threshold: float) -> StateValue:
    while True:
        old_v = v.copy()
        v = eval_one_step(pi, v, env, gamma)
        delta = 0.0  # the maximum delta of the state value
        for state in env.states():
            t = abs(v[state] - old_v[state])
            if delta < t:
                delta = t
        if delta < threshold:
            break
    return v
