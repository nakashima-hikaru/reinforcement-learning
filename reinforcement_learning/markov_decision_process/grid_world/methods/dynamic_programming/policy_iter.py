"""functions for computing and improving policies based on the state values in a grid world environment."""
from collections import defaultdict

from reinforcement_learning.markov_decision_process.grid_world.environment import Action, GridWorld, Policy, StateValue
from reinforcement_learning.markov_decision_process.grid_world.methods.dynamic_programming.policy_eval import (
    policy_eval,
)
from reinforcement_learning.util import argmax


def greedy_policy(*, v: StateValue, env: GridWorld, gamma: float) -> Policy:
    """Compute the greedy policy based on the given state values.

    Args:
    ----
        v: a dictionary representing the state values of each state in the environment
        env: a GridWorld object representing the environment
        gamma: a float representing the discount factor for future rewards

    Returns:
    -------
        pi: a Policy object representing the greedy policy based on the given state values
    """
    pi: Policy = Policy()
    for state in env.states():
        action_values: dict[Action, float] = {}
        for action in Action:
            next_state = env.next_state(state=state, action=action)
            reward = env.reward(next_state=next_state)
            value = reward + gamma * v[next_state]
            action_values[action] = value
            max_action = argmax(action_values)
            action_probs: dict[Action, float] = {
                Action.UP: 0.0,
                Action.DOWN: 0.0,
                Action.LEFT: 0.0,
                Action.RIGHT: 0.0,
            }
            action_probs[max_action] = 1.0
            pi[state] = action_probs
    return pi


def policy_iter(*, env: GridWorld, gamma: float, threshold: float) -> Policy:
    """Improve policy through iteration.

    Args:
    ----
        env (GridWorld): The grid world environment.
        gamma (float): The discount factor for future rewards.
        threshold (float): The convergence threshold for the value iteration.

    Returns:
    -------
        Policy: The optimal policy for the given grid world environment.
    """
    pi: Policy = defaultdict(
        lambda: {
            Action.UP: 0.25,
            Action.DOWN: 0.25,
            Action.LEFT: 0.25,
            Action.RIGHT: 0.25,
        },
    )
    v: StateValue = defaultdict(lambda: 0)
    while True:
        v = policy_eval(pi=pi, v=v, env=env, gamma=gamma, threshold=threshold)
        new_pi = greedy_policy(v=v, env=env, gamma=gamma)
        if new_pi == pi:
            break
        pi = new_pi
    return pi
