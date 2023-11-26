"""Module for reinforcement learning using Dynamic Programming techniques.

This module implements the policy evaluation and one-step evaluation methods for Value Iteration.

Value iteration involves two steps: policy evaluation and policy improvement.
In this module, we implement policy evaluation, where the value for each state
is iteratively updated, and one-step evaluation, that contributes to the update
in value_iteration, for a deterministic policy in a GridWorld environment.

The implemented methods can be used as building blocks for more complex
reinforcement learning algorithms in GridWorld type environments.
"""
from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld, Policy, StateValue


def eval_one_step(
    *,
    pi: Policy,
    v: StateValue,
    env: GridWorld,
    gamma: float,
) -> StateValue:
    """Evaluate one step of the Value Iteration algorithm.

    Args:
    ----
        pi: The policy, represented as a dictionary mapping states to action probability distributions.
        v: The current state value function, represented as a dictionary mapping states to values.
        env: The GridWorld environment.
        gamma: The discount factor.

    Returns:
    -------
        The updated state value function after one step of evaluation.
    """
    for state in env.states():
        if state == env.goal_state:
            v[state] = 0.0
            continue
        action_probs = pi[state]
        new_v: float = 0.0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state=state, action=action)
            reward = env.reward(next_state=next_state)
            new_v += action_prob * (reward + gamma * v[next_state])
        v[state] = new_v
    return v


def policy_eval(
    *,
    pi: Policy,
    v: StateValue,
    env: GridWorld,
    gamma: float,
    threshold: float,
) -> StateValue:
    """Perform policy evaluation for a given policy in a grid world environment using the iterative approach.

    Args:
    ----
        pi: The policy to be evaluated.
        v: The initial state values.
        env: The environment (grid world) in which the policy is to be evaluated.
        gamma: The discount factor for future rewards.
        threshold: The threshold for convergence.

    Returns:
    -------
        The updated state values after policy evaluation.
    """
    while True:
        old_v = v.copy()
        v = eval_one_step(pi=pi, v=v, env=env, gamma=gamma)
        delta = max(abs(v[state] - old_v[state]) for state in env.states())
        if delta < threshold:
            break
    return v
