"""functions for computing and improving policies based on the state values in a grid world environment."""
import logging
from collections import defaultdict
from typing import Final, TypeVar

from reinforcement_learning.dynamic_programming.grid_world import Action, GridWorld
from reinforcement_learning.dynamic_programming.policy_eval import (
    Policy,
    StateValue,
    policy_eval,
)

T = TypeVar("T")


def argmax(d: dict[T, float]) -> T:
    """Find the key with the highest value.

    Args:
    ----
        d: A dictionary with keys of type T and values of type float.

    Returns:
    -------
        T: The key from the dictionary with the highest corresponding value.

    """
    return max(d, key=lambda key: d[key])


def greedy_policy(v: StateValue, env: GridWorld, gamma: float) -> Policy:
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


def policy_iter(env: GridWorld, gamma: float, threshold: float) -> Policy:
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
        new_pi: Policy = greedy_policy(v=v, env=env, gamma=gamma)
        if new_pi == pi:
            break
        pi = new_pi
    return pi


if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    pi: Policy = policy_iter(env=env, gamma=0.9, threshold=1e-3)
    max_prob: Final[float] = 1.0
    for state in pi:
        for action in pi[state]:
            if pi[state][action] == max_prob:
                msg = f"{state, action=}"
                logging.info(msg)
