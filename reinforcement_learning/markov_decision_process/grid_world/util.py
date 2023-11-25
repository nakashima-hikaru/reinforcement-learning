"""Utility functions for grid-world problem."""
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, State
from reinforcement_learning.util import argmax


def greedy_probs(
    *,
    q: dict[tuple[State, Action], float],
    state: State,
    epsilon: float,
) -> dict[Action, float]:
    """Generate the probabilities of actions where each is chosen by a greedy approach.

    Args:
    ----
        q: A dictionary mapping pairs of state and action to a float value, representing Q-value function.
        state: The current state for which the action probabilities are to be generated.
        epsilon: The factor determining the trade-off between exploration and exploitation.

    Returns:
    -------
        action_probs: A dictionary where keys represent actions and values represent the probability for each action according to epsilon-greedy approach.
    """
    qs = {}
    for action in Action:
        qs[action] = q[(state, action)]
    max_action = argmax(qs)
    base_prob = epsilon / len(Action)
    action_probs: dict[Action, float] = {action: base_prob for action in Action}
    action_probs[max_action] += 1.0 - epsilon
    return action_probs
