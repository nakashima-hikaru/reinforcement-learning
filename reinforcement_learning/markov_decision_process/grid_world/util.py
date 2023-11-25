from reinforcement_learning.markov_decision_process.grid_world.environment import Action, State
from reinforcement_learning.util import argmax


def greedy_probs(
    *,
    q: dict[tuple[State, Action], float],
    state: State,
    epsilon: float,
) -> dict[Action, float]:
    """Compute the epsilon-greedy action probabilities for the given state.

    Args:
    ----
        q: A dictionary mapping (state, action) pairs to their respective value.
        state: The current state of the system.
        epsilon: The factor determining the trade-off between exploration and exploitation.

    Returns:
    -------
        A dictionary mapping actions to their epsilon-greedy probability.
    """
    qs = {}
    for action in Action:
        qs[action] = q[(state, action)]
    max_action = argmax(qs)
    base_prob = epsilon / len(Action)
    action_probs: dict[Action, float] = {action: base_prob for action in Action}
    action_probs[max_action] += 1.0 - epsilon
    return action_probs
