import logging
from typing import Final

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.dynamic_programming.policy_iter import (
    policy_iter,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    pi = policy_iter(env=env, gamma=0.9, threshold=1e-3)
    max_prob: Final[float] = 1.0
    for state in pi:
        for action in pi[state]:
            if pi[state][action] == max_prob:
                msg = f"{state, action=}"
                logging.info(msg)


if __name__ == "__main__":
    main()
