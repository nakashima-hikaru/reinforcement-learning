import logging

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.agent_episodes import (
    run_td_episode,
)
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.td_eval import (
    TdAgent,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = TdAgent(gamma=0.9, alpha=0.01)
    n_episodes: int = 1000
    for _ in range(n_episodes):
        run_td_episode(env=env, agent=agent)
    logging.info(agent.v)


if __name__ == "__main__":
    main()
