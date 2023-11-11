import logging

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.td_eval import TdAgent


def main() -> None:
    np.random.default_rng(314)
    logging.basicConfig(level=logging.INFO)
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = TdAgent(gamma=0.9, alpha=0.01)
    n_episodes: int = 1000
    for _ in range(n_episodes):
        env.reset()
        state = env.agent_state
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action=action)
            agent.evaluate(state=state, reward=reward, next_state=next_state, done=done)
            if done:
                break
            state = next_state
    logging.info(agent.v)


if __name__ == "__main__":
    main()
