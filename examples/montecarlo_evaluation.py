import logging

import numpy as np

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_eval import RandomAgent


def main() -> None:
    """Run the evaluation of a random agent in a grid world environment."""
    logging.basicConfig(level=logging.INFO)
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = RandomAgent(gamma=0.9)
    n_episodes: int = 1000

    for _ in range(n_episodes):
        env.reset_agent_state()
        state = env.agent_state
        agent.reset_memory()

        while True:
            action = agent.get_action(state=state)
            result = env.step(action=action)
            agent.add_memory(state=state, action=action, result=result)
            if result.done:
                break

            state = result.next_state
        agent.update()

    logging.info(agent.state_value)


if __name__ == "__main__":
    main()
