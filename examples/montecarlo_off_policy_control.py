import numpy as np
from tqdm import tqdm

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_control_off_policy import (
    McOffPolicyAgent,
)


def main() -> None:
    np.random.default_rng(314)
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = McOffPolicyAgent(gamma=0.9, epsilon=0.01, alpha=0.01)
    n_episodes: int = 1000

    for _i_episode in tqdm(range(n_episodes)):
        env.reset()
        agent.reset()
        state = env.agent_state
        while True:
            action = agent.get_action(state=state)
            next_state, reward, done = env.step(action=action)
            agent.add(state=next_state, action=action, reward=reward)
            if done:
                agent.update()
                break

            state = next_state


if __name__ == "__main__":
    main()
