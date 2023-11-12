import numpy as np
from tqdm import tqdm

from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.monte_carlo.mc_agent import McMemory
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
        env.reset_agent_state()
        agent.reset_memory()
        state = env.agent_state
        while True:
            action = agent.get_action(state=state)
            result = env.step(action=action)
            memory = McMemory(state=state, action=action, reward=result.reward)
            agent.add_memory(memory=memory)
            if result.done:
                break

            state = result.next_state
        agent.update()


if __name__ == "__main__":
    main()
