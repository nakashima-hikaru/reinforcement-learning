"""Episode runner."""
from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.td_eval import (
    TdAgent,
    TdMemory,
)


def run_agent_episode(env: GridWorld, agent: TdAgent) -> None:
    """Run an episode for the given agent in the environment.

    Args:
    ----
        env: The GridWorld environment in which the agent will run.
        agent: The TdAgent object representing the agent.

    Returns:
    -------
        None

    """
    env.reset_agent_state()
    state = env.agent_state
    agent.reset_memory()
    while True:
        action = agent.get_action(state)
        result = env.step(action=action)
        memory = TdMemory(state=state, reward=result.reward, next_state=result.next_state, done=result.done)
        agent.add_memory(memory=memory)
        agent.update()
        if result.done:
            break
        state = result.next_state
