"""Episode runner."""
from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld


def run_td_episode(*, env: GridWorld, agent: AgentBase, add_goal_state_to_memory: bool) -> None:
    """Run an episode for a temporary difference agent in the environment.

    Args:
    ----
        env: The GridWorld environment in which the agent will run.
        agent: The TdAgent.
        add_goal_state_to_memory: Whether to add goal state to the agent's memory at the end of an episode.

    Returns:
    -------
        None

    """
    env.reset_agent_state()
    agent.reset_memory()
    while True:
        state = env.agent_state
        action = agent.get_action(state=state)
        result = env.step(action=action)
        agent.add_memory(state=state, action=action, result=result)
        agent.update()
        if result.done:
            break
    if add_goal_state_to_memory:
        agent.add_memory(state=state, action=None, result=None)
        agent.update()
