"""Episode runner."""
from reinforcement_learning.markov_decision_process.grid_world.agent_base import AgentBase
from reinforcement_learning.markov_decision_process.grid_world.environment import GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.sarsa_agent import (
    SarsaAgentBase,
)


def run_td_episode(env: GridWorld, agent: AgentBase) -> None:
    """Run an episode for a temporary difference agent in the environment.

    Args:
    ----
        env: The GridWorld environment in which the agent will run.
        agent: The TdAgent.

    Returns:
    -------
        None

    """
    env.reset_agent_state()
    state = env.agent_state
    agent.reset_memory()
    while True:
        action = agent.get_action(state=state)
        result = env.step(action=action)
        agent.add_memory(state=state, action=action, result=result)
        agent.update()
        if result.done:
            break
        state = result.next_state


def run_sarsa_episode(env: GridWorld, agent: SarsaAgentBase) -> None:
    """Run an episode for a SARSA agent in the environment.

    Args:
    ----
        env: The GridWorld environment in which the agent will run.
        agent: The SARSA agent.

    Returns:
    -------
        None

    """
    env.reset_agent_state()
    state = env.agent_state
    agent.reset_memory()
    while True:
        action = agent.get_action(state=state)
        result = env.step(action=action)
        agent.add_memory(state=state, action=action, result=result)
        agent.update()

        if result.done:
            break
        state = result.next_state
    agent.add_memory(state=state, action=None, result=None)
    agent.update()
