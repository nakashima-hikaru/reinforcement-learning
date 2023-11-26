import numpy as np
import pytest

from reinforcement_learning.errors import NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.environment import Action, GridWorld
from reinforcement_learning.markov_decision_process.grid_world.methods.neural_network.q_learning import QLearningAgent
from reinforcement_learning.markov_decision_process.grid_world.methods.temporal_difference.agent_episodes import (
    run_td_episode,
)


def test_update_with_empty_memory(mocker: GridWorld) -> None:
    agent = QLearningAgent(seed=0, env=mocker)
    with pytest.raises(NotInitializedError):
        agent.update()


def test_q_learning() -> None:
    test_map = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, None, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
    agent = QLearningAgent(seed=0, env=env)
    for _ in range(2):
        run_td_episode(env=env, agent=agent)
    expected = pytest.approx(0.1085729799230003)
    assert agent.average_loss == expected
    assert agent.action_value == {
        ((0, 0), Action.UP): pytest.approx(0.012391064316034317, abs=1e-5),
        ((0, 0), Action.DOWN): pytest.approx(0.03795982897281647, abs=1e-5),
        ((0, 0), Action.LEFT): pytest.approx(-0.024705907329916954, abs=1e-5),
        ((0, 0), Action.RIGHT): pytest.approx(0.122371606528759, abs=1e-5),
        ((0, 1), Action.UP): pytest.approx(0.02835841476917267, abs=1e-5),
        ((0, 1), Action.DOWN): pytest.approx(0.03410963714122772, abs=1e-5),
        ((0, 1), Action.LEFT): pytest.approx(-0.06401852518320084, abs=1e-5),
        ((0, 1), Action.RIGHT): pytest.approx(0.08941204845905304, abs=1e-5),
        ((0, 2), Action.UP): pytest.approx(0.13887614011764526, abs=1e-5),
        ((0, 2), Action.DOWN): pytest.approx(-0.12020474672317505, abs=1e-5),
        ((0, 2), Action.LEFT): pytest.approx(-0.09385768324136734, abs=1e-5),
        ((0, 2), Action.RIGHT): pytest.approx(0.14876116812229156, abs=1e-5),
        ((0, 3), Action.UP): pytest.approx(0.08330991864204407, abs=1e-5),
        ((0, 3), Action.DOWN): pytest.approx(-0.05502215027809143, abs=1e-5),
        ((0, 3), Action.LEFT): pytest.approx(-0.024705795571208, abs=1e-5),
        ((0, 3), Action.RIGHT): pytest.approx(0.07155784964561462, abs=1e-5),
        ((1, 0), Action.UP): pytest.approx(0.08537131547927856, abs=1e-5),
        ((1, 0), Action.DOWN): pytest.approx(0.030280638486146927, abs=1e-5),
        ((1, 0), Action.LEFT): pytest.approx(-0.09365365654230118, abs=1e-5),
        ((1, 0), Action.RIGHT): pytest.approx(0.055440232157707214, abs=1e-5),
        ((1, 1), Action.UP): pytest.approx(0.12489913403987885, abs=1e-5),
        ((1, 1), Action.DOWN): pytest.approx(-0.06309452652931213, abs=1e-5),
        ((1, 1), Action.LEFT): pytest.approx(-0.06900961697101593, abs=1e-5),
        ((1, 1), Action.RIGHT): pytest.approx(0.17078934609889984, abs=1e-5),
        ((1, 2), Action.UP): pytest.approx(0.03831055760383606, abs=1e-5),
        ((1, 2), Action.DOWN): pytest.approx(-0.01612631231546402, abs=1e-5),
        ((1, 2), Action.LEFT): pytest.approx(-0.06555116921663284, abs=1e-5),
        ((1, 2), Action.RIGHT): pytest.approx(0.028176531195640564, abs=1e-5),
        ((1, 3), Action.UP): pytest.approx(0.10358510911464691, abs=1e-5),
        ((1, 3), Action.DOWN): pytest.approx(-0.08015541732311249, abs=1e-5),
        ((1, 3), Action.LEFT): pytest.approx(-0.005136379972100258, abs=1e-5),
        ((1, 3), Action.RIGHT): pytest.approx(0.16100265085697174, abs=1e-5),
        ((2, 0), Action.UP): pytest.approx(0.08069709688425064, abs=1e-5),
        ((2, 0), Action.DOWN): pytest.approx(0.005144223570823669, abs=1e-5),
        ((2, 0), Action.LEFT): pytest.approx(-0.12853993475437164, abs=1e-5),
        ((2, 0), Action.RIGHT): pytest.approx(0.038303472101688385, abs=1e-5),
        ((2, 1), Action.UP): pytest.approx(0.16602256894111633, abs=1e-5),
        ((2, 1), Action.DOWN): pytest.approx(-0.016644027084112167, abs=1e-5),
        ((2, 1), Action.LEFT): pytest.approx(-0.058554477989673615, abs=1e-5),
        ((2, 1), Action.RIGHT): pytest.approx(0.15737693011760712, abs=1e-5),
        ((2, 2), Action.UP): pytest.approx(0.06638707220554352, abs=1e-5),
        ((2, 2), Action.DOWN): pytest.approx(0.05980466678738594, abs=1e-5),
        ((2, 2), Action.LEFT): pytest.approx(0.05435100942850113, abs=1e-5),
        ((2, 2), Action.RIGHT): pytest.approx(0.1403900384902954, abs=1e-5),
        ((2, 3), Action.UP): pytest.approx(0.048222266137599945, abs=1e-5),
        ((2, 3), Action.DOWN): pytest.approx(-0.024945009499788284, abs=1e-5),
        ((2, 3), Action.LEFT): pytest.approx(-0.166785329580307, abs=1e-5),
        ((2, 3), Action.RIGHT): pytest.approx(0.13613100349903107, abs=1e-5),
    }
