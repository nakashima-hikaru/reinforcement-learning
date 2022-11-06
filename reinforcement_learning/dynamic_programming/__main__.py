from collections import defaultdict

from reinforcement_learning.dynamic_programming.grid_world import GridWorld, Action
from reinforcement_learning.dynamic_programming.policy_eval import policy_eval, Policy, StateValue

import numpy as np

test_map = np.array([[0.0, 0.0, 0.0, 1.0],
                     [0.0, None, 0.0, -1.0],
                     [0.0, 0.0, 0.0, 0.0]], dtype=float)
env = GridWorld(reward_map=test_map, goal_state=(0, 3), start_state=(2, 0))
pi: Policy = defaultdict(lambda: {Action.UP: 0.25, Action.DOWN: 0.25, Action.LEFT: 0.25, Action.RIGHT: 0.25})
v: StateValue = defaultdict(lambda: 0)
v = policy_eval(pi=pi, v=v, env=env, gamma=0.9, threshold=1e-3)
print(v)
