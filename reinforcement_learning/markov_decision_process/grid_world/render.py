"""A Renderer class that is used to visually represent the state of an environment modeled as a grid world.

It relies on the Matplotlib library for visualization of the state values and Q-values of the grid world environment.
"""
from typing import TYPE_CHECKING, Self, TypeAlias, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Polygon, Rectangle

from reinforcement_learning.errors import NotInitializedError
from reinforcement_learning.markov_decision_process.grid_world.environment import (
    Action,
    ActionValue,
    Map,
    Policy,
    State,
    StateValue,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

RET: TypeAlias = dict[Action, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]


class Renderer:
    """Class to render the grid world environment.

    This class provides methods to render the state values and Q-values of the grid world environment.
    It uses the Matplotlib library for visualization.

    Attributes:    -----------
    reward_map (Map): The reward map representing the grid world environment.
    goal_state (State): The goal state in the grid world.
    wall_states (set[State]): The set of wall states in the grid world.
    ys (int): The number of rows in the reward map.
    xs (int): The number of columns in the reward map.
    ax (mpl.axes.Axes): The matplotlib axes for rendering.
    fig (mpl.figure.Figure): The matplotlib figure for rendering.
    first_flg (bool): A flag indicating if this is the first render call.
    """

    def __init__(self: Self, *, reward_map: Map, goal_state: State, wall_states: frozenset[State]) -> None:
        """Initialize the Renderer class.

        Args:
        ----
            reward_map (Map): The reward map representing the grid world environment.
            goal_state (State): The goal state in the grid world.
            wall_states (set[State]): The set of wall states in the grid world.

        Attributes:
        ----------
            reward_map (Map): The reward map representing the grid world environment.
            goal_state (State): The goal state in the grid world.
            wall_states (frozenset[State]): The set of wall states in the grid world.
            ys (int): The number of rows in the reward map.
            xs (int): The number of columns in the reward map.
            ax (mpl.axes.Axes): The matplotlib axes for rendering.
            fig (mpl.figure.Figure): The matplotlib figure for rendering.
            first_flg (bool): A flag indicating if this
        """
        self.reward_map: Map = reward_map
        self.goal_state: State = goal_state
        self.wall_states: frozenset[State] = wall_states
        self.ys: int = len(self.reward_map)
        self.xs: int = len(self.reward_map[0])

        self.ax: Axes | None = None
        self.fig: Figure | None = None
        self.first_flg: bool = True

    def set_figure(self: Self, *, figsize: tuple[float, float] | None = None) -> None:
        """Set the figure size for rendering.

        Args:
        ----
            figsize: A tuple of two floats representing the width and height of the figure.
                     If None, default figure size will be used.
        """
        fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111)
        ax = self.ax
        ax.clear()
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.set_xticks(range(self.xs))
        ax.set_yticks(range(self.ys))
        ax.set_xlim(0, self.xs)
        ax.set_ylim(0, self.ys)
        ax.grid(visible=True)

    def render_v(
            self: Self, *, v: StateValue | None = None, policy: Policy | None = None, print_value: bool = True
    ) -> None:
        """Render the state values of the grid world environment.

        Args:
        ----
            v: StateValue or None, the state value function to be rendered
            policy: Policy or None, the policy to be rendered
            print_value: bool, whether to print the state values on the grid

        """
        self.set_figure()
        if self.ax is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="ax")

        v_array = self.get_value_array(v=v)

        if v is not None:
            self.render_values(v_array)

        self.render_annotations(v_array=v_array, policy=policy, print_value=print_value)

        plt.show()

    def get_value_array(self: Self, *, v: StateValue | None) -> npt.NDArray[np.float64]:
        """Retrieve the value array from the given StateValue object.

        Args:
        ----
            v: The StateValue object from which to extract the value array.

        Returns:
        -------
            The value array as a numpy NDArray of type np.float64.
        """
        v_array = np.zeros(shape=self.reward_map.shape)

        if v is not None:
            for state, value in v.items():
                v_array[state] = value

        return v_array

    def render_values(self: Self, v_array: npt.NDArray[np.float64]) -> None:
        """Render a 2D array of values using a color map.

        Args:
        ----
        v_array: A 2D numpy array of float64 values representing the values to be rendered.

        Returns:
        -------
        None
        """
        color_list = ["red", "white", "green"]
        cmap = mpl.colors.LinearSegmentedColormap.from_list("colormap_name", color_list)

        vmax, vmin = v_array.max(), v_array.min()
        vmax = max(vmax, abs(vmin))
        vmin = -1 * vmax
        vmax = 1 if vmax < 1 else vmax
        vmin = -1 if vmin > -1 else vmin
        if self.ax is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="ax")

        self.ax.pcolormesh(np.flipud(v_array), cmap=cmap, vmin=vmin, vmax=vmax)

    def render_annotations(
            self: Self, *, v_array: npt.NDArray[np.float64], policy: Policy | None, print_value: bool
    ) -> None:
        """Render the annotations for the grid world environment.

        Args:
        ----
            v_array: A 2D numpy array containing the state values for each state in the grid world.
            policy: The policy object containing the action probabilities for each state.
            print_value: A boolean indicating whether to print the state values on the grid (default=False).
        """
        if self.ax is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="ax")
        for y in range(self.ys):
            for x in range(self.xs):
                state = (y, x)
                self.render_reward(state=state)
                if v_array is not None:
                    self.render_value(v_array=v_array, state=state, print_value=print_value)
                if policy is not None:
                    self.render_policy(policy, state)
                if state in self.wall_states:
                    self.ax.add_patch(Rectangle((x, self.ys - y - 1), 1, 1, fc=(0.4, 0.4, 0.4, 1.0)))

    def render_reward(self: Self, state: State) -> None:
        """Render the reward value for a given state on the grid.

        Args:
        ----
            state (State): The state for which to render the reward value.

        Returns:
        -------
            None
        """
        if self.ax is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="ax")
        r = self.reward_map[state]
        if r != 0 and r is not None:
            txt = "R " + str(r)
            if state == self.goal_state:
                txt = txt + " (GOAL)"

            self.ax.text(state[1] + 0.1, self.ys - state[0] - 0.9, txt)

    def render_value(self: Self, *, v_array: npt.NDArray[np.float64], state: State, print_value: bool) -> None:
        """Render the value of a given state on the matplotlib figure.

        Args:
        ----
            v_array (npt.NDArray[np.float64]): Array containing the state values.
            state (State): The state for which the value is to be rendered.
            print_value (bool): Whether to print the value on the figure.

        Returns:
        -------
            None
        """
        if self.ax is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="ax")
        if state in self.wall_states or not print_value:
            return
        offsets = [(0.4, -0.15), (-0.15, -0.3)]
        key = 0
        max_row = 7
        if v_array.shape[0] > max_row:
            key = 1
        offset = offsets[key]
        self.ax.text(state[1] + offset[0], self.ys - state[0] + offset[1], f"{v_array[state]:12.2f}")

    def render_policy(self: Self, policy: Policy, state: State) -> None:
        """Render the policy on a grid world.

        Args:
        ----
            policy (Policy): The policy to render.
            state (State): The current state.

        Returns:
        -------
            None
        """
        if self.ax is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="ax")
        if state in self.wall_states:
            return
        actions = policy[state]
        max_actions = [kv[0] for kv in actions.items() if kv[1] == max(actions.values())]
        arrows = ["↑", "↓", "←", "→"]
        offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
        for action in max_actions:
            if state == self.goal_state:
                continue
            arrow = arrows[action]
            offset = offsets[action]
            self.ax.text(state[1] + 0.45 + offset[0], self.ys - state[0] - 0.5 + offset[1], arrow)

    def generate_policy(self: Self, *, q: ActionValue) -> Policy:
        """Generate a policy based on the state-action values.

        Returns
        -------
            Policy: A dictionary representing the policy, where each state maps to a dictionary of action probabilities.
        """
        policy: Policy = Policy()
        for y in range(self.ys):
            for x in range(self.xs):
                state: State = (y, x)
                qs = [q[state, action] for action in Action]  # action_size
                max_action = Action(int(np.argmax(qs)))
                probs: dict[Action, float] = {Action.UP: 0.0, Action.DOWN: 0.0, Action.LEFT: 0.0, Action.RIGHT: 0.0}
                probs[max_action] = 1.0
                policy[state] = probs

        return policy

    def render_q(self: Self, *, q: ActionValue, show_greedy_policy: bool = True) -> None:
        """Render the Q-values of the grid world environment.

        Args:
        ----
            q: A dictionary containing the Q-values for each state-action pair.
            show_greedy_policy: A boolean indicating whether to show the greedy policy.

        Returns:
        -------
            None


        ------      Raises:
            None

        """
        self.set_figure()

        if self.ax is None:
            raise NotInitializedError(instance_name=str(self), attribute_name="ax")

        qmax, qmin = max(q.values()), min(q.values())
        qmax = max(qmax, abs(qmin))
        qmax = 1 if qmax < 1 else qmax

        color_list = ["red", "white", "green"]
        self.cmap = mpl.colors.LinearSegmentedColormap.from_list("colormap_name", color_list)

        for y in range(self.ys):
            for x in range(self.xs):
                for action in Action:
                    state = (y, x)
                    r = cast(float, self.reward_map[y, x])

                    self.render_reward_q(state=state, reward=r)

                    if state == self.goal_state:
                        continue

                    tq = q[(state, action)]
                    color_scale = 0.5 + (tq / qmax) / 2  # normalize: 0.0-1.0
                    self.render_polygon_q(state, action, color_scale, tq)

        plt.show()

        if show_greedy_policy:
            policy = self.generate_policy(q=q)
            self.render_v(v=None, policy=policy)

    def render_reward_q(self: Self, *, state: State, reward: float) -> None:
        """Render the reward value of a given state.

        Args:
        ----
            state (State): The state for which the reward is being rendered.
            reward (float): The reward value for the state.
        """
        y, x = state
        zero = 0.0
        if reward != zero and reward is not None:
            txt = "R " + str(reward)
            if state == self.goal_state:
                txt = txt + " (GOAL)"
            if self.ax is None:
                raise NotInitializedError(instance_name=str(self), attribute_name="ax")
            self.ax.text(x + 0.05, self.ys - y - 0.95, txt)

    def render_polygon_q(self: Self, state: State, action: Action, color_scale: float, tq: float) -> None:
        """Render a polygon representing a state-action pair in the grid world.

        Args:
        ----
            state (State): The state of the grid world.
            action (Action): The action taken in the state.
            color_scale: The color scale for the polygon.
            tq: The value associated with the state-action pair.
        """
        if state not in self.wall_states and state != self.goal_state:
            action_map = self.generate_action_map(state)
            poly = Polygon(action_map[action], fc=self.cmap(color_scale))
            if self.ax is None:
                raise NotInitializedError(instance_name=str(self), attribute_name="ax")
            self.ax.add_patch(poly)
            offset = self.generate_offsets_q()[action]
            x, y = state
            tx, ty = x, self.ys - y - 1
            self.ax.text(tx + offset[0], ty + offset[1], f"{tq:12.2f}")

    @classmethod
    def generate_offsets_q(cls: type[Self]) -> dict[Action, tuple[float, float]]:
        """Generate Offsets Q.

        Returns
        -------
        - A dictionary that maps each action to a tuple of offset values.
          The offset values represent the coordinates where the action label should be placed in the grid.
        """
        return {
            Action.UP: (0.1, 0.8),
            Action.DOWN: (0.1, 0.1),
            Action.LEFT: (-0.2, 0.4),
            Action.RIGHT: (0.4, 0.4),
        }

    def generate_action_map(self: Self, state: State) -> RET:
        """Generate the action map for a given state.

        Args:
        ----
            state (State): The current state for which the action map needs to be generated.

        Returns:
        -------
            Dict[Action, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
            The action map consisting of the coordinates of the four vertices of each action.
        """
        y, x = state
        tx, ty = float(x), float(self.ys) - float(y) - 1.0
        return {
            Action.UP: ((0.5 + tx, 0.5 + ty), (tx + 1.0, ty + 1.0), (tx, ty + 1.0)),
            Action.DOWN: ((tx, ty), (tx + 1.0, ty), (tx + 0.5, ty + 0.5)),
            Action.LEFT: ((tx, ty), (tx + 0.5, ty + 0.5), (tx, ty + 1.0)),
            Action.RIGHT: ((0.5 + tx, 0.5 + ty), (tx + 1.0, ty), (tx + 1.0, ty + 1.0)),
        }
