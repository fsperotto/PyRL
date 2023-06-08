from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
from gymnasium.spaces import Space

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Button, Switch, Static
from textual.containers import Horizontal, ScrollableContainer, Middle, Center
from textual.screen import Screen

from pyrl import Agent

class QLearning(Agent):
    """The QLearning class"""

    def __init__(self, observation_space: Space, action_space: Space, initial_observation=None, discount=0.9, learning_rate=0.1, should_explore: Callable=None, initial_Q: np.ndarray=None, initial_Q_value: float=None):

        self.initial_Q = None
        self.initial_Q_value = None

        super(QLearning, self).__init__(observation_space, action_space, initial_observation)

        self.current_state = initial_observation
        self.current_reward = None
        self.last_action = None
        self.last_state = None
        self.time = 0
        self.discount = discount
        self.learning_rate = learning_rate
        self.should_explore = should_explore if should_explore is not None else self.builtin_should_explore
        self.saved_should_explore = self.should_explore

        if initial_Q is not None:
            self.Q_check(initial_Q)
            self.initial_Q = initial_Q
        elif initial_Q_value is not None:
            self.initial_Q_value = initial_Q_value

        self.reset(initial_observation, reset_knowledge=True)

    def act(self) -> list:
        if self.current_state is None:
            raise ValueError("current_state property should be initilized. Maybe you forgot to call the reset method ?")

        should_explore = self.should_explore(self)

        if should_explore:
            a = np.random.randint(0, self.action_space.n)
        else:
            a = self.Q[self.current_state, :].argmax()

        self.last_action = a

        return a, should_explore

    def reset(self, state: int, reset_knowledge=True) -> int:
        super(QLearning, self).reset(state, reset_knowledge)
        self.current_state = state
        self.time = 0

        if reset_knowledge:
            if self.initial_Q is not None:
                self.Q = self.initial_Q
            elif self.initial_Q_value is not None:
                self.Q = np.full((self.observation_space.n, self.action_space.n), self.initial_Q_value, dtype=float)
            else:
                self.Q = np.full((self.observation_space.n, self.action_space.n), 0, dtype=float)

    def observe(self, state: int, reward: float) -> None:
        if self.current_state is None:
            raise ValueError("current_state property should be initilized. Maybe you forgot to call the reset method ?")

        """Memorize the observed state and received reward."""
        self.last_state = self.current_state
        self.current_state = state
        self.current_reward = reward
        self.time = self.time + 1

    def learn(self) -> None:
        self.Q[self.last_state, self.last_action] = (1 - self.learning_rate) * self.Q[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.Q[self.current_state, :].max())

        return self.Q[self.last_state, self.last_action]

    def builtin_should_explore(self, agent: Agent) -> bool:
        pn = np.random.random()
        return pn < (1 - (1 / math.log(self.time + 2)))

    def Q_check(self, Q: np.ndarray) -> None:
        if Q.shape is not (self.observation_space.n, self.action_space.n):
            raise ValueError(f"Q should have shape ({self.observation_space.n}, {self.action_space.n}). Got {Q.shape}.")

    def debugger_compose(self):
        return ScrollableContainer(
            Button("Modify agent", variant="primary", id="modify"),
            DataTable(id="agent_q_table"),
            DataTable(id="agent_exploration_table"),
        )

    def debugger_on_mount(self, app: App):
        self.debugger = app
        self.Q_table = self.debugger.query_one("#agent_q_table")
        self.Q_table.add_columns("action 0", "action 1", "action 2", "action 3")

        for i in range(self.Q.shape[0]):
            row = []
            for j in range(self.Q.shape[1]):
                row.append(self.Q[i, j])

            self.Q_table.add_row(*row, label=str(i))


        self.exploration_table = self.debugger.query_one("#agent_exploration_table")
        self.exploration_table.add_columns("time", "exploration action")

    def debugger_update(self, data):
        self.Q_table.update_cell_at((data["learning"]["x"], data["learning"]["y"]), data["learning"]["value"])

    def debugger_on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "modify":
            self.debugger.action_pause()
            self.debugger.push_screen(ModifyScreen())

class ModifyScreen(Screen):
    """docstring for ModifyScreen."""

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(
            Static("only exploitation:      ", classes="label"),
            Switch(value=(False if self.app.agent.should_explore == self.app.agent.saved_should_explore else True)),
            Button("Close", variant="error", id="close"),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.app.pop_screen()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.value = event.value

        if self.value:
            self.app.agent.should_explore = lambda agent: False
        else:
            self.app.agent.should_explore = self.app.agent.saved_should_explore
