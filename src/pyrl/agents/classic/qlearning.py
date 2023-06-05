from typing import Iterable, Callable

import numpy as np
import math

from pyrl import Agent

class QLearning(Agent):
    """The QLearning class"""

    def __init__(self, observation_space, action_space, initial_observation, discount=0.9, learning_rate=0.1, epsilon: Callable = None):
        super(QLearning, self).__init__(observation_space, action_space, initial_observation)

        self.current_state = initial_observation
        self.current_reward = None
        self.last_action = None
        self.last_state = None
        self.time = 0
        self.discount = discount
        self.learning_rate = learning_rate

    def act(self) -> list:
        pn = np.random.random()
        #if pn < (1 - (1 / math.log(self.time + 2))):
            # optimal_action = self.Q[s, :].max()
        a = self.Q[self.current_state, :].argmax()
        #else:
            #a = np.random.randint(0, self.action_space.n)

        self.last_action = a

        return a

    def reset(self, state: int, reset_knowledge=True) -> int:
        super(QLearning, self).reset(state, reset_knowledge)
        self.current_state = state
        self.time = 0

        if reset_knowledge:
            self.Q = np.full((self.observation_space.n, self.action_space.n), 100)

    def observe(self, state: int, reward: float, learn=True) -> None:
        """Memorize the observed state and received reward."""
        self.last_state = self.current_state
        self.current_state = state
        self.current_reward = reward
        self.time = self.time + 1

        print(f"Reward at step {self.time} is {self.current_reward}")

        if learn:
            self.learn()

    def learn(self) -> None:
        self.Q[self.last_state, self.last_action] = (1 - self.learning_rate) * self.Q[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.Q[self.current_state, :].max())
