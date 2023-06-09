from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
from gymnasium.spaces import Space

from pyrl import Agent

class QLearning(Agent):
    """The QLearning class"""

    def __init__(self, observation_space: Space, action_space: Space, initial_observation=None, 
                 discount=0.9, learning_rate=0.1, exploration_rate=None, should_explore=None, 
                 initial_Q: np.ndarray=None, initial_Q_value: float=None, budget: int=None):
        
        self.initial_Q = initial_Q
        self.initial_Q_value = initial_Q_value

        super(QLearning, self).__init__(observation_space, action_space, initial_observation, budget)

        self.current_state = initial_observation
        self.current_reward = None
        self.last_action = None
        self.last_state = None
        self.time = 0
        self.discount = discount
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        if should_explore is not None:
            self.should_explore = should_explore  
        elif exploration_rate is not None:
            self.should_explore = self.builtin_epsilon_should_explore
        else:
            self.should_explore = self.builtin_log_decreasing_should_explore
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

        if self.should_explore(self):
            a = np.random.randint(0, self.action_space.n)
        else:
            maxq = self.Q[self.current_state, :].max()
            a = np.random.choice(np.flatnonzero(self.Q[self.current_state, :] == maxq))

        self.last_action = a

        return a

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

    def observe(self, state: int, reward: float, terminated: bool, truncated: bool) -> None:
        if self.current_state is None:
            raise ValueError("current_state property should be initilized. Maybe you forgot to call the reset method ?")
        
        super().observe(state, reward, terminated=terminated, truncated=truncated)

        """Memorize the observed state and received reward."""
        self.last_state = self.current_state
        self.current_state = state
        self.current_reward = reward
        self.time = self.time + 1

    def learn(self) -> None:
        self.Q[self.last_state, self.last_action] = (1 - self.learning_rate) * self.Q[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.Q[self.current_state, :].max())

        return self.Q[self.last_state, self.last_action]

    def builtin_log_decreasing_should_explore(self, agent: Agent) -> bool:
        return np.random.random() < (1 - (1 / math.log(self.time + 2)))
    
    def builtin_epsilon_should_explore(self, agent: Agent) -> bool:
        return np.random.rand() < self.exploration_rate
    

    def Q_check(self, Q: np.ndarray) -> None:
        if Q.shape is not (self.observation_space.n, self.action_space.n):
            raise ValueError(f"Q should have shape ({self.observation_space.n}, {self.action_space.n}). Got {Q.shape}.")
