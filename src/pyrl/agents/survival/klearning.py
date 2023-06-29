from typing import Callable
import numpy as np
from gymnasium.spaces import Space
import numpy as np

from pyrl.agents.survival import QLearning

class KLearning(QLearning):
    def __init__(self, observation_space: Space, action_space: Space, initial_observation=None, discount=0.9, learning_rate=0.1, should_explore: Callable = None, initial_Q: np.ndarray = None, initial_Q_value: float = None, budget=100, survival_threshold=10, exploration_threshold=100, initial_K_value: float = 0):
        self.initial_K_value = initial_K_value
        self.exploration_threshold = exploration_threshold
        self.K = None
        super().__init__(observation_space, action_space, initial_observation, discount, learning_rate, should_explore, initial_Q, initial_Q_value, budget, survival_threshold)

    def act(self) -> list:
        if self.recharge_mode:
            maxq = self.Q[self.current_state, :].max()
            a = np.random.choice(np.flatnonzero(self.Q[self.current_state, :] == maxq))
        else:
            maxk = self.K[self.current_state, :].max()
            a = np.random.choice(np.flatnonzero(self.K[self.current_state, :] == maxk))
        
        self.last_action = a

        return self.last_action
    
    def reset(self, state: int, reset_knowledge=True) -> int:
        self.K = np.full((self.observation_space.n, self.action_space.n), self.initial_K_value)
        self.recharge_mode = False
        return super().reset(state, reset_knowledge)
    
    def observe(self, state: int, reward: float, terminated: bool = False, truncated: bool = False) -> None:
        super().observe(state, reward, terminated, truncated)

        if not self.recharge_mode and self.budget < self.survival_threshold:
            self.recharge_mode = True
        
        if self.recharge_mode and self.budget > self.exploration_threshold:
            self.recharge_mode = False

    def learn(self) -> None:
        super().learn()
        self.K[self.last_state, self.last_action] = (1 - self.learning_rate) * self.K[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.K[self.current_state, :].max())