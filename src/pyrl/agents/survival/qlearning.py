from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
from gymnasium.spaces import Space

from pyrl import Agent
from pyrl.agents.classic import QLearning as ClassicQLearning


class QLearning(ClassicQLearning):
    """The Survival QLearning class"""

    def __init__(self, observation_space, action_space, 
                 default_action=None, initial_budget=None,
                 discount=0.9, learning_rate=0.1, exploration_rate=None, should_explore=None, 
                 initial_Q=None, initial_Q_value=None,
                 store_N = True, store_V = False, store_policy = False,
                 survival_threshold=None,
                 name="ST-Q-Learning"):
        
        super().__init__(observation_space=observation_space, 
                         action_space=action_space, 
                         initial_budget=initial_budget,
                         default_action=default_action,
                         discount=discount, 
                         learning_rate=learning_rate, 
                         exploration_rate=exploration_rate, 
                         should_explore=should_explore,
                         initial_Q=initial_Q,
                         initial_Q_value=initial_Q_value,
                         store_N=store_N, store_V=store_V, store_policy=store_policy,
                         name=name)

        self.survival_threshold = survival_threshold
        

    def choose_action(self):
        if self.s is None:
            raise ValueError("current_state property should be initilized. Maybe you forgot to call the reset method ?")

        maxq = self.Q[self.get_state()].max()
        
        if self.b < self.survival_threshold and maxq > 0:
            a = np.random.choice(np.flatnonzero(self.Q[self.get_state()] == maxq))
        else:
            a = super().choose_action()

        self.a = a

        return a


    def reset(self, initial_observation, reset_knowledge=True, reset_budget=True, learning=True):

        super().reset(initial_observation=initial_observation, 
                      reset_knowledge=reset_knowledge,
                      reset_budget=reset_budget,
                      learning=learning)


