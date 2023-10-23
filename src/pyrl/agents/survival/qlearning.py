from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
from gymnasium.spaces import Space

from pyrl import Agent
from pyrl.agents import QLearning as ClassicQLearning


class QLearning(ClassicQLearning):
    """The Survival QLearning class"""

    def __init__(self, env, *,
                 default_action=None,
                 discount=0.9, learning_rate=0.1, exploration_rate=None, should_explore=None, 
                 initial_Q=None, initial_Q_value=None,
                 store_N=True, store_V=False, store_policy=True,
                 survival_threshold=None,
                 name="ST-Q-Learning",
                 remember_prev_a=False,
                 store_N_sa=False, store_N_saz=False, store_N_z=False, store_N_a=False
                ):
        
        super().__init__(env=env, 
                         default_action=default_action,
                         discount=discount, 
                         learning_rate=learning_rate, 
                         exploration_rate=exploration_rate, 
                         should_explore=should_explore,
                         initial_Q=initial_Q,
                         initial_Q_value=initial_Q_value,
                         remember_prev_a=remember_prev_a,
                         store_N_sa=store_N_sa, store_N_saz=store_N_saz, store_N_z=store_N_z, store_N_a=store_N_a, 
                         store_V=store_V, store_policy=store_policy,
                         name=name)

        self.survival_threshold = survival_threshold
        self.recharge_mode = False
        

    #--------------------------------------------------------------    
    def _choose(self):
        
        if self.s is None:
            raise ValueError("current_state property should be initilized. Maybe you forgot to call the reset method ?")

        maxq = self.Q[self.get_state_tpl()].max()
        
        if self.b < self.survival_threshold and maxq > 0:
            self.recharge_mode = True
            a = np.random.choice(np.flatnonzero(self.Q[self.get_state_tpl()] == maxq))
        else:
            self.recharge_mode = False
            a = super()._choose()

        return a


