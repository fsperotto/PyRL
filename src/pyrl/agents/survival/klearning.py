from typing import Callable
import numpy as np
from gymnasium.spaces import Space

from pyrl.agents.survival import QLearning

class KLearning(QLearning):

    def __init__(self, observation_space: Space, action_space: Space, 
                 default_action=None, initial_budget:float=None,
                 discount=0.9, learning_rate=0.1, 
                 exploration_rate=None, should_explore: Callable = None, 
                 initial_Q: np.ndarray = None, initial_Q_value: float = None, 
                 store_N = True, store_V = False, store_policy = True,
                 survival_threshold=10, exploration_threshold=None, 
                 initial_K_value: float = 0,
                 name:str="K-Learning"):
        
        super().__init__(observation_space=observation_space, 
                         action_space=action_space, 
                         initial_budget=initial_budget,
                         default_action=default_action,
                         discount=discount, learning_rate=learning_rate,
                         exploration_rate=exploration_rate,
                         should_explore=should_explore,
                         survival_threshold=survival_threshold,
                         initial_Q=initial_Q, initial_Q_value=initial_Q_value,
                         store_N=store_N, store_V=store_V, store_policy=store_policy,
                         name=name)
       
        self.initial_K_value = initial_K_value

        if exploration_threshold is None:
           self.exploration_threshold = survival_threshold
        else:
           self.exploration_threshold = exploration_threshold
        self.K = None

    def choose_action(self):
        
        #if self.recharge_mode and maxq > 0:
        if self.recharge_mode :
        
            maxq = self.Q[self.get_state()].max()
            a = np.random.choice(np.flatnonzero(self.Q[self.get_state()] == maxq))
        
        else:
        
            if self.should_explore(self):

               a = self.action_space.sample()
               
            else:
           
               maxk = self.K[self.get_state()].max()
               a = np.random.choice(np.flatnonzero(self.K[self.get_state()] == maxk))
        
        self.a = a

        return self.a
    
    def reset(self, state: int, reset_knowledge=True):
        if reset_knowledge:
           self.K = np.full(self.observation_shape + self.action_shape, self.initial_K_value)
        self.recharge_mode = False
        return super().reset(state, reset_knowledge)
    
    def observe(self, state: int, reward: float, terminated: bool = False, truncated: bool = False) -> None:
        super().observe(state, reward, terminated, truncated)

        if not self.recharge_mode and self.b <= self.survival_threshold:
            self.recharge_mode = True
        
        if self.recharge_mode and self.b > self.exploration_threshold:
            self.recharge_mode = False

    def learn(self):
        super().learn()
        #self.K[self.get_state(self.last_s) + self.get_action()] = (1 - self.learning_rate) * self.K[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.K[self.current_state, :].max())
        index_sa = self.get_state(self.last_s) + self.get_action()
        index_s = self.get_state()
        K_sa = self.K[index_sa]
        K_s = self.K[index_s]
        new_k = (1 - self.learning_rate) * K_sa + self.learning_rate * (self.r + self.discount * K_s.max())
        self.K[index_sa] = new_k
        return new_k
        