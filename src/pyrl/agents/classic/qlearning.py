from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
from gymnasium.spaces import Space

from pyrl import Agent

class QLearning(Agent):
    """The QLearning class"""

    def __init__(self, observation_space: Space, action_space: Space,
                 default_action=None, initial_budget:float=None,
                 discount=0.9, learning_rate=0.1, exploration_rate=None, should_explore=None, 
                 initial_Q: np.ndarray=None, initial_Q_value: float=None):
        
        super().__init__(observation_space=observation_space, 
                         action_space=action_space, 
                         initial_budget=initial_budget,
                         default_action=default_action)
        
        #memory
        self.last_s = None
        
        self.initial_Q = initial_Q
        self.initial_Q_value = initial_Q_value

        if initial_Q is not None:
            self.Q_check(initial_Q)
            self.initial_Q = initial_Q
        elif initial_Q_value is not None:
            self.initial_Q_value = initial_Q_value
        

        self.discount = discount
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        if should_explore is not None:
            self.should_explore = should_explore  
        elif exploration_rate is not None:
            self.should_explore = self.builtin_epsilon_should_explore
        else:
            self.should_explore = self.builtin_log_decreasing_should_explore

        self.Q = None
        self.N = None
        
        #self.reset(initial_observation, reset_knowledge=True)



    def reset(self, initial_observation, reset_knowledge=True, reset_budget=True, learning=True):
    
        super().reset(initial_observation=initial_observation, 
                      reset_knowledge=reset_knowledge,
                      reset_budget=reset_budget,
                      learning=learning)
        
        if reset_knowledge:
            if self.initial_Q is not None:
                self.Q = self.initial_Q
            elif self.initial_Q_value is not None:
                self.Q = np.full(self.observation_shape + self.action_shape, self.initial_Q_value, dtype=float)
            else:
                self.Q = np.random.sample(self.observation_shape + self.action_shape)
                
            self.N = np.zeros(self.observation_shape + self.action_shape, dtype=int)



    def choose_action(self):
        #if the agent was not reseted after initialization, then reset
        if self.should_reset:
            raise ValueError("ERROR: Agent properties should be initilized. Maybe you forgot to call the reset method ?")

        if self.should_explore(self):
            a = self.action_space.sample()
        else:
            #Q_s = np.take(self.Q, self.get_state())
            Q_s = self.Q[self.get_state()]
            maxq = Q_s.max()
            a = np.random.choice(np.flatnonzero(Q_s == maxq))

        self.a = a

        return a



    def observe(self, state, reward: float, terminated: bool, truncated: bool) -> None:
        #if the agent was not reseted after initialization, then reset
        if self.should_reset:
            raise ValueError("ERROR: Agent properties should be initilized. Maybe you forgot to call the reset method ?")
        
        super().observe(state, reward, terminated=terminated, truncated=truncated)

        self.last_s = self.s


    def learn(self) -> None:
        #self.Q[self.last_state, self.last_action] = (1 - self.learning_rate) * self.Q[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.Q[self.current_state, :].max())
        index_sa = self.get_state(self.last_s) + self.get_action()
        self.N[index_sa] += 1
        index_s = self.get_state()
        Q_sa = self.Q[index_sa]
        Q_s = self.Q[index_s]
        new_q = (1 - self.learning_rate) * Q_sa + self.learning_rate * (self.r + self.discount * Q_s.max())
        self.Q[index_sa] = new_q
        return new_q

    def builtin_log_decreasing_should_explore(self, agent: Agent) -> bool:
        return np.random.random() < (1 - (1 / math.log(self.time + 2)))
    
    def builtin_epsilon_should_explore(self, agent: Agent) -> bool:
        return np.random.rand() < self.exploration_rate
    

    def Q_check(self, Q: np.ndarray) -> None:
        if Q.shape is not (self.observation_space.n, self.action_space.n):
            raise ValueError(f"Q should have shape ({self.observation_space.n}, {self.action_space.n}). Got {Q.shape}.")
