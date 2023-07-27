from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
from gymnasium.spaces import Space

from pyrl import Agent

class QLearning(Agent):
    """The QLearning class"""

    #--------------------------------------------------------------    
    def __init__(self, observation_space, action_space,
                 default_action=None, initial_budget=None,
                 discount=0.9, learning_rate=0.1, exploration_rate=None, should_explore=None, 
                 initial_Q=None, initial_Q_value=None,
                 store_N = True, store_V = False, store_policy = True,
                 name="Q-Learning"):
        
        super().__init__(observation_space, action_space, 
                         initial_budget=initial_budget,
                         default_action=default_action,
                         name=name)
        
        #memory
        self.last_s = None
        
        self.initial_Q = initial_Q
        self.initial_Q_value = initial_Q_value

        if initial_Q is not None:
            self.Q_check(initial_Q)
            self.initial_Q = initial_Q
        elif initial_Q_value is not None:
            self.initial_Q_value = initial_Q_value
        
        #parameters
        self.discount = discount                   #gamma
        self.learning_rate = learning_rate         #alpha
        self.exploration_rate = exploration_rate   #epsilon
        
        #epsilon as a function
        if should_explore is not None:
            self.should_explore = should_explore  
        elif exploration_rate is not None:
            self.should_explore = self.builtin_epsilon_should_explore
        else:
            self.should_explore = self.builtin_log_decreasing_should_explore

        # Q(s, a) table
        self.Q = None
        
        self.N = None
        self.V = None
        self.policy = None
        
        self.store_N = store_N
        self.store_V = store_V
        self.store_policy = store_policy
        
        #self.reset(initial_observation, reset_knowledge=True)



    #--------------------------------------------------------------    
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
                
            if self.store_N:
               self.N = np.zeros(self.observation_shape + self.action_shape, dtype=int)

            if self.store_V:
               self._reset_V()
               #self.V = np.zeros(self.observation_shape)

            if self.store_policy:
               self._reset_policy()
               #self.policy = np.zeros(self.observation_shape + self.action_shape)

    #--------------------------------------------------------------    
    def _reset_V(self):
       self.V = np.zeros(self.observation_shape)
       for obs in self.observation_iterator:
           self.V[obs] = self.Q[obs].max()
       
    #--------------------------------------------------------------    
    def _update_V(self, s):
       self.V[s] = self.Q[s].max()

    #--------------------------------------------------------------    
    def _reset_policy(self):
       
       self.policy = np.ones(self.observation_shape + self.action_shape)
       
       for obs in self.observation_iterator:
           
           if self.store_V:
              v = self.V[obs]
           else:
              v = self.Q[obs].max()
              
           for a in self.action_iterator:
              if self.Q[obs + a] == v:
                 self.policy[obs + a] = 1
              else:
                 self.policy[obs + a] = 0

    #--------------------------------------------------------------    
    def _update_policy(self, s):

         if self.store_V:
            v = self.V[s]
         else:
            v = self.Q[s].max()

         #print([a for a in self.act_iterator])
         for a in self.action_iterator:
            if self.Q[s + a] == v:
               self.policy[s + a] = 1
            else:
               self.policy[s + a] = 0

            

    #--------------------------------------------------------------    
    def choose_action(self):
        #if the agent was not reseted after initialization, then reset
        if self.should_reset:
            raise ValueError("ERROR: Agent properties should be initilized. Maybe you forgot to call the reset method ?")

        if self.should_explore(self):

            a = self.action_space.sample()

        else:
        
            if self.store_policy:
               
               a = np.random.choice(np.flatnonzero(self.policy[self.get_state()]))
               
            else:
           
               #Q_s = np.take(self.Q, self.get_state())
               Q_s = self.Q[self.get_state()]
               #Q_s = self.Q[self.s]
               
               if self.store_V:
                  maxq = Q_s.max()
               else:
                  maxq = Q_s.max()
                  
               a = np.random.choice(np.flatnonzero(Q_s == maxq))

        self.a = a

        return a



    #--------------------------------------------------------------    
    def observe(self, state, reward: float, terminated: bool, truncated: bool) -> None:
        #if the agent was not reseted after initialization, then reset
        if self.should_reset:
            raise ValueError("ERROR: Agent properties should be initilized. Maybe you forgot to call the reset method ?")
        
        self.last_s = self.s

        super().observe(state, reward, terminated=terminated, truncated=truncated)



    #--------------------------------------------------------------    
    def learn(self) -> None:
        
        #self.Q[self.last_state, self.last_action] = (1 - self.learning_rate) * self.Q[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.Q[self.current_state, :].max())
        
        index_s = self.get_state(self.last_s)
        index_sa = self.get_state(self.last_s) + self.get_action()
        
        if self.store_N:
           self.N[index_sa] += 1
           
        index_next_s = self.get_state()
        #index_next_s = self.s
        
        Q_sa = self.Q[index_sa]
        
        if self.store_V:
           V_next_s = self.V[index_next_s]
        else:
           V_next_s = self.Q[index_next_s].max()
        
        new_q = (1 - self.learning_rate) * Q_sa + self.learning_rate * (self.r + self.discount * V_next_s)
        
        self.Q[index_sa] = new_q
        
        if self.store_V:
           self._update_V(index_s)

        if self.store_policy:
           self._update_policy(index_s)
        
        return new_q

    #--------------------------------------------------------------    
    def builtin_log_decreasing_should_explore(self, agent: Agent) -> bool:
        return np.random.random() < (1 - (1 / math.log(self.time + 2)))
    
    def builtin_epsilon_should_explore(self, agent: Agent) -> bool:
        return np.random.rand() < self.exploration_rate
    

    #--------------------------------------------------------------    
    def Q_check(self, Q: np.ndarray) -> None:
        if Q.shape is not (self.observation_space.n, self.action_space.n):
            raise ValueError(f"Q should have shape ({self.observation_space.n}, {self.action_space.n}). Got {Q.shape}.")
