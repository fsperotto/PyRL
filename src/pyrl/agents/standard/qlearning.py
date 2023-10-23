from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
#from gymnasium.spaces import Space

from pyrl import Agent


#--------------------------------------------------------------    

class QLearning(Agent):
    """The QLearning class"""

    #--------------------------------------------------------------    
    def __init__(self, env, 
                 default_action=None, 
                 discount=0.9, learning_rate=0.1, exploration_rate=None, should_explore:Callable=None, 
                 initial_Q=None, initial_Q_value=None,
                 name="Q-Learning", 
                 remember_prev_a=False,
                 store_N_sa=False, store_N_saz=False, store_N_z=False, store_N_a=False,
                 store_V=False, store_policy=True):

        super().__init__(env, 
                         default_action=default_action,
                         name=name,
                         remember_prev_s=True, remember_prev_a=remember_prev_a,
                         store_N_sa=store_N_sa, store_N_saz=store_N_saz, store_N_z=store_N_z, store_N_a=store_N_a)
        
        self.initial_Q = initial_Q
        self.initial_Q_value = initial_Q_value

        if initial_Q is not None:
            self.Q_check(initial_Q)
        #    self.initial_Q = initial_Q
        #elif initial_Q_value is not None:
        #    self.initial_Q_value = initial_Q_value
        
        #parameters
        self.discount = discount                   #gamma
        self.learning_rate = learning_rate         #alpha
        self.exploration_rate = exploration_rate   #epsilon
        
        #epsilon as a function
        if should_explore is not None:
            self.should_explore = should_explore  
        elif isinstance(exploration_rate, float):
            self.should_explore = self._builtin_epsilon_should_explore
        elif isinstance(exploration_rate, dict):
            self.should_explore = self._builtin_epsilon_should_explore
        elif isinstance(exploration_rate, Iterable):
            self.should_explore = self._builtin_epsilon_should_explore
        else:
            self.should_explore = self._builtin_log_decreasing_should_explore

        # Q(s, a) table
        self.Q = None
        
        self.store_V = store_V
        if store_V:
           self.V = None
        
        self.store_policy = store_policy
        if store_policy:
           self.policy = None


    #--------------------------------------------------------------    
    def reset(self, initial_observation, *, 
              reset_knowledge=True, learning_mode='off-policy',
              initial_budget=None, reset_budget=True):
       
        if reset_knowledge:
           
            if self.initial_Q is not None:
                self.Q = self.initial_Q
                
            elif self.initial_Q_value is not None:
                self.Q = np.full(self.observation_shape + self.action_shape, self.initial_Q_value, dtype=float)
                
            else:
                self.Q = np.random.sample(self.observation_shape + self.action_shape)
                
            if self.store_V:
               self._reset_V()
               #self.V = np.zeros(self.observation_shape)

            if self.store_policy:
               self._reset_policy()
               #self.policy = np.zeros(self.observation_shape + self.action_shape)

        super().reset(initial_observation=initial_observation, 
                      reset_knowledge=reset_knowledge, learning_mode=learning_mode,
                      initial_budget=initial_budget, reset_budget=reset_budget)

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
           
           # V(s) is max Q(s,a)
           v = self.V[obs] if self.store_V else self.Q[obs].max()
              
           # Pi(s) is arg max Q(s,a)
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
    def _choose(self) :
        #if the agent was not reseted after initialization, then reset

        if self.should_explore(self):

            a = self.action_space.sample()

        else:
        
            if self.store_policy:
               
               a = np.random.choice(np.flatnonzero(self.policy[self.get_state_tpl()]))
               
            else:
           
               #Q_s = np.take(self.Q, self.get_state_tpl
               Q_s = self.Q[self.get_state_tpl()]
               #Q_s = self.Q[self.s]
               
               if self.store_V:
                  maxq = Q_s.max()
               else:
                  maxq = Q_s.max()
                  
               a = np.random.choice(np.flatnonzero(Q_s == maxq))

        return a


    #--------------------------------------------------------------    
    def _learn(self) -> None:
        
        #self.Q[self.last_state, self.last_action] = (1 - self.learning_rate) * self.Q[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.Q[self.current_state, :].max())
        
        # get S , A, and S'
        index_s = self.get_state_tpl(self.prev_s)
        index_sa = self.get_state_tpl(self.prev_s) + self.get_action_tpl()
        index_next_s = self.get_state_tpl()
        #index_next_s = self.s
        
        #get current Q(s,a)
        Q_sa = self.Q[index_sa]
        
        #if in fast mode, get V(s') from stored table
        if self.store_V:
           V_next_s = self.V[index_next_s]
        #if in light mode, calculate V(s') from max_a'[Q(s',a')]
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
    def _builtin_log_decreasing_should_explore(self, agent: Agent) -> bool:
        return np.random.random() < (1 / math.log(self.t + 2))
    
    def _builtin_epsilon_should_explore(self, agent: Agent) -> bool:
        return np.random.rand() < self.exploration_rate
    

    #--------------------------------------------------------------    
    def Q_check(self, Q: np.ndarray) -> None:
        if Q.shape is not (self.observation_space.n, self.action_space.n):
            raise ValueError(f"Q should have shape ({self.observation_space.n}, {self.action_space.n}). Got {Q.shape}.")
