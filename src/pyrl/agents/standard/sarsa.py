from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
from gymnasium.spaces import Space

from pyrl.agents import Qlearning

class SARSA(Qlearning):
    """The SARSA class"""

    #--------------------------------------------------------------    
    def __init__(self, observation_space, action_space,
                 default_action=None, initial_budget=None,
                 discount=0.9, learning_rate=0.1, exploration_rate=None, should_explore=None, 
                 initial_Q=None, initial_Q_value=None,
                 store_N = True, store_V = False, store_policy = True,
                 name="SARSA"):
        
        super().__init__(observation_space, action_space, 
                         initial_budget=initial_budget,
                         default_action=default_action,
                         discount=discount, learning_rate=learning_rate, exploration_rate=exploration_rate, should_explore=should_explore, 
                         initial_Q=initial_Q, initial_Q_value=initial_Q_value,
                         store_N = store_N, store_V = store_V, store_policy = store_policy,
                         name=name)
        
        #additional memory
        self.prev_last_s = None
        self.last_a = None
        
    #--------------------------------------------------------------    
    def _observe(self, state, reward: float, terminated: bool, truncated: bool) -> None:
        #if the agent was not reseted after initialization, then reset
        
        self.prev_last_s = self.prev_s
        self.prev_s = self.s
        self.prev_a = self.a

        super().observe(state, reward, terminated=terminated, truncated=truncated)



    #--------------------------------------------------------------    
    def _learn(self) -> None:
        # S  A  R  S' A'
        #self.Q[self.last_state, self.last_action] = (1 - self.learning_rate) * self.Q[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.Q[self.current_state, :].max())
        
        if self.prev_last_s is None or self.last_a is None:
            return
            
        #S and A
        index_prev_s = self.get_state_tpl(self.prev_last_s)
        index_prev_sa = index_prev_s + self.get_action_tplself.last_a)

        #S' and A'
        index_s = self.get_state_tpl(self.prev_s)
        index_sa = index_s + self.get_action()

    	oldQ = aadTableValueSA[iPrevPrevState][iPrevPrevAction];

        double dQ = dReward + (dGamma * aadTableValueSA[iPrevState][iPrevAction]);
    
        double dDiff = dQ - dOldQ;
        
        if (dDiff != 0.0) {
    
        	bUtilityModified = true;
        	
        	double dNewQ = dOldQ + dAlpha * dDiff;
        	
        	aadTableValueSA[iPrevState][iPrevAction] = dNewQ;
        	
        	updateSpecificValueSfromSA(iPrevState);
        	
        }
        
        if self.store_N:
           self.N[index_sa] += 1
           
        # S''
        index_next_s = self.get_state_tpl()
        #index_next_s = self.s
        
        Q_prev_sa = self.Q[index_prev_sa]
        
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
