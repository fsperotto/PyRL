# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:58:28 2023

@author: fperotto
"""

from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
#from gymnasium.spaces import Space

from pyrl import Agent, ensure_tuple

from tqdm import tqdm

###############################################################################

class PolicyIteration(Agent):
    """Policy Iteration class"""

    #--------------------------------------------------------------    
    def __init__(self, env,
                 R=None, reward_mode=None,
                 P=None, transition_mode=None,
                 default_action=None,
                 discount=0.9,  
                 initial_policy=None,
                 theta=0.001, max_value_iterations=1000, max_policy_iterations=1000,
                 name="PolicyIteration"):
        
        super().__init__(env, 
                         default_action=default_action,
                         name=name)
        
        #parameters
        self.discount = discount                   #gamma

        #initial policy
        self.initial_policy = initial_policy
        
        self.theta = theta
        self.max_policy_iterations = max_policy_iterations
        self.max_value_iterations = max_value_iterations
        
        if R is not None:
           self.R = R
        else:
           self.R = env.get_reward_matrix()

        if reward_mode is not None:
           self.reward_mode = reward_mode
        else:
           self.reward_mode = env.reward_mode

        if P is not None:
           self.P = P
        else:
           self.P = env.get_transition_matrix()

        # Q(s, a) table
        self.Q = None
        # V(s) table
        self.V = None
        #policy
        self.policy = None
        
    #--------------------------------------------------------------    
    def reset(self, initial_observation, reset_knowledge=True, reset_budget=True, initial_budget=None, learning_mode=None):
    
        if reset_knowledge:

           #initial policy is given
           if self.initial_policy is not None:
              self.policy = self.initial_policy
           #initial policy is random
           else:
              self.policy = np.zeros(self.observation_shape + self.action_shape)
              for obs in self.observation_iterator:
                 a = ensure_tuple(self.action_space.sample())
                 self.policy[obs + a] = 1
           
           self.Q = np.zeros(self.observation_shape + self.action_shape, dtype=float)

           self.V = np.zeros(self.observation_shape, dtype=float)
           
           self._plan()

                
        super().reset(initial_observation=initial_observation, 
                      reset_knowledge=reset_knowledge,
                      reset_budget=reset_budget, initial_budget=initial_budget,
                      learning_mode=learning_mode)
        


    #--------------------------------------------------------------    
    def _choose(self):
        #return np.random.choice(np.flatnonzero(self.policy[self.get_state_tpl()]))
        return np.unravel_index(np.argmax(self.policy[self.get_state_tpl()], axis=None), self.env.action_shape)


    #--------------------------------------------------------------    
    def _plan(self, theta=None, max_value_iterations=None, max_policy_iterations=None) -> None:
        
        if theta is None: theta = self.theta
        if max_value_iterations is None: max_value_iterations = self.max_value_iterations
        if max_policy_iterations is None: max_policy_iterations = self.max_policy_iterations
        
        new_policy = self.policy     
        old_policy = self.policy.copy()	
             
        #Loop until convergence or max iterations
        for i in tqdm(range(max_policy_iterations)): 
   	    		
           #POLICY EVALUATION : UPDATE V WITH CURRENT POLICY
           #Loop until convergence or max iterations
           for j in range(max_value_iterations): 
	           	
              #max update difference (learning progression)
              delta = 0.0
              
              old_V = self.V.copy()
              
              #Update V(s)
              for prev_obs in self.env.observation_iterator:
                 new_v = 0.0
                 #act = self.policy[prev_obs].argmax()
                 for act in self.env.action_iterator:
                     act_prob = self.policy[prev_obs + act]
                     if act_prob > 0.0:
                        for next_obs in self.env.observation_iterator:
                           transition_prob = self.P[prev_obs + act + next_obs]
                           expected_reward = self.R[prev_obs + act + next_obs]
                           next_value = old_V[next_obs]
                           value_fragment = act_prob * transition_prob * (expected_reward + (self.discount * next_value))
                           new_v += value_fragment
                 
                 delta = max(delta, abs(old_V[prev_obs] - new_v))
   
                 self.V[prev_obs] = new_v 
                 
              #if value convergence, stop iterations
              if (delta < theta):
                 #print("value converged")
                 break
           #V CONVERGED    
   
           #UPDATE Q
           for prev_obs in self.env.observation_iterator:
              for act in self.env.action_iterator:
                 new_q = 0.0
                 for next_obs in self.env.observation_iterator:
                    transition_prob = self.P[prev_obs + act + next_obs]
                    expected_reward = self.R[prev_obs + act + next_obs]
                    next_value = self.V[next_obs]
                    value_fragment = transition_prob * (expected_reward + (self.discount * next_value))
                    new_q += value_fragment
                 self.Q[prev_obs + act] = new_q
	   	   #Q UPDATED		
                 
           #UPDATE POLICY
           self.policy.fill(0.0)
           for obs in self.env.observation_iterator:
              #v = self.V[obs]
              Q_s = self.Q[obs]
              max_q = Q_s.max()
              a_policy = np.unravel_index(np.argmax(Q_s == max_q, axis=None), self.env.action_shape)
              #a_policy = np.random.choice(np.flatnonzero(Q_s == max_q))
              #for a in self.action_iterator:
              #   if (a == a_policy):
              #      self.policy[obs + a] = 1.0
              #   else:
              #      self.policy[obs + a] = 0.0
              #self.policy[obs].fill(0.0)
              self.policy[obs + ensure_tuple(a_policy)] = 1.0
           #POLICY UPDATED
           
           #verify policy convergence
           if np.array_equiv(old_policy, new_policy):
              #print("policy converged.")
              break
        #POLICY CONVERGED
           