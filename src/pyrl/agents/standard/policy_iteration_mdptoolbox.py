# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:58:28 2023

@author: fperotto
"""

from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
#from gymnasium.spaces import Space

from pyrl import Agent

from mdptoolbox.mdp import MDP, PolicyIteration, PolicyIterationModified, FiniteHorizon


###############################################################################

class PolicyIteration_MDPtoolbox(Agent):
    """Policy Iteration class"""

    #--------------------------------------------------------------    
    def __init__(self, env, R=None, P=None,
                 default_action=None,
                 discount=0.99,  
                 initial_policy=None,
                 max_iter=1000, eval_type="iterative", # "iterative" or iterative
                 name="PolicyIteration"):
        
        super().__init__(env, 
                         default_action=default_action,
                         name=name)
        
        #parameters
        self.discount = discount                   #gamma

        #initial policy
        self.initial_policy = initial_policy
        
        if R is not None:
           self.R = R
        else:
           self.R = env.get_reward_matrix()

        if P is not None:
           self.P = P
        else:
           self.P = env.get_transition_matrix()
        
        self.mdp = PolicyIteration( self.P, self.R, discount=discount, max_iter=max_iter, eval_type=eval_type)
        #self.mdp = PolicyIteration(self.P, self.R, discount=discount, max_iter=1000, eval_type="iterative")
        
        self.Q = None
        self.V = None
        self.policy = None
        
        self._plan()
        

    #--------------------------------------------------------------    
    def _plan(self):
   
        self.mdp.run()

        #print(self.mdp.policy)
        #fact_policy = np.array(self.mdp.policy).reshape(self.observation_shape)
        #print(fact_policy)
        self.policy = np.zeros( self.observation_shape + self.action_shape, dtype=float)
        self.V = np.zeros(self.observation_shape, dtype=float)
        for obs_idx, obs_tpl in enumerate(self.observation_iterator):
        #for obs_idx in range(self.observation_comb):
        #   obs_tpl = np.unravel_index(obs_idx, self.observation_shape)
           act_idx = self.mdp.policy[obs_idx]
           act_tpl = np.unravel_index(act_idx, self.action_shape)
           self.policy[obs_tpl + act_tpl] = 1.0
           self.V[obs_tpl] = self.mdp.V[obs_idx]
        
        #self.Q = np.reshape(self.mdp.Q, self.observation_shape + self.action_shape)
        #self.V = np.reshape(self.mdp.V, self.observation_shape)
      
    #--------------------------------------------------------------    
    def _choose(self):
        
        #self.a = np.random.choice(np.flatnonzero(self.policy[self.get_state_tpl()]))
        self.a = self.mdp.policy[self.get_state_idx()]
        
        return self.a

###############################################################################

#class FiniteHorizon_MDPtoolbox(Agent):
#    """Finite Horizon Backwards class"""
"""
    #--------------------------------------------------------------    
    def __init__(self, env, R=None, P=None,
                 default_action=None, budgeted=None,
                 horizon=1000,  
                 initial_policy=None,
                 name="FiniteHorizonBackwards"):
        
        super().__init__(env, 
                         budgeted=budgeted,
                         default_action=default_action,
                         name=name)
        
        #parameters
        self.horizon = horizon

        #initial policy
        self.initial_policy = initial_policy
        
        if R is not None:
           self.R = R
        else:
           self.R = env.get_reward_matrix()

        if P is not None:
           self.P = P
        else:
           self.P = env.get_transition_matrix()
        
        self.mdp = FiniteHorizon( self.P, self.R, discount=discount)
        self.mdp.run()
        
        #self.Q = self.mdp.Q
        self.V = self.mdp.V
        #self.policy = self.mdp.policy

      
    #--------------------------------------------------------------    
    def _choose(self):
        
        #self.a = np.random.choice(np.flatnonzero(self.policy[self.get_state()]))
        self.a = self.mdp.policy[self.get_state_idx()]
        
        return self.a
"""