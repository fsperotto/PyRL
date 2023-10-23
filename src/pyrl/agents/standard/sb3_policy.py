# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:53:50 2023

@author: fperotto
"""


from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
#from gymnasium.spaces import Space

from pyrl import Agent, ensure_tuple

from stable_baselines3.common.vec_env import DummyVecEnv

#from stable_baselines3.common.utils import obs_as_tensor
import torch as th


class SB3Policy(Agent):
    """Wrapper Agent receiving a SB3 model"""

    #--------------------------------------------------------------    
    def __init__(self, env, model,
                 default_action=None, budgeted=None,
                 training_steps=200000,
                 name="SB3Policy"):

        #env = DummyVecEnv([lambda: env])
        #env = model.get_env()
        
        super().__init__(env, 
                         budgeted=budgeted,
                         default_action=default_action,
                         name=name)
        
        self.training_steps=training_steps
        
        #the sb3 model to be wrapped
        self.model = model

        # Q(s, a) table
        self.Q = None
        # V(s) table
        self.V = None
        #policy
        self.policy = None
        
    #--------------------------------------------------------------    
    def reset(self, initial_observation, reset_knowledge=True, reset_budget=True, initial_budget=None, learning_mode=None):
    
        if reset_knowledge:
           self._plan()
           
           self.Q = np.zeros( self.observation_shape+self.action_shape, dtype=float)
           self.policy=np.zeros( self.observation_shape+self.action_shape, dtype=float)

           for obs in self.observation_iterator:
              obs_th = self.model.q_net.obs_to_tensor(np.array(obs))[0]
              #dis = model.policy.get_distribution(obs_th)
              #probs = dis.distribution.probs
              #probs_np = probs.detach().numpy()
              self.Q[obs] = self.model.q_net(obs_th).numpy(force=True)
              #self.policy[obs] = np.flatnonzero(self.Q[obs]==self.Q[obs].max())
              action = self.model.predict(obs, deterministic=True)
              action = ensure_tuple(action)
              self.policy[obs + action] = 1.0
            

        super().reset(initial_observation=initial_observation, 
                      reset_knowledge=reset_knowledge,
                      reset_budget=reset_budget, initial_budget=initial_budget,
                      learning_mode=learning_mode)

    #--------------------------------------------------------------    
    def _choose(self):
        
        obs = self.get_state_tpl()
        action, _states = self.model.predict([obs], deterministic=True)
        
        self.a = action[0]

        return self.a


    #--------------------------------------------------------------    
    def _plan(self, total_timesteps=None, progress_bar=True) -> None:
       
       if total_timesteps is None:
          total_timesteps = self.training_steps
          
       # Train the agent and display a progress bar
       self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

    #--------------------------------------------------------------
#    @cache
#    @property
#    def policy(self):
        
