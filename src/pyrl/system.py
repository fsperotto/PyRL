# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:15:16 2023

@author: fperotto
"""

###################################################################

from typing import Callable, Union


###################################################################

class System():
    """
    Control System Class, with agent and environment
    """
    
    #--------------------------------------------------------------    
    def __init__(self, env, agent, observation_function:Union[Callable,None]=None):
        self.env = env
        self.agent = agent
        if observation_function is not None:
            self.observation_function = observation_function
        else:
            self.observation_function = self._observation_function
        
    #--------------------------------------------------------------    
    def reset(self):
        initial_state, info = self.env.reset()
        initial_observation = self.observation_function(initial_state)
        self.agent.reset(initial_observation)
        return initial_state, initial_observation, info
    
    #--------------------------------------------------------------    
    def step(self):
        action = self.agent.choose_action()
        state, reward, terminated, truncated, info = self.env.step(action)
        observation = self.observation_function(state)
        self.agent.observe(observation, reward, terminated, truncated)
        return state, observation, action, reward, terminated, truncated, info

    #--------------------------------------------------------------    
    def _observation_function(self, state):
        return state
        
###################################################################
