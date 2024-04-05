#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Classes.

This module implements abstract classes, directly available from PyRL module:

   - Sim : simulator

"""

__version__ = "0.0.1"
__author__ = "Filipo S. Perotto, Aymane Ouahbi, Melvine Nargeot"
__license__ = "MIT"
__status__ = "Development"

################

import sys
from typing import Iterable, Callable, List, Union

from pyrl.agent import Agent
from pyrl.env import Env, EnvWrapper

from numba import njit

###################################################################

            





            
            
            
class EventBasedObject():

    def __init__(self):
        self._listeners = {}
   
    def add_listener(self, name:str, listeners:Union[Callable, List[Callable]]):
        if isinstance(listeners, Callable):
            listeners = [listeners]
        if len(listeners) > 0:
            if name in self._listeners.keys():
                self._listeners[name] = self._listeners[name] + listeners
            else:
                self._listeners[name] = listeners
         
    def _evoke_listeners(self, name:str, *args, **kwargs):
        if name in self._listeners.keys():
            for callback in self._listeners[name]:
                return_cancel = callback(self, *args, **kwargs)
                if return_cancel is not None:
                   break

    def clear_listeners(self, name:str=None):
        if name is not None:
           if name in self._listeners.keys():
              self._listeners.pop(name)
        else:
           self._listeners.clear()

###################################################################


class Sim(EventBasedObject):
    """
    Simulator Class

    """

    #--------------------------------------------------------------    
    def __init__(self, agents:Union[Agent, Iterable], env:Union[Env, EnvWrapper], 
                 episode_horizon:int=100, num_episodes:int=1, num_repetitions:int=1,
                 close_on_finish=True):
                 #renderers=[]
        
        super().__init__()
        
        #self.envs = envs if isinstance(envs, Iterable) else [envs]
        self.env = env 

        self.agents = agents if isinstance(agents, Iterable) else [agents]
        self.num_agents = len(self.agents)
        
        self.episode_horizon = episode_horizon
        self.num_episodes = num_episodes
        self.num_repetitions = num_repetitions
        
        self.close_on_finish = close_on_finish
        
        self.reset()
        # self.metrics = {"time": 0, "exploration": []}

        #self.metrics = dict(
        #    time = 0,
        #    exploration = np.zeros((self.envs[0].observation_space.n, self.envs[0].action_space.n)),
        #    budget = np.zeros((self.episode_horizon,), dtype=int)
        #)

    #--------------------------------------------------------------    
    def reset(self):

        self.finished = False

        #self.env = None
        #self.env_idx = -1

        self.rep = -1

        self.agent = None
        self.agent_idx = -1

        self.ep = -1

        self.t = -1
        
        self.episode_finished = True
        self.simulation_finished = True
        self.repetition_finished = True
        self.environment_finished = True


    #--------------------------------------------------------------    
    def step(self):
       
        #if self.ready and not self.finished:
        if not self.finished:
           
            if self.episode_finished:
               
               if self.simulation_finished:
                  
                  if self.repetition_finished:

                     #if self.environment_finished:
                     #   
                     #   #next_environment
                     #   self.env_idx += 1
                     #   self.env = self.envs[self.env_idx]
                     #
                     #   self.environment_finished = False
                     #
                     #   self.rep = -1
                     #    
                     #   #env started event callback
                     #   self._evoke_listeners('environment_started')
                        
                     #next_repetition
                     self.rep += 1

                     self.repetition_finished = False

                     self.agent_idx =-1
                     self.agent = None
            
                     #repetition started event callback
                     self._evoke_listeners('repetition_started')
                     
                  #next_simulation
                  self.agent_idx += 1
                  self.agent = self.agents[self.agent_idx]

                  self.simulation_finished = False
            
                  self.ep = -1
            
                  #simulation started event callback
                  self._evoke_listeners('simulation_started')                  
               
               #next episode
               self.t = 0
   
               self.ep += 1
               self.episode_finished = False
                
               #observation, initial_budget, info = self.env.reset()
               observation, info = self.env.reset()
               is_first_episode = (self.ep==0)
               self.agent.reset( observation, initial_budget=self.env.initial_budget, reset_knowledge=is_first_episode)
   
               #episode started event callback
               self._evoke_listeners('episode_started')

            #episode is not finished, next round
            else:               
               
               self.t += 1
          
               #round started event callback
               self._evoke_listeners('round_started')
  
               #action = self.agent.choose_action()  # agent policy that uses the observation and info
               #action = self.agent._choose()  # agent policy that uses the observation and info
               
               action = self.agent.a
               #action = self.agent.get_action()
               
               observation, reward, terminated, truncated, info = self.env.step(action)
                      
               self.agent.step(observation, reward, terminated, truncated)

               #round finished event callback
               self._evoke_listeners('round_finished')

               ruined = False
               if self.agent.b is not None:
                   if self.agent.b <= 0:
                       ruined = True
   
               if (self.t >= self.episode_horizon):
                  truncated = True
                  
               if terminated or truncated or ruined:
                  
                  self.episode_finished = True
                  self._evoke_listeners('episode_finished')
   
                  if self.ep >= self.num_episodes-1:
                
                     self.simulation_finished = True
                     self._evoke_listeners('simulation_finished')
                     
                     if self.agent_idx >= len(self.agents)-1:
   
                        self.repetition_finished = True
                        self._evoke_listeners('repetition_finished')
                        
                        if self.rep >= self.num_repetitions-1:
                        
                           # self.environment_finished = True
                           # self._evoke_listeners('environment_finished')
                           # 
                           # if self.env_idx >= len(self.envs)-1:
              
                           self.finished = True
                           
                           if self.close_on_finish:
                              self.env.close()


    #--------------------------------------------------------------    
    def run(self, steps=None):
       
       if not self.finished:
       
          try:
   
             #run a precised number of steps                   
             if isinstance(steps, int):

                for i in range(steps):
                   self.step()
                   if self.finished:
                      break
                
             elif steps == 'episode':   
                self.step()
                while not self.episode_finished:
                   self.step()

             elif steps == 'simulation':   
                self.step()
                while not self.simulation_finished:
                   self.step()

             elif steps == 'repetition':   
                self.step()
                while not self.repetition_finished:
                   self.step()

             #elif steps == 'environment':   
             #   while not self.environment_finished:
             #      self.step()

             #run until the end
             else:
                
                while not self.finished:
                   self.step()
             
          
          except KeyboardInterrupt:
             self.close()
             print("KeyboardInterrupt: simulation interrupted by the user.")
             sys.exit()
   
          except:
             self.close()
             raise
             
          if self.close_on_finish:
             self.close()


    #--------------------------------------------------------------    
    def close(self):
       
       if self.env is not None:
          self.env.close()
