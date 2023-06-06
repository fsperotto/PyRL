#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Classes.

This module implements abstract classes, directly available from PyRL module:
   
   - Agent : the agent, implementing a controller 
   
   - Env : environment, implementing the controlled system (the problem)
   
   - Sim : simulator
   
"""

__version__ = "0.0.1"
__author__ = "Filipo S. Perotto, Aymane Ouahbi, Melvine Nargeot"
__license__ = "MIT"
__status__ = "Development"

################

import numpy as np
#from abc import ABC, abstractmethod  #ABC is abstract base class
from collections.abc import Iterable

###################################################################

class Agent():
    """
    Agent Class
    
    It represents the controller, interacting with the system, also called environment.
    
        Parameters:
            states : list (or iterable)
                the list of variables that constitute the space of states. 
                e.g.: [4, 10] means two categorical variables assuming 4 and 10 different values, respectively.
            actions : list (or iterable)
                the list of variables that constitute the space of actions. 
            num_state_vars : int
                number of variables that represent the state space.
            num_action_vars : int
                number of variables that represent the action space.
            num_flat_states : int
                number of possible flat states (all the different combinations of state variables values)
            num_flat_actions : int
                number of possible flat actions (all the different combinations of action variables values, i.e. joint actions)
            t : int
                the current time-step or round during execution, $t \in \mathbb{N}$.
            s : list
                current state, from the last observation.
            r : float
                last received reward.
    """
    
    def __init__(self, observation_space, action_space, initial_observation=None):
        """Agent Constructor. The dimensions concerning observable states and actions must be informed."""
        #observations (what the agent perceives from the environment state)
        #self.states  = states  if isinstance(states, Iterable)  else  [states]
        #self.num_state_vars = len(states)
        #self.num_flat_states = np.prod(self.states)
        #actions
        #self.actions = actions if isinstance(actions, Iterable) else  [actions]
        #self.num_action_vars = len(actions)
        #self.num_flat_actions = np.prod(self.actions)
        #reset
        self.observation_space  = observation_space
        self.action_space = action_space
        self.reset(initial_observation)
        
    def reset(self, initial_observation, reset_knowledge=True):
        """
        Reset $t, r, s, a$, and can also reset the learned knowledge.
        
            Parameters:
                s (list): the initial state of the environment, observed by the agent
                reset_knowledge (bool) = True : if the agent should be completely reseted

            Returns:
                action : list representing the joint action chosen by the controller
        """
        #time, or number of elapsed rounds 
        self.t = 0   
        #memory of the current state and last received reward
        self.s = initial_observation  if isinstance(initial_observation, Iterable)  else  [initial_observation]
        self.r = 0.0
        #next chosen action
        #self.a = [None for _ in range(self.num_action_vars)] 
        self.a = self.action_space.sample()

    def act(self) -> list :
        """
        Choose an action to execute, and return it.
        
            Parameters:
                a (int): A decimal integer
                b (int): Another decimal integer

            Returns:
                action : list representing the joint action chosen by the controller
                
        """
        #choose an action
        # self.a = [...]
        #return the chosen action
        return self.a
        
    def observe(self, s, r):
        """Memorize the observed state and received reward."""
        self.s = s  if isinstance(s, Iterable)  else  [s]
        self.r = r
        
    def learn(self):
        pass
        
###################################################################
        
class Env():
    """
    Environment Class
    
    It represents the system to be controlled by an agent.
    """
    
    def __init__(self, states=[2], actions=[2]):
        self.t = 0
        self.states  = states  if isinstance(states, Iterable)  else  [states]
        self.actions = actions if isinstance(actions, Iterable) else  [actions]
        self.reset()
        
    def reset(self):
        self.t = 0
        self.s = [0 for _ in range(len(self.states))] 
        self.r = 0.0
        self.done = False

    def step(self, a):
        self.t += 1
        return self.s, self.r, self.done
                
###################################################################
  
        
class Sim():
    """
    Simulator Class
    
    """
    
    def __init__(self, agents, envs, episode_horizon=100, num_episodes=1, num_simulations=1,
                 episode_finished_callback=None, simulation_finished_callback=None, round_finished_callback=None):
        self.agents = agents  if isinstance(agents, Iterable)  else  [agents]
        self.envs   = envs    if isinstance(envs, Iterable)    else  [envs]
        #self.logger = logger
        self.episode_horizon = episode_horizon
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations
        self.round_finished_callback = round_finished_callback
        self.episode_finished_callback = episode_finished_callback
        self.simulation_finished_callback = simulation_finished_callback
        
    def reset(self):
        pass


    def run(self, episode_horizon=None, num_episodes=None, num_simulations=None):
        
        episode_horizon = episode_horizon  if  episode_horizon is not None else self.episode_horizon 
        num_episodes = num_episodes  if  num_episodes is not None  else  self.num_episodes
        num_simulations = num_simulations  if  num_simulations is not None   else  self.num_simulations
        
        for env in self.envs:
            for agent in self.agents:
                                
                for i in range(num_simulations):
                
                    observation, info = env.reset()
                    agent.reset(observation)
                    
                    for j in range(num_episodes):
                        
                        observation, info = env.reset()
                        agent.reset(observation, reset_knowledge=False)
                        
                        for t in range(1, episode_horizon+1):
                            
                            action = agent.act()  # agent policy that uses the observation and info
                            observation, reward, terminated, truncated, info = env.step(action)
                            agent.observe(observation, reward)
                            agent.learn()

                            if self.round_finished_callback is not None:
                                try:
                                    self.episode_finished_callback(env, agent)
                                except Exception as e:
                                    print(str(e))

                            if terminated or truncated:
                                break
                                #observation, info = env.reset()        

                        if self.episode_finished_callback is not None:
                            try:
                                self.episode_finished_callback(env, agent)
                            except Exception as e:
                                print(str(e))
                            
                    if self.simulation_finished_callback is not None:
                       try:
                           self.simulation_finished_callback(env, agent)
                       except Exception as e:
                            print(str(e))
                           