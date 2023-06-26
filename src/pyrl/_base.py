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

import gymnasium as gym
from gymnasium.spaces import Space, Discrete, MultiDiscrete
from gymnasium.spaces.utils import flatdim, flatten_space


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

    def __init__(self, observation_space, action_space, initial_observation=None, initial_budget=None):
        """
        Agent Constructor. 
        The dimensions concerning observable states and actions must be informed.
        """

        #observations (what the agent perceives from the environment state)
        self.observation_space = observation_space
        #self.observation_space  = observation_space  if isinstance(observation_space, Iterable)  else  [observation_space]
        #self.num_state_vars = len(observation_space)
        #self.num_flat_states = np.prod(self.observation_space)
        self.num_flat_states = flatdim(flatten_space(observation_space))
        
        #actions
        self.action_space = action_space
        #self.action_space = action_space if isinstance(action_space, Iterable) else  [action_space]
        #self.num_action_vars = len(action_space)
        #self.num_flat_actions = np.prod(self.action_space)
        self.num_flat_actions = flatdim(flatten_space(action_space))
        
        #reset
        self.reset(initial_observation)
        self.terminal = False
        
        self.initial_budget = initial_budget


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

    def act(self):
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

    def observe(self, s, r, terminated, truncated):
        """Memorize the observed state and received reward."""
        self.s = s  #if isinstance(s, Iterable)  else  [s]
        self.r = r
        self.terminated = terminated
        self.truncated = truncated

        self.t = self.t + 1

        if self.initial_budget is not None:
            self.initial_budget = self.initial_budget + r

    def learn(self):
        pass

###################################################################
        
class Env(gym.Env):

    """
    Environment Class

    It represents the system to be controlled by an agent.
    """

    metadata = {}

    def __init__(self, observation_space=[2], action_space=[2], render_mode=None):
        
        self.t = 0
        
        if isinstance(observation_space, int):
            self.observation_space = Discrete(observation_space)
        elif isinstance(observation_space, Iterable):
            self.observation_space = MultiDiscrete(shape=observation_space)
        else:
            self.observation_space = observation_space

        if isinstance(action_space, int):
            self.action_space = Discrete(action_space)
        elif isinstance(action_space, Iterable):
            self.action_space = MultiDiscrete(shape=action_space)
        else:
            self.action_space = action_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference to the window that we draw to. 
        `self.clock` will be a clock that is used to ensure that the environment is rendered at the correct framerate in human-mode. 
        They will remain `None` until human-mode is used for the first time.
        """
        self.window = None
        self.clock = None    
        
        self.reset()

    def reset(self, *, seed:int=None, options:dict=None) -> tuple:
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.t = 0

        self.s = [0 for _ in range(len(self.states))]
        self.r = 0.0
        self.terminated = False
        self.truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
        

    def step(self, action):
        self.t += 1
        return self.s, self.r, self.terminated, self.truncated
        
    def _get_obs(self):
        """
        translates the environment’s state into an observation
        """
        pass

    def _get_info(self):
        """
        method for the auxiliary information (a dict) that is returned by step and reset
        """
        return {"time-step": self.t}        

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
                            agent.observe(observation, reward, terminated, truncated)
                            agent.learn()

                            if self.round_finished_callback is not None:
                                try:
                                    self.episode_finished_callback(env, agent)
                                except Exception as e:
                                    print(str(e))

                            if terminated or truncated:
                                break
                                #observation, info = env.reset()
                            
                            if agent.initial_budget is not None:
                                if agent.initial_budget <= 0:
                                    break

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
