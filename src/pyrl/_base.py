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
#from collections.abc import Iterable
from typing import Iterable, Callable, TypeVar, Generic, Tuple, List, Union

import gymnasium as gym
from gymnasium.spaces import Space, Discrete, MultiDiscrete
from gymnasium.spaces.utils import flatdim, flatten_space


###################################################################

def pyrl_space(space_or_dimensions:Union[int, Iterable[int], Space]) -> (Space, Union[int, None], Union[int, None]):
    """
    Function to convert the representation of a state_space, an observation_space, or an action_space

        Return:
            space 
            num_vars : int
                number of variables that represent the space.
            num_comb : int
                number of possible flat states (all the different combinations of variables values)
    """

    space = space_or_dimensions
    
    if isinstance(space_or_dimensions, int):
        space = MultiDiscrete([space_or_dimensions])
    elif isinstance(space_or_dimensions, Iterable):
        space = MultiDiscrete(space_or_dimensions)
    #elif isinstance(space_or_dimensions, Discrete):
    #    space = MultiDiscrete([space_or_dimensions.n])
    
    num_vars = None
    num_comb = None
    shape=None
    if isinstance(space, Discrete):
        shape = (space.n,)
        num_vars = 1
        num_comb = space.n
    elif isinstance(space, MultiDiscrete):
        shape = tuple(space.nvec[::-1])
        num_vars = space.nvec.size
        num_comb = np.prod(space.nvec)
        #num_comb = flatdim(flatten_space(space))
    
    return space, shape, num_vars, num_comb
            
       
###################################################################


class Agent():
    """
    Agent Class

    It represents the controller, interacting with the system, also called environment.

        Parameters:
            observation_space : gym.Space or list or iterable.
                the list of variables that constitute the space of states.
                e.g.: [4, 10] means two categorical variables assuming 4 and 10 different values, respectively.
            action_space : gym.Space or list or iterable.
                the list of variables that constitute the space of actions.
            num_obs_vars : int
                number of variables that represent the state space.
            num_act_vars : int
                number of variables that represent the action space.
            num_obs_com : int
                number of possible flat states (all the different combinations of state variables values)
            num_act_comb : int
                number of possible flat actions (all the different combinations of action variables values, i.e. joint actions)
            t : int
                the current time-step or round during execution, $t \in \mathbb{N}$.
            s : list
                current state, from the last observation.
            r : float
                last received reward.
    """

    def __init__(self, observation_space, action_space,
                 default_action=None, initial_budget=None):
        """
        Agent Constructor. 
        The dimensions concerning observable states and actions must be informed.
        """
        
        #observations (what the agent perceives from the environment state)
        self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = pyrl_space(observation_space)
        
        #actions
        self.action_space, self.action_shape, self.num_act_var, self.num_act_comb = pyrl_space(action_space)

        if default_action is None:
            self.default_action = None
        #elif isinstance(default_action, int):
        #    self.default_action = np.array([default_action])
        else:
            self.default_action = np.array(default_action)

        self.initial_budget = initial_budget
        
        #time
        self.t = None
        
        @property
        def time(self):
           return self.t
        
        @property
        def time_step(self):
           return self.t

        #current state
        self.s = None
        
        @property
        def state(self):
           return self.s
        
        @property
        def current_state(self):
           return self.s

        #last received reward
        self.r = None
        
        @property
        def last_reward(self):
           return self.r
        
        @property
        def reward(self):
           return self.r
        
        
        #next chosen action
        self.a = None
        
        @property
        def action(self):
           return self.a
        
        @property
        def chosen_action(self):
           return self.a

        #current budget
        self.b = None

        @property
        def budget(self):
           return self.b
        
        @property
        def current_budget(self):
           return self.b

        self.learning = None

        self.terminated = None
        self.truncated = None
        self.ruined = None
        
        self.should_reset = True
        


    def reset(self, initial_observation, reset_knowledge=True, reset_budget=True, learning=True):
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
        
        #memory of the current state 
        self.s = initial_observation 
        ##memory of the last received reward
        #self.r = 0.0
        ##next chosen action
        #self.choose_action()
            
        #current budget
        if reset_budget:
           self.b = self.initial_budget
        
        self.terminated = False
        self.truncated = False

        if self.b is None or self.b > 0:
            self.ruined = False
        else:
            self.ruined = True
        
        self.learning = learning

        self.should_reset = False


    def choose_action(self):
        """
        Choose an action to execute, and return it.

            Returns:
                action : list representing the joint action chosen by the controller

        """
        #if the agent was not reseted after initialization, then reset
        if self.should_reset:
            raise ValueError("ERROR: Agent properties should be initilized. Maybe you forgot to call the reset method ?")
        
        #choose the default action, if defined
        if self.default_action is None:
            self.a = self.action_space.sample()
        #choose uniformly random action otherwise
        else:
            self.a = self.default_action
        #return the chosen action
        return self.a



    def observe(self, s, r, terminated, truncated):

        #if the agent was not reseted after initialization, then reset
        if self.should_reset:
            raise ValueError("ERROR: Agent properties should be initilized. Maybe you forgot to call the reset method ?")
              
        """Memorize the observed state and received reward."""
        self.s = s  #if isinstance(s, Iterable)  else  [s]
        self.r = r
        self.terminated = terminated
        self.truncated = truncated

        self.t = self.t + 1

        if self.b is not None:
            self.b = self.b + r
            if self.b <= 0:
                self.ruined = True
        
        if self.learning:
            self.learn()


    def learn(self):
        pass

    def get_state(self, s=None):
        if s is None:
           s = self.s
        if isinstance(s, Iterable):
            return tuple(s)
        else:
            return (s,)
         
    def get_action(self, a=None):
        if a is None:
           a = self.a
        if isinstance(a, Iterable):
            return tuple(a)
        else:
            return (a,)

    def get_state_action(self):
        return self.get_state() + self.get_action()
         

###################################################################

class EnvWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.env.state_space = self.env.observation_space
        self.env.t = None
        self.env.initial_state = None
        self.env.ready = False
        self.env.interrupted = False

    def reset(self, *, seed:int=None, initial_state=None, options:dict=None) -> tuple:
        
        self.env.t = 0
        self.env.initial_state = initial_state
        self.env.ready = True
        
        return self.env.reset(seed=seed, options=options)
        
        
    def step(self, action):
    
        if not self.env.ready:
            self.reset()

        self.env.t += 1

        return self.env.step(action)
     
    def show(self):
       pass


###################################################################


class Env(gym.Env):

    """
    Environment Class

    It represents the system to be controlled by an agent.
    """
    
    metadata = {}

    def __init__(self, state_space, action_space, observation_space=None,
                 initial_state=None, render_mode:Union[str,None]=None):
        
        #flag : ready to execute
        self.ready = False
        
        #time
        self.t = None
        
        #states
        self.state_space, self.state_shape, self.num_state_var, self.num_state_comb = pyrl_space(state_space)
        
        #actions
        self.action_space, self.action_shape, self.num_act_var, self.num_act_comb = pyrl_space(action_space)

        #observations (what the agent perceives from the environment state)
        if observation_space is None:
            self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = self.state_space, self.state_shape, self.num_state_var, self.num_state_comb
        else:
            self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = pyrl_space(observation_space)
        
        
        if initial_state is None:
            self.initial_state = None
        elif isinstance(initial_state, int):
            self.initial_state = np.array([initial_state])
        else:
            self.initial_state = np.array(initial_state)

            
        """
        If human-rendering is used, `self.window` will be a reference to the window that we draw to. 
        `self.clock` will be a clock that is used to ensure that the environment is rendered at the correct framerate in human-mode. 
        They will remain `None` until human-mode is used for the first time.
        """
        self.window = None
        self.clock = None    
        
        self.render_mode = render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            print("WARNING: unknown render_mode:", render_mode)
            self.render_mode = None

        #flag : user interruption
        self.interrupted = False
         
        #self.reset()


    def reset(self, *, seed:int=None, initial_state=None, options:dict=None) -> tuple:
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.ready = True
        
        self.t = 0

        if initial_state is not None:
            if isinstance(initial_state, int):
                self.s = np.array([initial_state])
            else:
                self.s = np.array(initial_state)
        elif self.initial_state is not None:
            self.s = self.initial_state
        else:
            self.s = self.state_space.sample()
        
        self.r = 0.0
        self.terminated = False
        self.truncated = False
        self.interrupted = False
        
        observation = self._get_obs()

        info = self._get_info()
        
        return observation, info
        

    def step(self, action):
    
        if not self.ready:
            self.reset()

        self.t += 1

        #action effects
        # ...
        # self.s = ...
        # self.r = ...

        observation = self._get_obs()

        info = self._get_info()

        if self.render_mode is not None:
            if not self.interrupted:
               self._render_frame()

        return observation, self.r, self.terminated, self.truncated, info

        
    def get_state(self, s=None):
        if s is None:
           s = self.s
        if isinstance(s, Iterable):
            return tuple(s)
        else:
            return (s,)

    def _get_info(self):
        """
        method for the auxiliary information (a dict) that is returned by step and reset
        """
        return {"time-step": self.t}     

    def _get_obs(self):
        """
        method for converting state to observation
        """
        return self.s
     
    def show(self):
       pass

        

###################################################################


class System():
    """
    Control System Class, with agent and environment
    """
    
    def __init__(self, env, agent, observation_function:Union[Callable,None]=None):
        self.env = env
        self.agent = agent
        if observation_function is not None:
            self.observation_function = observation_function
        else:
            self.observation_function = self._observation_function
        
    def reset(self):
        initial_state, info = self.env.reset()
        initial_observation = self.observation_function(initial_state)
        self.agent.reset(initial_observation)
        return initial_state, initial_observation, info
    
    def step(self):
        action = self.agent.choose_action()
        state, reward, terminated, truncated, info = self.env.step(action)
        observation = self.observation_function(state)
        self.agent.observe(observation, reward, terminated, truncated)
        return state, observation, action, reward, terminated, truncated, info

    def _observation_function(self, state):
        return state
        
###################################################################

        
class Sim():
    """
    Simulator Class

    """

    def __init__(self, agents, envs, episode_horizon=100, num_episodes=1, num_simulations=1,
                 simulation_started_callback=None, simulation_finished_callback=None, 
                 episode_started_callback=None, episode_finished_callback=None, 
                 round_started_callback=None, round_finished_callback=None ):
        if isinstance(agents, Agent):
            self.agents = [agents]
        else:
            self.agents = agents
        if isinstance(envs, Env) or isinstance(envs, EnvWrapper):
           self.envs   = [envs]
        else:  
           self.envs   = envs
           
        #self.logger = logger
        self.episode_horizon = episode_horizon
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations
        self.round_started_callback = round_started_callback
        self.round_finished_callback = round_finished_callback
        self.episode_started_callback = episode_started_callback
        self.episode_finished_callback = episode_finished_callback
        self.simulation_started_callback = simulation_started_callback
        self.simulation_finished_callback = simulation_finished_callback

        self.t = 0

    def reset(self):
        pass

    def run(self, episode_horizon=None, num_episodes=None, num_simulations=None):
    
        episode_horizon = episode_horizon  if  episode_horizon is not None else self.episode_horizon
        num_episodes = num_episodes  if  num_episodes is not None  else  self.num_episodes
        num_simulations = num_simulations  if  num_simulations is not None   else  self.num_simulations

        for env in self.envs:
           
            env.show()
            
            for agent in self.agents:

                for i in range(num_simulations):

                    observation, info = env.reset()
                    agent.reset(observation)

                    #simulation started event callback
                    if self.simulation_started_callback is not None:
                        self.simulation_started_callback(self, env, agent)

                    for j in range(num_episodes):

                        observation, info = env.reset()
                        is_first_episode = (j==0)
                        agent.reset(observation, reset_knowledge=is_first_episode)

                        #episode started event callback
                        if self.episode_started_callback is not None:
                            self.episode_started_callback(self, env, agent)

                        for t in range(1, episode_horizon+1):

                            #round started event callback
                            if self.round_started_callback is not None:
                                self.round_started_callback(self, env, agent)

                            action = agent.choose_action()  # agent policy that uses the observation and info
                            env.recharge_mode = hasattr(agent, "recharge_mode") and agent.recharge_mode
                            observation, reward, terminated, truncated, info = env.step(action)
                            agent.observe(observation, reward, terminated, truncated)
                            agent.learn()

                            #round finished event callback
                            if self.round_finished_callback is not None:
                                self.round_finished_callback(self, env, agent)

                            if terminated or truncated:
                                break
                                #observation, info = env.reset()

                            if agent.b is not None:
                                if agent.b <= 0:
                                    break
                                 
                            if env.interrupted:
                                print("Simulation interrupted by the user.")
                                break

                        #episode finished callback
                        if self.episode_finished_callback is not None:
                            self.episode_finished_callback(self, env, agent)

                        if env.interrupted:
                            break
                         
                    #simulation finished callback
                    if self.simulation_finished_callback is not None:
                        self.simulation_finished_callback(self, env, agent)

                    if env.interrupted:
                        break
                
                if env.interrupted:
                    break

            env.close()

            if env.interrupted:
                break

