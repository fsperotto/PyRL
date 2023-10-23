# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 23:08:46 2023

@author: fperotto

Base Classes.

This module implements abstract classes, directly available from PyRL module:

   - Env : environment, implementing the controlled system (the problem)

"""

from typing import Union
from itertools import product

import numpy as np

from pyrl.space import pyrl_space, ensure_tuple

import gymnasium

###############################################################################

class Env(gymnasium.Env):

    """
    PyRL Environment Class

    It represents the system to be controlled by an agent.
    
    Inspired on the main Gymnasium class for implementing Reinforcement Learning Agents environments:

    *The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the :meth:`step` and :meth:`reset` functions.
    An environment can be partially or fully observed by single agents.*
    
    The main API methods that users of this class need to know are:

     - :meth:`step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions,
       if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
     - :meth:`reset` - Resets the environment to an initial state, required before calling step.
       Returns the first agent observation for an episode and information, i.e. metrics, debug info.

     If using integrated environment rendering:

     - :meth:`render` - Renders the environments to help visualise what the agent see, examples modes are "human", "rgb_array", "ansi" for text.
     - :meth:`close` - Closes the environment, important when external software is used, i.e. pygame for rendering, databases
    
     Environments have additional attributes for users to understand the implementation

     - :attr:`action_space` - The Space object corresponding to valid actions, all valid actions should be contained within the space.
     - :attr:`state_space` - The Space object corresponding to the state of the environment.
     - :attr:`observation_space` - The Space object corresponding to valid observations (the agent's perception on the state), all valid observations should be contained within the space.
     - :attr:`reward_range` - A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode.
       The default reward range is set to :math:`(-\infty,+\infty)`.
     - :attr:`spec` - An environment spec that contains the information used to initialize the environment from :meth:`gymnasium.make`
     - :attr:`metadata` - The metadata of the environment, 
     - :attr:`np_random` - The random number generator for the environment. This is automatically assigned during
       ``super().reset(seed=seed)`` and when assessing ``self.np_random``.

     Note:
         To get reproducible sampling of actions, a seed can be set with ``env.action_space.seed(123)``.

      Note:
         Differently from Gym.Env, in PyRL it is prefereable to use external rendering.
     """

    
    metadata = {}

    #--------------------------------------------------------------    
    def __init__(self, state_space, action_space, observation_space=None,
                 initial_state=None, render_mode:Union[str,None]=None,
                 default_initial_budget=None,
                 reset_return=['obs', 'done'],
                 name="Environment"):
        
        self.name = name
       
        #flag : ready to execute
        self.ready = False
        
        #time
        self.t = None
        # print("state_space.nvec =", state_space.nvec)
        # state_space.nvec = state_space.nvec[::-1]
        #states
        self.state_space, self.state_shape, self.state_ndim, self.state_comb, self.state_factors = pyrl_space(state_space)
        
        #actions
        self.action_space, self.action_shape, self.action_ndim, self.action_comb, self.action_factors = pyrl_space(action_space)

        #observations (what the agent perceives from the environment state)
        if observation_space is None:
            self.observation_space, self.observation_shape, self.observation_ndim, self.observation_comb, self.observation_factors = self.state_space, self.state_shape, self.state_ndim, self.state_comb, self.state_factors
        else:
            self.observation_space, self.observation_shape, self.observation_ndim, self.observation_comb, self.observation_factors = pyrl_space(observation_space)
        
        
        if initial_state is None:
            self.initial_state = None
        else:
            #self.initial_state = ensure_tuple(initial_state)
            self.initial_state = initial_state
        #elif isinstance(initial_state, int):
        #    self.initial_state = np.array([initial_state])
        #else:
        #    self.initial_state = np.array(initial_state)

        self.reward_matrix_mode = None        # s'  sa   sas'   ass'  a   as'
    
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
        
        self.default_initial_budget = default_initial_budget
         
        #self.reset()

    #--------------------------------------------------------------    
    @property
    def state_iterator(self):
       return product( *map(range, self.state_shape) )

    @property
    def action_iterator(self):
       return product( *map(range, self.action_shape) )

    @property
    def observation_iterator(self):
       return product( *map(range, self.observation_shape) )

    #--------------------------------------------------------------    
    def reset(self, *, seed:int=None, initial_state=None, initial_budget=None, options:dict=None) -> tuple:
        
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
        
        if initial_budget is not None:
           self.initial_budget = initial_budget
        else:
           self.initial_budget = self.default_initial_budget
           
        self.r = self.initial_budget
        self.terminated = False
        self.truncated = False
        self.interrupted = False
        self.ruined = False
        
        observation = self._get_obs()

        info = self._get_info()
        
        #return observation, self.initial_budget, info
        return observation, info
        

    #--------------------------------------------------------------    
    def step(self, action, interval=1):
        """
         Like Gym.Env:
            
         Run one timestep of the environment's dynamics using the agent actions.

         When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
         reset this environment's state for the next episode.


         Args:
             action (ActType): an action provided by the agent to update the environment state.
             interval: the elapsed time for this step, generally 1 for discrete time problems.

         Returns:
             observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                 An example is a numpy array containing the positions and velocities of the pole in CartPole.
             reward (SupportsFloat): The reward as a result of taking the action.
             terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                 which can be positive or negative. An example is reaching the goal state or moving into the lava from
                 the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
             truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                 Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                 Can be used to end the episode prematurely before a terminal state is reached.
                 If true, the user needs to call :meth:`reset`.
             info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                 This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                 hidden from observations, or individual reward terms that are combined to produce the total reward.
                 In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                 however this is deprecated in favour of returning terminated and truncated variables.
        """
    
        if not self.ready:
            self.reset()

        self.t += interval


        #action effects
        # ...
        # self.s = ...
        # self.r = ...

        observation = self._get_obs()

        info = self._get_info()

        #if self.render_mode is not None:
        #    if not self.interrupted:
        #       self._render_frame()

        return observation, self.r, self.terminated, self.truncated, info

        
    #--------------------------------------------------------------    
    def get_state_tpl(self, s=None):
        if s is None:
           s = self.s
        return ensure_tuple(s)
         
    #--------------------------------------------------------------
    def get_state_idx(self, s=None):
        s = self.get_state_tpl(s)
        #return np.sum(np.multiply(s, self.observation_factors))
        return np.ravel_multi_index(s, self.observation_shape)

    #--------------------------------------------------------------    
    def get_reward_matrix(self):
       return None
       
    #--------------------------------------------------------------    
    def get_transition_matrix(self):
       return None


    #--------------------------------------------------------------    
    def _get_info(self):
        """
        method for the auxiliary information (a dict) that is returned by step and reset
        """
        return {"time-step": self.t}     

    #--------------------------------------------------------------    
    def _get_obs(self):
        """
        method for converting state to observation
        """
        return self.s


###################################################################

class EnvWrapper(gymnasium.Wrapper):

    #--------------------------------------------------------------    
    def _set_attribute(self, key, value, force=False):
        try:
           v = self.get_wrapper_attr(key) 
           if v is None or force:
              self.key = value
        except AttributeError:
           self.key = value

    #--------------------------------------------------------------    
    def _get_attribute(self, key):
        try:
           v = self.get_wrapper_attr(key) 
           return v
        except AttributeError:
           return None
      
    #--------------------------------------------------------------    
    def __init__(self, env, name="Environment"):
        
        super().__init__(env)
       
        self.env = env
        
        self._set_attribute('name', name)
           
        #flag : ready to execute
        self._set_attribute('ready', False)

        #time
        self._set_attribute('t', 0)

        #user interruption
        self._set_attribute('interrupted', False)

        #actions
        self.action_space, self.action_shape, self.action_ndim, self.action_comb = pyrl_space(self.action_space)
         
        #states
        self.state_space = self._get_attribute('state_space')
        if self.state_space is not None:
           self.state_space, self.state_shape, self.state_ndim, self.state_comb = pyrl_space(self.state_space)

        #observations (what the agent perceives from the environment state)
        self.observation_space = self._get_attribute('observation_space')
        if self.observation_space is not None:
            self.observation_space, self.observation_shape, self.observation_ndim, self.observation_comb = pyrl_space(self.observation_space)

        #states from observations
        if self.state_space is None:
            self.state_space, self.state_shape, self.state_ndim, self.state_comb = self.observation_space, self.observation_shape, self.observation_ndim, self.observation_comb

        #observations from states
        if self.observation_space is None:
            self.observation_space, self.observation_shape, self.observation_ndim, self.observation_comb = self.state_space, self.state_shape, self.state_ndim, self.state_comb
         
        self.initial_state = self._get_attribute('initial_state')
        if self.initial_state is not None:
            self.initial_state = ensure_tuple(self.initial_state)
            

    #--------------------------------------------------------------    
    @property
    def state_iterator(self):
       return product( *map(range, self.state_shape) )

    @property
    def action_iterator(self):
       return product( *map(range, self.action_shape) )

    @property
    def observation_iterator(self):
       return product( *map(range, self.observation_shape) )

    #--------------------------------------------------------------    
    def reset(self, *, seed:int=None, initial_state=None, options:dict=None) -> tuple:
        
        self.env.t = 0
        self.env.initial_state = initial_state
        self.env.ready = True
        
        return self.env.reset(seed=seed, options=options)
        
        
    #--------------------------------------------------------------    
    def step(self, action):
    
        if not self.env.ready:
            self.reset()

        self.env.t += 1

        return self.env.step(action)
     
    #--------------------------------------------------------------    
    def show(self):
       pass

    #--------------------------------------------------------------    
    def close(self):
       if self.env is not None:
          #super().close()
          self.env.close()
          self.env = None


