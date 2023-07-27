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

import sys
import numpy as np
#from abc import ABC, abstractmethod  #ABC is abstract base class
#from collections.abc import Iterable
from typing import Iterable, Callable, TypeVar, Generic, Tuple, List, Union

from itertools import product

import gymnasium as gym
from gymnasium.spaces import Space, Discrete, MultiDiscrete, Box
from gymnasium.spaces.utils import flatdim, flatten_space

import pygame as pg


###################################################################

def pyrl_space(space_or_dimensions:Union[int, Iterable[int], Space]) -> (Space, Union[int, None], Union[int, None]):
    """
    Function to convert the representation of a state_space, an observation_space, or an action_space

        Returns:
            space : MultiDiscrete (gymnasium space)
            shape : tuple
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
    #elif isinstance(space_or_dimensions, Box):
    #    space = flatten_space(space)
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
    elif isinstance(space, Box):
        shape = tuple(np.zeros(space.shape, dtype=int)) 
        num_vars = flatdim(space)
        num_comb = float('inf')
    
    return space, shape, num_vars, num_comb
            
###################################################################

def ensure_tuple(v):
    return tuple(v) if isinstance(v, Iterable) else (v,)   
   
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
            s : tuple
                current state, from the last observation.
            r : float
                last received reward.
    """

    #--------------------------------------------------------------    
    def __init__(self, observation_space, action_space,
                 default_action=None, initial_budget=None,
                 name=None):
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

        if name is not None:
           self.name = name
        else:
           if type(self) == Agent:
              if self.default_action is None:
                 self.name = "Random Agent"
              else:
                 self.name = "Constant Agent"
           else:
                 self.name = "Custom Agent"
              

        self.initial_budget = initial_budget

        #time
        self.t = None

        #current state
        self.s = None
        
        #next chosen action
        self.a = None

        #last received reward
        self.r = None
        

        self.learning = None

        self.terminated = None
        self.truncated = None
        self.ruined = None
        
        self.should_reset = True
        

    #--------------------------------------------------------------    
    @property
    def time_step(self):
      return self.t

    @property
    def current_round(self):
      return self.t

    @property
    def state(self):
      return self.s
   
    @property
    def current_state(self):
      return self.s

    @property
    def last_reward(self):
      return self.r
   
    @property
    def reward(self):
      return self.r

    @property
    def action(self):
      return self.a
   
    @property
    def chosen_action(self):
      return self.a

    @property
    def budget(self):
      return self.b
   
    @property
    def current_budget(self):
      return self.b

    #--------------------------------------------------------------    

    @property
    def action_iterator(self):
       return product( *map(range, self.action_shape) )

    @property
    def observation_iterator(self):
       return product( *map(range, self.observation_shape) )

    #--------------------------------------------------------------    
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
        
        self.r = None
        self.a = None
            
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


    #--------------------------------------------------------------    
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



    #--------------------------------------------------------------    
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

        if self.budget is not None:
            self.budget = self.budget + r

    #--------------------------------------------------------------    
    def learn(self):
        pass


    #--------------------------------------------------------------    
    def observe_and_learn(self, s, r, terminated, truncated):
       
        self.observe(s, r, terminated, truncated)
        self.learn()


    #--------------------------------------------------------------    
    def get_state(self, s=None):
        if s is None:
           s = self.s
        return ensure_tuple(s)

         
    #--------------------------------------------------------------    
    def get_action(self, a=None):
        if a is None:
           a = self.a
        return ensure_tuple(a)


    #--------------------------------------------------------------    
    def get_state_action(self):
        return self.get_state() + self.get_action()
         

###################################################################

class EnvWrapper(gym.Wrapper):
    
    #--------------------------------------------------------------    
    def __init__(self, env, name="Environment"):
        
        super().__init__(env)
       
        self.env = env
        
        #self.env.name = name
        if not hasattr(self, 'name') or self.name is None:
           self.name = name
       
        #flag : ready to execute
        #self.env.ready = False
        if not hasattr(self, 'ready') or self.ready is None:
           self.ready = False
        
        #time
        #self.env.t = None
        if not hasattr(self, 't') or self.t is None:
           self.t = 0

        #self.env.interrupted = False
        if not hasattr(self, 'interrupted') or self.interrupted is None:
           self.interrupted = False

        #actions
        self.action_space, self.action_shape, self.num_act_var, self.num_act_comb = pyrl_space(self.action_space)
         
        #states
        if hasattr(self, 'state_space') and self.state_space is not None:
           self.state_space, self.state_shape, self.num_state_var, self.num_state_comb = pyrl_space(self.state_space)

        #observations (what the agent perceives from the environment state)
        if hasattr(self, 'observation_space') and self.observation_space is not None:
            self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = pyrl_space(self.observation_space)

        #states
        if not hasattr(self, 'state_space') or self.state_space is None:
            self.state_space, self.state_shape, self.num_state_var, self.num_state_comb = self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb

        #observations (what the agent perceives from the environment state)
        if not hasattr(self, 'observation_space') or self.observation_space is None:
            self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = self.state_space, self.state_shape, self.num_state_var, self.num_state_comb
         
        if not hasattr(self, 'initial_state') or self.initial_state is None:
            self.initial_state = None
        else:
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


###################################################################

    # metadata = {"render_modes": []}
    # render_mode = None
    # reward_range = (-float("inf"), float("inf"))
    # spec = None
    #
    # action_space
    # observation_space
    #
    # _np_random = None
    #
    # def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    #
    # def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
    #     """Resets the environment to an initial internal state, returning an initial observation and info.

    #     This method generates a new starting state often with some randomness to ensure that the agent explores the
    #     state space and learns a generalised policy about the environment. This randomness can be controlled
    #     with the ``seed`` parameter otherwise if the environment already has a random number generator and
    #     :meth:`reset` is called with ``seed=None``, the RNG is not reset.

    #     Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

    #     For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
    #     the seeding correctly.

    #     .. versionchanged:: v0.25

    #         The ``return_info`` parameter was removed and now info is expected to be returned.

    #     Args:
    #         seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
    #             If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
    #             a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
    #             However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
    #             If you pass an integer, the PRNG will be reset even if it already exists.
    #             Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
    #             Please refer to the minimal example above to see this paradigm in action.
    #         options (optional dict): Additional information to specify how the environment is reset (optional,
    #             depending on the specific environment)

    #     Returns:
    #         observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
    #             (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
    #         info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
    #             the ``info`` returned by :meth:`step`.
    #     """
    #     # Initialize the RNG if the seed is manually passed
    #     if seed is not None:
    #         self._np_random, seed = seeding.np_random(seed)

    # def render(self) -> RenderFrame | list[RenderFrame] | None:
    #     """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

    #     The environment's :attr:`metadata` render modes (`env.metadata["render_modes"]`) should contain the possible
    #     ways to implement the render modes. In addition, list versions for most render modes is achieved through
    #     `gymnasium.make` which automatically applies a wrapper to collect rendered frames.

    #     Note:
    #         As the :attr:`render_mode` is known during ``__init__``, the objects used to render the environment state
    #         should be initialised in ``__init__``.

    #     By convention, if the :attr:`render_mode` is:

    #     - None (default): no render is computed.
    #     - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption.
    #       This rendering should occur during :meth:`step` and :meth:`render` doesn't need to be called. Returns ``None``.
    #     - "rgb_array": Return a single frame representing the current state of the environment.
    #       A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing RGB values for an x-by-y pixel image.
    #     - "ansi": Return a strings (``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
    #       for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
    #     - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the
    #       wrapper, :py:class:`gymnasium.wrappers.RenderCollection` that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
    #       The frames collected are popped after :meth:`render` is called or :meth:`reset`.

    #     Note:
    #         Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes the list of supported modes.

    #     .. versionchanged:: 0.25.0

    #         The render function was changed to no longer accept parameters, rather these parameters should be specified
    #         in the environment initialised, i.e., ``gymnasium.make("CartPole-v1", render_mode="human")``
    #     """
    #     raise NotImplementedError

    # def close(self):
    #     """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

    #     This is critical for closing rendering windows, database or HTTP connections.
    #     Calling ``close`` on an already closed environment has no effect and won't raise an error.
    #     """
    #     pass

    # @property
    # def unwrapped(self) -> Env[ObsType, ActType]:
    #     """Returns the base non-wrapped environment.

    #     Returns:
    #         Env: The base non-wrapped :class:`gymnasium.Env` instance
    #     """
    #     return self

    # @property
    # def np_random(self) -> np.random.Generator:
    #     """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

    #     Returns:
    #         Instances of `np.random.Generator`
    #     """
    #     if self._np_random is None:
    #         self._np_random, _ = seeding.np_random()
    #     return self._np_random

    # @np_random.setter
    # def np_random(self, value: np.random.Generator):
    #     self._np_random = value

    # def __str__(self):
    #     """Returns a string of the environment with :attr:`spec` id's if :attr:`spec.

    #     Returns:
    #         A string identifying the environment
    #     """
    #     if self.spec is None:
    #         return f"<{type(self).__name__} instance>"
    #     else:
    #         return f"<{type(self).__name__}<{self.spec.id}>>"

    # def __enter__(self):
    #     """Support with-statement for the environment."""
    #     return self

    # def __exit__(self, *args: Any):
    #     """Support with-statement for the environment and closes the environment."""
    #     self.close()
    #     # propagate exception
    #     return False

    # def get_wrapper_attr(self, name: str) -> Any:
    #     """Gets the attribute `name` from the environment."""
    #     return getattr(self, name)

class Env(gym.Env):

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
                 name="Environment"):
        
        self.name = name
       
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
        else:
            self.initial_state = ensure_tuple(initial_state)
        #elif isinstance(initial_state, int):
        #    self.initial_state = np.array([initial_state])
        #else:
        #    self.initial_state = np.array(initial_state)

            
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
    def get_state(self, s=None):
        if s is None:
           s = self.s
        return ensure_tuple(s)
         
         
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

class GUI():

    #--------------------------------------------------------------    
    def __init__(self, sim):

        self.sim = sim       #reference to the simulation
        
    #--------------------------------------------------------------    
    def reset(self):
        pass

    #--------------------------------------------------------------    
    def launch(self):
        pass

    #--------------------------------------------------------------    
    def refresh(self):
        pass
                
    #--------------------------------------------------------------    
    def close(self):
        pass


###################################################################

class PyGameGUI(GUI):

    #--------------------------------------------------------------    
    def __init__(self, sim,
                 height=800, width=800,
                 fps=40,
                 batch_run=10000,
                 on_close_listeners:Iterable[Callable]=[],
                 close_on_finish=True):

        super().__init__(sim) 
        
        self.height = height
        self.width = width
        
        self.fps = fps
        self.refresh_interval_ms = max(10, 1000 // self.fps)
        
        self.batch_run = batch_run
        
        self.close_on_finish = close_on_finish
        
        self.on_close_listeners = on_close_listeners
        
        self._is_closing = False
        
        pg.init()

        self.CLOCKEVENT = pg.USEREVENT+1
        #self.clock = pg.time.Clock()
        
        self.window = None
        
        #self.sim.add_listener('round_finished', self.on_clock)
        
    #--------------------------------------------------------------    
    def reset(self):
       
        self._is_closing = False

    #--------------------------------------------------------------    
    def set_timer_state(self, state:bool):
       
        if state == True:
           pg.time.set_timer(self.CLOCKEVENT, self.refresh_interval_ms)
        else:
           pg.time.set_timer(self.CLOCKEVENT, 0)

    #--------------------------------------------------------------    
    def launch(self, give_first_step=True, start_running=True):

        pg.display.init()
        self.window = pg.display.set_mode( (self.width, self.height) )
        #pg.display.set_caption('Exp')        
        #self.window.set_caption('Exp')
        
        if give_first_step:
           self.sim.step()
           self.refresh()
        
        if start_running:
           self.set_timer_state(True)
        
        #RUNNING
        try:

           while not self._is_closing:
              
              event = pg.event.wait()
              
              self.process_event(event)
              
              if self.close_on_finish and self.sim.finished:
                 self.close()

        except KeyboardInterrupt:
           self.close()
           print("KeyboardInterrupt: simulation interrupted by the user.")

        except:
           self.close()
           raise

     
    #--------------------------------------------------------------    
    def refresh(self):
        
        #refresh
        pg.display.update()
                
    #--------------------------------------------------------------    
    def close(self):
       
        self._is_closing = True
 
        if self.window is not None:

            #CLOSING
            for callback in self.on_close_listeners:
               callback(self)
               
            pg.display.quit()
            pg.quit()

        if self.sim.env is not None:
           self.sim.env.close()

    #--------------------------------------------------------------    
    def process_event(self, event):
       
         if (event.type == pg.QUIT):
             self.close()
   
         elif event.type == pg.KEYDOWN:
             self.on_keydown(event.key)
             
         elif event.type == self.CLOCKEVENT:
             self.sim.step()
             self.refresh()
      
    #--------------------------------------------------------------    
    def on_keydown(self, key):
       
       #ESC = exit
       if key == pg.K_ESCAPE:
          self.close()
          
       #P = pause
       elif key == pg.K_p:
          self.set_timer_state(False)
          
       #R = run
       elif key == pg.K_r:
          self.set_timer_state(True)
          
       #S = step
       elif key == pg.K_s:
          self.sim.step()
          self.refresh()
            
       #B = batch run
       elif key == pg.K_b:
          self.sim.run(self.batch_run)
          self.refresh()

       #E = episode run
       elif key == pg.K_e:
          self.sim.run('episode')
          self.refresh()

       #Q = simulation run
       elif key == pg.K_q:
          self.sim.run('simulation')
          self.refresh()

       #Z = repetition run
       elif key == pg.K_z:
          self.sim.run('repetition')
          self.refresh()
          
    #--------------------------------------------------------------    
    #def on_clock(self, *args, **kwargs):
    #   self.refresh()
    #   self.clock.tick(self.fps)

    #--------------------------------------------------------------    
    #def step(self):
    #   self.sim.next_step()

    #--------------------------------------------------------------    
    # def process_events(self):

    #     keys = [] 

    #     events = pg.event.get()
        
    #     for event in events:

    #         if event.type == pg.QUIT:
    #             self._is_closing = True
      
    #         if event.type == pg.KEYDOWN:
    #             keys = keys + [event.key]
                
    #     if len(keys) > 0:
    #        self.on_keydown(keys)

    #--------------------------------------------------------------    
    # def refresh(self):

    #     self.process_events()
       
    #     self.render()
                   
            
            
            
            
###################################################################

class Renderer():

    #--------------------------------------------------------------    
    def __init__(self, env=None, agent=None):

        self.env = env       #reference to the environment
        self.agent = agent       #reference to the environment
        
        self.ready = (self.env is not None)

    #--------------------------------------------------------------    
    def reset(self, env=None, agent=None):

        if env is not None:
           self.env = env       #reference to the environment
           
        if agent is not None:
           self.agent = agent       #reference to the environment
        
        self.ready = (self.env is not None)

    #--------------------------------------------------------------    
    def render(self):
        pass
     
      
    #--------------------------------------------------------------    
    def refresh(self):
        pass
                
    #--------------------------------------------------------------    
    def close(self):
        pass


###################################################################

class PyGameRenderer(Renderer):

    #--------------------------------------------------------------    
    def __init__(self, env=None, agent=None, 
                 height=800, width=800,
                 on_close_listeners:Callable=None):

        super().__init__(env, agent) 
        
        self.height = height
        self.width = width
        
        self.on_close_listeners = on_close_listeners
        
        self._is_closing = False
        
        pg.init()
        pg.display.init()

        self.window = pg.display.set_mode( (self.width, self.height) )
        

    #--------------------------------------------------------------    
    def reset(self):
       
        self._is_closing = False


    #--------------------------------------------------------------    
    def render(self):

        #refresh
        pg.display.update()

    #--------------------------------------------------------------    
    def process_events(self):

        keys = [] 

        events = pg.event.get()
        
        for event in events:

            if event.type == pg.QUIT:
                self._is_closing = True
      
            if event.type == pg.KEYDOWN:
                keys += event.key
                
        if len(keys) > 0:
           self.on_keydown(keys)

    #--------------------------------------------------------------    
    def on_keydown(self, keys):
       pass
        
      
    #--------------------------------------------------------------    
    def refresh(self):

        if self._is_closing:
            
            if self.on_close_listeners is not None:
               self.on_close_listeners()

            self.close()

        else:

            self.process_events()

            self.render()
                   
            
    #--------------------------------------------------------------    
    def close(self) -> None:
       
        if self.window is not None:
            pg.display.quit()
            pg.quit()

         
    #--------------------------------------------------------------    
    def add_close_listener(self, listener:Callable):
        self.on_close_listeners.append(listener)
        
        
        
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
        
        self.episode_horizon = episode_horizon
        self.num_episodes = num_episodes
        self.num_repetitions = num_repetitions
        
        self.close_on_finish = close_on_finish
        
        self.reset()
        # self.metrics = {"time": 0, "exploration": []}

        self.metrics = dict(
            time = 0,
            exploration = np.zeros((self.envs[0].observation_space.n, self.envs[0].action_space.n)),
            budget = np.zeros((self.episode_horizon,), dtype=int)
        )

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
                
               observation, info = self.env.reset()
               is_first_episode = (self.ep==0)
               self.agent.reset(observation, reset_knowledge=is_first_episode)
   
               #episode started event callback
               self._evoke_listeners('episode_started')

            #episode is not finished, next round
            else:               
               
               self.t += 1
          
               #round started event callback
               self._evoke_listeners('round_started')
  
               action = self.agent.choose_action()  # agent policy that uses the observation and info
               observation, reward, terminated, truncated, info = self.env.step(action)
                      
               self.agent.observe_and_learn(observation, reward, terminated, truncated)

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
             #sys.exit()
   
          except:
             self.close()
             raise
             
          if self.close_on_finish:
             self.close()


    #--------------------------------------------------------------    
    def close(self):
       
       if self.env is not None:
          self.env.close()