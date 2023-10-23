#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Agent Class.
"""

__version__ = "0.0.1"
__author__ = "Filipo S. Perotto, Aymane Ouahbi, Melvine Nargeot"
__license__ = "MIT"
__status__ = "Development"

################

from typing import Iterable, Union
from itertools import product

from gymnasium.spaces import Space

from pyrl.space import pyrl_space, ensure_tuple

from pyrl.env import Env

import unittest

import numpy as np

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
    def __init__(self, env,
                 action_space : Union[None, int, Iterable[int], Space] = None,
                 default_action = None,
                 name:Union[None, str] = None,
                 remember_prev_s=False, remember_prev_a=False,
                 store_N_sa=False, store_N_saz=False, store_N_z=False, store_N_a=False
                ):
        """
        Agent Constructor.
        The dimensions concerning observable states and actions must be informed.
        """

        self.env = env

        self.observation_space = self.env.observation_space
        self.observation_shape = self.env.observation_shape
        self.observation_ndim = self.env.observation_ndim
        self.observation_comb = self.env.observation_comb
        self.observation_factors = self.env.observation_factors

        self.action_space = self.env.action_space
        self.action_shape = self.env.action_shape
        self.action_ndim = self.env.action_ndim
        self.action_comb = self.env.action_comb
        self.action_factors = self.env.action_factors

        #self.default_action = ensure_tuple(default_action)
        self.default_action = default_action

        #name (label)
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

        #current time step, zero on reset(), incremented on observe()
        self.t = None

        #current state
        self.s = None

        #next chosen action
        self.a = None

        #last received reward
        self.r = None

        #current budget, if budgeted
        self.b = None

        #if the agent is learning something set to 'on-policy' or 'off-policy', otherwise 'disabled'
        self.learning_mode = 'disabled'

        #flag to say that the episode finished correctly
        self.terminated = None

        #flag to say that the episode was interrupted
        self.truncated = None

        #flag to say if the agent is ruined (depleted budget)
        self.ruined = None

        #flag to say that the agent must be reseted
        self.ready = False
        
        #previous transition memory
        self.remember_prev_s=remember_prev_s
        if remember_prev_s:
           self.prev_s = None
        
        self.remember_prev_a=remember_prev_a
        if remember_prev_a:
           self.prev_a = None
        
        #counter of observed transitions
        self.store_N_sa=store_N_sa
        if store_N_sa:
           self.N_sa = None
           
        self.store_N_saz=store_N_saz
        if store_N_saz:
           self.N_saz = None

        self.store_N_z=store_N_z
        if store_N_z:
           self.N_z = None
        
        self.store_N_a=store_N_a
        if store_N_a:
           self.N_a = None
        
    #--------------------------------------------------------------

    @property
    def action_iterator(self):
       return product( *map(range, self.action_shape) )

    @property
    def observation_iterator(self):
       return product( *map(range, self.observation_shape) )


    #--------------------------------------------------------------
    def reset(self, initial_observation, *,
              reset_knowledge=True, learning_mode=None,
              reset_budget=True, initial_budget=None):
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
        #self.s = ensure_tuple(initial_observation)
        self.s = initial_observation

        self.r = None

        #current budget
        if reset_budget:
           self.b = initial_budget

        self.terminated = False
        self.truncated = False

        if self.b is None or self.b > 0:
            self.ruined = False
        else:
            self.ruined = True

        if learning_mode is not None:
           self.learning_mode = learning_mode    # 'on-policy', 'off-policy', 'disabled'

        if self.remember_prev_s:
           self.prev_s = None
        
        if self.remember_prev_a:
           self.prev_a = None

        if self.store_N_z:
           self.N_z = np.zeros(self.observation_shape, dtype=int)

        if self.store_N_sa:
           self.N_sa = np.zeros(self.observation_shape + self.action_shape, dtype=int)

        if self.store_N_saz:
           self.N_saz = np.zeros(self.observation_shape + self.action_shape + self.observation_shape, dtype=int)

        if self.store_N_saz:
           self.N_a = np.zeros(self.action_shape, dtype=int)

        self.a = self._choose()

        self.ready = True
        
        #set the good step method, after reset
        #self.step = self._step

        return self.a


    #--------------------------------------------------------------
    def _choose(self):
        """
        Choose an action to execute, and return it.

            Returns:
                action : list representing the joint action chosen by the controller

        """
        #if the agent was not reseted after initialization, then reset
        #if self.should_reset:
        #    raise ValueError("ERROR: Agent properties should be initilized. Maybe you forgot to call the reset method ?")

        #choose the default action, if defined
        if self.default_action is None:
            #self.a = ensure_tuple(self.action_space.sample())
            return self.action_space.sample()
        #choose uniformly random action otherwise
        else:
            return self.default_action
        #return the chosen action
        #return self.a


#    act = choose
#    choose_action = choose

    #--------------------------------------------------------------
    def step(self, s=None, r:float=0.0, terminated:bool=False, truncated:bool=False, info=None, **kwargs):
       
       if not self.ready:
          raise ValueError("ERROR: Agent properties should be initilized. Maybe you forgot to call the reset method ?")       
       
       #PERCEPTION AND MEMORY
       self._observe(s, r, terminated, truncated)
       
       #OFF-POLICY LEARNING
       if self.learning_mode == 'off-policy':
          self._learn()

       #ACTION DECISION
       self.a = self._choose()
       #self.a_idx = ensure_tuple(self.a)

       #ON-POLICY LEARNING
       if self.learning_mode == 'on-policy':
          self._learn()

       return self.a

    #--------------------------------------------------------------

    #def _observe(self, s=None, r:float=0.0, terminated:bool=False, truncated:bool=False, info=None, **kwargs) -> None :
    def _observe(self, s=None, r:float=0.0, terminated:bool=False, truncated:bool=False) -> None :
       """
       Parameters
       ----------
       s : tuple
          joint observation (or state).
       r : float
          received reward.
       terminated : bool
          episode ended normally.
       truncated : bool
          episode ended abnormally.

       Raises
       ------
       ValueError
          Error if reset() not called before.

       Returns
       -------
          action : list representing the joint action chosen by the controller

       """

       self.t = self.t + 1

       #count N(s,a), the number of observations of this pair
       if self.store_N_sa:
          self.N_sa[self.get_state_action_tpl()] += 1

       if self.store_N_a:
          self.N_a[self.get_action_tpl()] += 1


       if self.remember_prev_s:
          self.prev_s = self.s

       if self.remember_prev_a:
          self.prev_a = self.a

       """Memorize the observed state and received reward."""
       self.s = s
       #self.s_idx = ensure_tuple(self.s)
       self.r = r
       self.terminated = terminated
       self.truncated = truncated

       if self.store_N_z:
          self.N_z[self.get_state_tpl()] += 1

       #if self.store_N_saz:
       #   self.N_a[self.get_action_tpl()] += 1

       if self.b is not None:
          self.b = self.b + r



    #--------------------------------------------------------------
    def _learn(self):
        pass

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

    @property
    def s_idx(self):
      return self.get_state_tpl()

    @property
    def a_idx(self):
      return self.get_action_tpl()

    @property
    def sa_idx(self):
      return self.get_state_tpl() + self.get_action_tpl()

    #--------------------------------------------------------------
    def get_state_tpl(self, s=None):
        if s is None:
           s = self.s
        return ensure_tuple(s)

    #--------------------------------------------------------------
    #single integer index, from combined dimensions
    def get_state_idx(self, s=None):
        return np.ravel_multi_index(self.get_state_tpl(s), self.observation_shape)

    #--------------------------------------------------------------
    def state_idx_to_tpl(self, s_idx:int):
        return np.unravel_index(s_idx, self.observation_shape)

    #--------------------------------------------------------------
    def get_action_tpl(self, a=None):
        if a is None:
           a = self.a
        return ensure_tuple(a)

    #--------------------------------------------------------------
    def get_action_idx(self, a=None):
        return np.ravel_multi_index(self.get_action_tpl(a), self.action_shape)

    #--------------------------------------------------------------
    def action_idx_to_tpl(self, a_idx:int):
        return np.unravel_index(a_idx, self.action_shape)

    #--------------------------------------------------------------
    def get_state_action_tpl(self):
        return self.get_state_tpl() + self.get_action_tpl()


###################################################################


#unit test
if __name__ == "__main__":

   import math

   from gymnasium.spaces import Discrete, MultiDiscrete #, MultiBinary, Box

   class TestMethods(unittest.TestCase):

      def test_module_simple_spaces(self):
         # unit test
         m = 10
         observation_space = m
         observation_ndim = 1
         observation_shape = (m,)
         observation_comb = m
         n = 3
         action_space = n
         action_ndim = 1
         action_shape = (n,)
         action_comb = n
         env=Env(observation_space, action_space)
         a = Agent(env)
         assert a is not None, "Agent not created."
         assert a.action_ndim == action_ndim, "Wrong action ndim: " + str(a.action_ndim) + " != " + str(action_ndim)
         assert a.action_comb == action_comb, "Wrong action comb: " + str(a.action_comb) + " != " + str(action_comb)
         assert a.action_shape == action_shape, "Wrong action shape: " + str(a.action_shape) + " != " + str(action_shape)
         assert a.observation_ndim == observation_ndim, "Wrong action ndim: " + str(a.observation_ndim) + " != " + str(observation_ndim)
         assert a.observation_comb == observation_comb, "Wrong action comb: " + str(a.observation_comb) + " != " + str(observation_comb)
         assert a.observation_shape == observation_shape, "Wrong action shape: " + str(a.observation_shape) + " != " + str(observation_shape)

         act = a.reset(0, initial_budget=100)
         assert a.t == 0
         assert a.s == 0
         assert a.a in [v for v in range(n)]
         assert a.a == act
         assert a.b == 100

         act = a.step(1, -1, False, False)
         assert a.t == 1
         assert a.s == 1
         assert a.a in [v for v in range(n)]
         assert a.a == act
         assert a.b == 99



      def test_module_array_spaces(self):
         m = [10, 2, 3]
         observation_space = m
         observation_ndim = len(m)
         observation_shape = tuple(m)
         observation_comb = math.prod(m)
         n = [3, 4]
         action_space = n
         action_ndim = len(n)
         action_shape = tuple(n)
         action_comb = math.prod(n)
         env=Env(observation_space, action_space)
         a = Agent(env)
         assert a is not None, "Agent not created."
         assert a.action_ndim == action_ndim, "Wrong action ndim: " + str(a.action_ndim) + " != " + str(action_ndim)
         assert a.action_comb == action_comb, "Wrong action comb: " + str(a.action_comb) + " != " + str(action_comb)
         assert a.action_shape == action_shape, "Wrong action shape: " + str(a.action_shape) + " != " + str(action_shape)
         assert a.observation_ndim == observation_ndim, "Wrong action ndim: " + str(a.observation_ndim) + " != " + str(observation_ndim)
         assert a.observation_comb == observation_comb, "Wrong action comb: " + str(a.observation_comb) + " != " + str(observation_comb)
         assert a.observation_shape == observation_shape, "Wrong action shape: " + str(a.observation_shape) + " != " + str(observation_shape)

      def test_module_gymnasium_discrete_spaces(self):
         m = 10
         observation_space = Discrete(m)
         observation_ndim = 1
         observation_shape = (observation_space.n,)
         observation_comb = observation_space.n
         n = 3
         action_space = Discrete(n)
         action_ndim = 1
         action_shape = (action_space.n,)
         action_comb = action_space.n
         env=Env(observation_space, action_space)
         a = Agent(env)
         assert a is not None, "Agent not created."
         assert a.action_ndim == action_ndim, "Wrong action ndim: " + str(a.action_ndim) + " != " + str(action_ndim)
         assert a.action_comb == action_comb, "Wrong action comb: " + str(a.action_comb) + " != " + str(action_comb)
         assert a.action_shape == action_shape, "Wrong action shape: " + str(a.action_shape) + " != " + str(action_shape)
         assert a.observation_ndim == observation_ndim, "Wrong action ndim: " + str(a.observation_ndim) + " != " + str(observation_ndim)
         assert a.observation_comb == observation_comb, "Wrong action comb: " + str(a.observation_comb) + " != " + str(observation_comb)
         assert a.observation_shape == observation_shape, "Wrong action shape: " + str(a.observation_shape) + " != " + str(observation_shape)

      def test_module_gymnasium_multidiscrete_spaces(self):
         m = [10, 2, 3]
         observation_space = MultiDiscrete(m)
         observation_ndim = len(m)
         observation_shape = tuple(m)
         observation_comb = math.prod(m)
         n = [3, 4]
         action_space = MultiDiscrete(n)
         action_ndim = len(n)
         action_shape = tuple(n)
         action_comb = math.prod(n)
         env=Env(observation_space, action_space)
         a = Agent(env)
         assert a is not None, "Agent not created."
         assert a.action_ndim == action_ndim, "Wrong action ndim: " + str(a.action_ndim) + " != " + str(action_ndim)
         assert a.action_comb == action_comb, "Wrong action comb: " + str(a.action_comb) + " != " + str(action_comb)
         assert a.action_shape == action_shape, "Wrong action shape: " + str(a.action_shape) + " != " + str(action_shape)
         assert a.observation_ndim == observation_ndim, "Wrong action ndim: " + str(a.observation_ndim) + " != " + str(observation_ndim)
         assert a.observation_comb == observation_comb, "Wrong action comb: " + str(a.observation_comb) + " != " + str(observation_comb)
         assert a.observation_shape == observation_shape, "Wrong action shape: " + str(a.observation_shape) + " != " + str(observation_shape)
         env=Env(observation_space, action_space)
         a = Agent(env)
         assert a is not None, "Agent not created."
         assert a.action_ndim == action_ndim, "Wrong action ndim: " + str(a.action_ndim) + " != " + str(action_ndim)
         assert a.action_comb == action_comb, "Wrong action comb: " + str(a.action_comb) + " != " + str(action_comb)
         assert a.action_shape == action_shape, "Wrong action shape: " + str(a.action_shape) + " != " + str(action_shape)
         assert a.observation_ndim == observation_ndim, "Wrong action ndim: " + str(a.observation_ndim) + " != " + str(observation_ndim)
         assert a.observation_comb == observation_comb, "Wrong action comb: " + str(a.observation_comb) + " != " + str(observation_comb)
         assert a.observation_shape == observation_shape, "Wrong action shape: " + str(a.observation_shape) + " != " + str(observation_shape)

   unittest.main()
