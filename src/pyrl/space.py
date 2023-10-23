#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:49:46 2023

@author: fperotto
"""

__version__ = "0.0.1"
__author__ = "Filipo S. Perotto, Aymane Ouahbi, Melvine Nargeot"
__license__ = "MIT"
__status__ = "Development"

################

from typing import Iterable, Union  # Callable, List
import numpy as np
import math
import gymnasium 
import gym 
from gymnasium.spaces.utils import flatdim   # flatten_space

from gymnasium.spaces import Space

###################################################################

def ensure_tuple(v):
    return tuple(v) if isinstance(v, Iterable) else (v,)   

###################################################################

class OneSpace(Space):
   
   def __init__(self, shape, dtype, seed):

      super().__init__(shape, dtype, seed)
      
      
###################################################################

def pyrl_space(space_or_dimensions:Union[int, Iterable[int], gym.spaces.Space, gymnasium.spaces.Space]) -> (gymnasium.spaces.Space, Union[int, None], Union[int, None]):
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
    num_vars = None
    num_comb = None
    shape=None
    idx_factors=None
    
    if isinstance(space_or_dimensions, int):
        space = gymnasium.spaces.Discrete(space_or_dimensions)
        shape = (space_or_dimensions,)   # (space.n,)
        num_vars = 1
        num_comb = space_or_dimensions   # space.n
        
    elif isinstance(space_or_dimensions, Iterable):
        space = gymnasium.spaces.MultiDiscrete(space_or_dimensions)
        shape = tuple(space_or_dimensions)
        num_vars = len(space_or_dimensions)
        num_comb = np.prod(space_or_dimensions)
        #num_comb = math.prod(space_or_dimensions)
        
    elif isinstance(space_or_dimensions, gymnasium.spaces.Discrete) or isinstance(space_or_dimensions, gym.spaces.Discrete):
        space = space_or_dimensions
        shape = (space.n,)
        num_vars = 1
        num_comb = space.n
        
    elif isinstance(space_or_dimensions, gymnasium.spaces.MultiDiscrete) or isinstance(space_or_dimensions, gym.spaces.MultiDiscrete):
        space = space_or_dimensions
        shape = tuple(space.nvec)
        num_vars = space.nvec.size
        num_comb = np.prod(space.nvec)

    elif isinstance(space_or_dimensions, gymnasium.spaces.MultiBinary) or isinstance(space_or_dimensions, gym.spaces.MultiBinary):
        space = space_or_dimensions
        shape = space.shape
        num_vars = len(space.shape)
        num_comb = math.prod(space.shape)
        
    elif isinstance(space_or_dimensions, gymnasium.spaces.Box) or isinstance(space_or_dimensions, gym.spaces.Box):
        #space = flatten_space(space)
        space = space_or_dimensions
        shape = tuple(np.zeros(space.shape, dtype=int)) 
        num_vars = flatdim(space)
        num_comb = float('inf')

    elif isinstance(space_or_dimensions, gymnasium.spaces.Text) or isinstance(space_or_dimensions, gym.spaces.Text):
        #space = flatten_space(space)
        space = space_or_dimensions
        num_vars = 1
        num_comb = math.sum([math.comb(len(space.characters), c) for c in range(space.min_length, space.max_length)])
        shape = (num_comb,) 
      
    f = num_comb
    idx_factors = [(f:=f//n) for n in shape]
      
    return space, shape, num_vars, num_comb, idx_factors
