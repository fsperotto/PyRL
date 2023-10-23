from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
from gymnasium.spaces import Space

from pyrl import Agent

from mdptoolbox.mdp import PolicyIterationModified as PIM
from mdptoolbox.mdp import PolicyIteration as PI
from mdptoolbox.mdp import ValueIteration as VI
from mdptoolbox.mdp import ValueIterationGS as VIGS
from mdptoolbox.mdp import RelativeValueIteration as RVI


class PolicyIteration(PI, Agent):
    """The Policy Iteration wrapper to Py-MDP-ToolBox"""

    def __init__(self, 
                 transitions, reward, 
                 observation_space: Space, action_space: Space,
                 default_action=None, initial_budget:float=None,
                 discount:float=0.9,
                 max_error:float=0.01, max_iter:int=10, 
                 skip_check:bool=False):
        
        PI.__init__(self, transitions, reward, discount, epsilon=max_error, max_iter=max_iter, skip_check=skip_check)

        Agent.__init__(self, observation_space=observation_space, 
                          action_space=action_space, 
                          initial_budget=initial_budget,
                          default_action=default_action)
        
        #calculate policy
        self.run()
        


    def choose_action(self):
        self.a = self.policy(self.get_state_tpl())
        return self.a

