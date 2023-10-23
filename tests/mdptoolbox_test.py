# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:16:09 2023

@author: fperotto
"""

import numpy as np
import gymnasium as gym

from mdptoolbox.mdp import PolicyIteration

from pyrl import Sim, Agent, EnvWrapper #, System
from pyrl.environments.grid import GridEnv, GridEnvGUI


env = EnvWrapper(gym.make('FrozenLake-v1', render_mode="rgb_array"))

print(dir(env))

#############

#GRID ENVIRONMENT PARAMETERS
num_rows=1
num_cols=3

minor_r = 5.0
major_r = 20.0

reward_targets = {major_r : [(num_cols - 2, num_rows // 2)],
                  minor_r : [(3*(num_cols - 1) // 5, num_rows // 2), ((num_cols - 1) // 3, num_rows // 2)]}

env = GridEnv(num_rows=num_rows, num_cols=num_cols, 
              reward_mode="s'", reward_targets=reward_targets, default_reward=-1.0,
              render_mode="external")

#############

gamma = 0.9

P = env.get_transition_matrix()
print(P.shape)
#P = np.swapaxes(x,0,1)   # transform S,A,S' into A,S,S'
P = P.swapaxes(0, 1)
print(P.shape)

R = env.get_reward_matrix()

agent = PolicyIteration(P, R, gamma)

agent.run()

# existent déjà, avec des refs : pyRL, oneRL, uniRL (?)
# possibles : fedeRL, fedRL, supeRL, wideRL, airl, 

# we need wrapper classes : EnvOneRL, SpaceOneRL, AgentOneRL