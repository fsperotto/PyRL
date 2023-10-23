# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:57:14 2023

@author: fperotto
"""

do_install = True

if do_install:

   #INSTALLING ROMS  -  LAST PRACTICE
   import AutoROM
   #AutoROM.main(True, None, False)
   AutoROM.cli()


   #INSTALLING ROMS  -  OLD PRACTICE
   from ale_py import ALEInterface
   ale = ALEInterface()

   #!ale-import-roms D:\fperotto\SourceCode\PyRL\data\roms\atari2600\ROMS


#PRINT EXISTING ROMS
import ale_py.roms as roms
print(dir(roms))


#TEST WITH GYM

import gym
x = gym.make('SpaceInvaders-v4')
print(x)


#TEST WITH GYMNASIUM

import gymnasium
x = gymnasium.make('SpaceInvaders-v4')


x = gymnasium.make('ALE/RoadRunner-v5')


import gym
from gym import envs

all_envs = envs.registry
env_ids = [env_spec.id for env_spec in all_envs.values()]
print(env_ids)

black_list = ['Defender']

for skip in black_list:
    env_ids = [env_id for env_id in env_ids if not skip in env_id]

env_ids = sorted(env_ids)

print(len(env_ids), 'environments')