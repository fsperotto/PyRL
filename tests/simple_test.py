# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 00:55:00 2023

@author: fperotto
"""

from pyrl import Env
from pyrl import Agent

##############################

env = Env(10, 2)

agent = Agent(observation_space=env.observation_space, action_space=env.action_space)

s, b, info = env.reset()
print('s_0 =', s)

a = agent.reset(s)
print('a_0 =', a)

for t in range(10):
   s, r, terminated, truncated, info = env.step(a)
   print('s =', s, 'r =', r)
   a = agent.step(s, r, terminated, truncated)
   print('a =', a)
   
##############################

agent = Agent(env)

s, b, info = env.reset()
print('s_0 =', s)

a = agent.reset(s)
print('a_0 =', a)

for t in range(10):
   s, r, terminated, truncated, info = env.step(a)
   print('s =', s, 'r =', r)
   a = agent.step(s, r, terminated, truncated)
   print('a =', a)

##############################

agent = Agent(env=env, default_action=1)

s, b, info = env.reset()
print('s_0 =', s)

a = agent.reset(s)
print('a_0 =', a)

for t in range(10):
   s, r, terminated, truncated, info = env.step(a)
   print('s =', s, 'r =', r)
   a = agent.step(s, r, terminated, truncated)
   print('a =', a)
