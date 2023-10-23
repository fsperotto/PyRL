# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 21:59:29 2023

@author: fperotto
"""

import gymnasium as gym

from pyrl import Sim, Agent, EnvWrapper #, System

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.ppo import MlpPolicy

#################################################

def start_episode_callback(sim):
   print("START EPISODE", sim.ep+1, '/', sim.num_episodes)

def end_episode_callback(sim):
   print(sim.t, 'rounds')
   
#################################################

#env = EnvWrapper(gym.make('CartPole-v1', render_mode="rgb_array"))
env = EnvWrapper(gym.make('CartPole-v1', render_mode=None))

models = [PPO(MlpPolicy, env, verbose=1),
          A2C(MlpPolicy, env, verbose=1)]
          #DQN(MlpPolicy, env, verbose=1)]
for model in models:
    model.learn(total_timesteps=10000, progress_bar=True)

env = EnvWrapper(gym.make('CartPole-v1', render_mode="human"))

for model in models:

    s, b = env.reset()
    a, _s = model.predict(s, deterministic=True)

    for t in range(1000):
       s, r, terminated, truncated, info = env.step(a)
       if (terminated or truncated):
          s, b = env.reset()
       a, _s = model.predict(s, deterministic=True)
    
env.close()

#################################################

env = EnvWrapper(gym.make('ALE/SpaceInvaders-v5', render_mode=None))

models = [PPO(MlpPolicy, env, verbose=1),
          A2C(MlpPolicy, env, verbose=1)]
          #DQN(MlpPolicy, env, verbose=1)]
for model in models:
    model.learn(total_timesteps=10000, progress_bar=True)

env = EnvWrapper(gym.make('ALE/SpaceInvaders-v5', render_mode="human"))

for model in models:

    s, b = env.reset()
    a, _s = model.predict(s, deterministic=True)

    for t in range(1000):
       s, r, terminated, truncated, info = env.step(a)
       if (terminated or truncated):
          s, b = env.reset()
       a, _s = model.predict(s, deterministic=True)
    
env.close()

#################################################


#################################################

#agent = Agent(env)
#
#sim = Sim( agent, env, num_episodes=num_episodes, episode_horizon=env_props['horizon'])
#
#sim.add_listener('episode_started', start_episode_callback )
#sim.add_listener('episode_finished', end_episode_callback )
#
#sim.run()
   

#################################################

