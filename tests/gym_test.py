import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from pyrl import Sim, Agent, EnvWrapper, System

#################################################

env = gym.make('MountainCar-v0')

#################################################

def start_episode_callback(sim):
   print("START EPISODE", sim.ep+1, '/', sim.num_episodes)

def end_episode_callback(sim):
   print(sim.t, 'rounds')
   
#################################################

horizon = 500

num_episodes = 5

env = EnvWrapper(gym.make('CartPole-v1', render_mode='human'))
agent = Agent(observation_space = env.observation_space, action_space = env.action_space)

sim = Sim(agent, env, num_episodes=num_episodes, episode_horizon=horizon)

sim.add_listener('episode_started', start_episode_callback )
sim.add_listener('episode_finished', end_episode_callback )

sim.run()

#env.close()

###################################################

horizon = 500

num_episodes = 20

env = EnvWrapper(gym.make('FrozenLake-v1', render_mode='ansi'))
#env = EnvWrapper(gym.make('Taxi-v3', render_mode='human', fps=40))
agent = Agent(observation_space = env.observation_space, action_space = env.action_space)

sim = Sim(agent, env, num_episodes=num_episodes, episode_horizon=horizon)

sim.add_listener('episode_started', start_episode_callback )
sim.add_listener('episode_finished', end_episode_callback )

sim.run()

#env.close()

###################################################

horizon = 500

num_episodes = 20

#env = EnvWrapper(gym.make('FrozenLake-v1', render_mode='ansi'))
env = EnvWrapper(gym.make('Taxi-v3', render_mode='ansi'))
agent = Agent(observation_space = env.observation_space, action_space = env.action_space)

sim = Sim(agent, env, num_episodes=num_episodes, episode_horizon=horizon)

sim.add_listener('episode_started', start_episode_callback )
sim.add_listener('episode_finished', end_episode_callback )

sim.run()

#env.close()
