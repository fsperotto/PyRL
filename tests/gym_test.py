import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from pyrl.agents.classic import QLearning as ClassicQLearning
from pyrl.agents.survival import QLearning as SurvivalQLearning
from pyrl.agents.survival import KLearning
from pyrl.environments.survival import SurvivalEnv
from pyrl import Sim, Agent, EnvWrapper, System

from pyrl.environments.grid import GridEnv, GridEnvRender

#################################################

env = EnvWrapper(gym.make('CartPole-v1', render_mode='human'))
agent = Agent(env.observation_space, env.action_space)
system = System(env=env, agent=agent)

system.reset()

for i in range(200):
    state, observation, action, reward, terminated, truncated, info = system.step()
    if terminated or truncated:
        system.reset()

env.close()

#################################################

horizon = 2000

env = EnvWrapper(gym.make('CartPole-v1', render_mode='human'))
agent = Agent(observation_space = env.observation_space, action_space = env.action_space)

def simulation_started_callback(sim, env, agent):
    print("START SIM")
    #print(env.observation_shape)
    #print(env.action_shape)

def simulation_finished_callback(sim, env, agent):
    print("END SIM")

def episode_started_callback(sim, env, agent):
    print("START EPISODE")

def episode_finished_callback(sim, env, agent):
    print("END EPISODE")

def round_started_callback(sim, env, agent):
    #print("START ROUND")
    pass

def round_finished_callback(sim, env, agent):
    #print("END ROUND")
    pass

sim = Sim(agent, env, episode_horizon=horizon,
         simulation_started_callback=simulation_started_callback,
         simulation_finished_callback=simulation_finished_callback,
         episode_started_callback=episode_started_callback,
         episode_finished_callback=episode_finished_callback,
         round_started_callback=round_started_callback,
         round_finished_callback=round_finished_callback
         )

sim.run()



###################################################

