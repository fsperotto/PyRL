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

num_rows=5
num_cols=50

minor_r = 5.0
major_r = 100.0

#env = GridEnv(num_cols=3, num_rows=4, reward_matrix=[[-1,-1,5,-1],[0,0,0,0],[-1,-1,-1,100]], reward_mode="s'", render_mode="external")
#env = GridEnv(num_rows=num_rows, num_cols=num_cols, reward_mode="s'", render_mode="external")

reward_targets = {major_r : [(num_cols - 2, num_rows // 2)],
                  minor_r : [(3*(num_cols - 1) // 5, num_rows // 2), ((num_cols - 1) // 3, num_rows // 2)]}

env = GridEnv(num_rows=num_rows, num_cols=num_cols, 
              reward_mode="s'", reward_targets=reward_targets, default_reward=-1.0,
              render_mode="external")

horizon = 2000

initial_budget = 400

repeat = 10

def simulation_started_callback(sim, env, agent):
    print("START SIM")
    print(env.observation_shape)
    print(env.action_shape)
    print(env.observation_shape + env.action_shape)
    sim.metrics = dict(
        time = 0,
        exploration = np.zeros(env.observation_shape + env.action_shape),
        budget = np.zeros((sim.episode_horizon,), dtype=int)
    )

def simulation_finished_callback(sim, env, agent):
    print("END SIM")

def episode_started_callback(sim, env, agent):
    print("START EPISODE")

def episode_finished_callback(sim, env, agent):
    print("END EPISODE")
    #pass

def round_started_callback(sim, env, agent):
    #print("START ROUND")
    pass

def round_finished_callback(sim, env, agent):
    #print("END ROUND")
    sim.metrics["time"] = sim.metrics["time"] + 1
    #state_action_index = tuple(np.concatenate( (agent.get_state(), agent.get_action()) ) )
    state_action_index = tuple(agent.get_state_action())
    v = sim.metrics["exploration"].item(state_action_index)
    sim.metrics["exploration"].itemset(state_action_index, v+1)
    sim.metrics["budget"][sim.t-1] = agent.b        




###################################################

env = GridEnv(num_rows=num_rows, num_cols=num_cols, 
              reward_mode="s'", reward_targets=reward_targets, default_reward=-1.0,
              render_mode="external")

initial_Q_value = 0.0
gamma = 0.95 #discount factor
alpha = 0.5 #learning rate
epsilon=0.3 #exploration rate

agent_Q = ClassicQLearning(observation_space = env.observation_space, 
                           action_space = env.action_space, 
                           initial_budget=initial_budget, 
                           discount=gamma,
                           learning_rate=alpha,
                           exploration_rate=epsilon,
                           initial_Q_value=initial_Q_value)

survival_threshold = 250 

agent_ST_Q = SurvivalQLearning(observation_space = env.observation_space, 
                           action_space = env.action_space, 
                           initial_budget=initial_budget, 
                           discount=gamma,
                           learning_rate=alpha,
                           exploration_rate=epsilon,
                           initial_Q_value=initial_Q_value,
                           survival_threshold=survival_threshold)

initial_K_value = 200
exploration_threshold = 500

agent_K = KLearning(observation_space = env.observation_space, 
                           action_space = env.action_space, 
                           initial_budget=initial_budget, 
                           discount=gamma,
                           learning_rate=alpha,
                           exploration_rate=epsilon,
                           initial_Q_value=initial_Q_value,
                           initial_K_value=initial_K_value,
                           survival_threshold=survival_threshold,
                           exploration_threshold=exploration_threshold)


#agent = agent_Q
#agent = agent_ST_Q
agent = agent_K

window = GridEnvRender(env, agent, cell_size=35)

env._render_frame = window.refresh

print("TEST CLASSIC Q AGENT")
        
sim = Sim(agent, env, episode_horizon=horizon, num_simulations=repeat,
         simulation_started_callback=simulation_started_callback,
         simulation_finished_callback=simulation_finished_callback,
         episode_started_callback=episode_started_callback,
         episode_finished_callback=episode_finished_callback,
         round_started_callback=round_started_callback,
         round_finished_callback=round_finished_callback
         )

try:
    sim.run()
except:
    window.close()
    raise

window.close()



###################################################
"""
env = GridEnv(num_rows=num_rows, num_cols=num_cols, 
              reward_mode="s'", reward_targets=reward_targets, default_reward=-1.0,
              render_mode="external")


initial_Q_value = 0.0
initial_K_value = 200
survival_threshold = 250
exploration_threshold = 500
gamma = 0.95 #discount factor
alpha = 0.5 #learning rate
epsilon=0.5 #exploration rate
initial_budget = 500

agent_K = KLearning(
    env.observation_space,
    env.action_space,
    initial_budget=initial_budget,
    should_explore=None,
    discount=gamma,
    learning_rate=alpha,
    initial_Q_value=initial_Q_value,
    initial_K_value=initial_K_value,
    survival_threshold=survival_threshold,
    exploration_threshold=exploration_threshold,
)

window = GridEnvRender(env, agent_K, cell_size=30)

env._render_frame = window.refresh

sim = Sim( agent_K, env, episode_horizon=horizon, 
         simulation_started_callback=simulation_started_callback,
         simulation_finished_callback=simulation_finished_callback,
         episode_started_callback=episode_started_callback,
         episode_finished_callback=episode_finished_callback,
         round_started_callback=round_started_callback,
         round_finished_callback=round_finished_callback
         )


try:
    sim.run()
except:
    env.close()
    raise

env.close()
"""
