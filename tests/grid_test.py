import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from pyrl.agents.classic import QLearning as ClassicQLearning
from pyrl.agents.classic import PolicyIteration
#from pyrl.agents.classic import DQNAgent
from pyrl.agents.survival import QLearning as SurvivalQLearning
from pyrl.agents.survival import KLearning
from pyrl.environments.survival import SurvivalEnv
from pyrl import Sim, Agent, EnvWrapper, System, PyGameRenderer, PyGameGUI

from pyrl.environments.grid import GridEnv, GridEnvGUI

#################################################

num_rows=3
num_cols=20

minor_r = 5.0
major_r = 100.0

#env = GridEnv(num_cols=3, num_rows=4, reward_matrix=[[-1,-1,5,-1],[0,0,0,0],[-1,-1,-1,100]], reward_mode="s'", render_mode="external")
#env = GridEnv(num_rows=num_rows, num_cols=num_cols, reward_mode="s'", render_mode="external")

reward_targets = {major_r : [(num_cols - 2, num_rows // 2)],
                  minor_r : [(3*(num_cols - 1) // 5, num_rows // 2), ((num_cols - 1) // 3, num_rows // 2)]}

env = GridEnv(num_rows=num_rows, num_cols=num_cols, 
              reward_mode="s'", reward_targets=reward_targets, default_reward=-1.0,
              render_mode="external")

horizon = 100000

initial_budget = 400

repeat = 5

episodes = 1

def repetition_started_callback(sim):
    print("START REPETITION", sim.rep+1, '/', sim.num_repetitions)
    print(env.observation_shape)
    print(env.action_shape)
    print(env.observation_shape + env.action_shape)
    sim.metrics = dict(
        time = 0,
        exploration = np.zeros(env.observation_shape + env.action_shape),
        budget = np.zeros((sim.episode_horizon,), dtype=int)
    )

def repetition_finished_callback(sim):
    print("END REPETITION", sim.rep+1, '/', sim.num_repetitions)

def simulation_started_callback(sim):
    print("START SIMULATION :", sim.agent.name, '- repetition', sim.rep+1)

def simulation_finished_callback(sim):
    print("END SIMULATION")

def episode_started_callback(sim):
    print("START EPISODE", sim.ep+1, '/', sim.num_episodes, ':', sim.episode_horizon, 'rounds')

def episode_finished_callback(sim):
    print("END EPISODE", sim.ep+1, '/', sim.num_episodes)

def round_started_callback(sim):
    #print("START ROUND")
    pass

def round_finished_callback(sim):
    #print("END ROUND")
    #print(".", end="")
    sim.metrics["time"] = sim.t
    #state_action_index = tuple(np.concatenate( (agent.get_state(), agent.get_action()) ) )
    state_action_index = tuple(sim.agent.get_state_action())
    v = sim.metrics["exploration"].item(state_action_index)
    sim.metrics["exploration"].itemset(state_action_index, v+1)
    sim.metrics["budget"][sim.t-1] = sim.agent.b        




###################################################

#agent_PI = PolicyIteration(transitions = env.transition_matrix, 
#                           reward = env.reward_matrix, 
#                           observation_space = env.observation_space, 
#                           action_space = env.action_space)

initial_Q_value = 0.0
gamma = 0.95 #discount factor
alpha = 0.5 #learning rate

epsilon_Q=0.4 #exploration rate

agent_Q = ClassicQLearning(observation_space = env.observation_space, 
                           action_space = env.action_space, 
                           initial_budget=initial_budget, 
                           discount=gamma,
                           learning_rate=alpha,
                           exploration_rate=epsilon_Q,
                           initial_Q_value=initial_Q_value)

survival_threshold = 250 

agent_ST_Q = SurvivalQLearning(observation_space = env.observation_space, 
                           action_space = env.action_space, 
                           initial_budget=initial_budget, 
                           discount=gamma,
                           learning_rate=alpha,
                           exploration_rate=epsilon_Q,
                           initial_Q_value=initial_Q_value,
                           survival_threshold=survival_threshold)

initial_K_value = 200
exploration_threshold = 500
epsilon_K=0.1 #exploration rate

agent_K = KLearning(observation_space = env.observation_space, 
                           action_space = env.action_space, 
                           initial_budget=initial_budget, 
                           discount=gamma,
                           learning_rate=alpha,
                           exploration_rate=epsilon_K,
                           initial_Q_value=initial_Q_value,
                           initial_K_value=initial_K_value,
                           survival_threshold=survival_threshold,
                           exploration_threshold=exploration_threshold)


#agent = agent_Q
#agent = agent_ST_Q
#agent = agent_K
agent = [agent_K, agent_Q]

#window = GridEnvRender(env, agent, cell_size=35)
#env._render_frame = window.refresh


sim = Sim(agent, env, episode_horizon=horizon, num_repetitions=repeat, num_episodes=episodes)

sim.add_listener('round_started', round_started_callback)
sim.add_listener('round_finished', round_finished_callback)
sim.add_listener('episode_started', episode_started_callback)
sim.add_listener('episode_finished', episode_finished_callback)
sim.add_listener('simulation_started', simulation_started_callback)
sim.add_listener('simulation_finished', simulation_finished_callback)
sim.add_listener('repetition_started', repetition_started_callback)
sim.add_listener('repetition_finished', repetition_finished_callback)


#gui = PyGameGUI(sim)
gui = GridEnvGUI(sim, cell_size=35, fps=50, close_on_finish=True)

gui.launch(give_first_step=True, start_running=True)

#gui.close()




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
