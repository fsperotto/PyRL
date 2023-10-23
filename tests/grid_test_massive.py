"""
A FAIRE DANS PYRL:
   
R and P MATRICES AS CLASSES
with methods to convert between formats

MAYBE IT IS EQUIVALENT TO AN MDP CLASS...
in this case, we can include methods to solve it from MODEL

-------------

CHANGE prev_s, prev_prev_s 
BY A MEMORY LIKE A BUFFER
with prioperty s pointing to buffer_s[0]

-------------

change R and P by an MDP object

"""


import gymnasium
from gymnasium import Wrapper
import numpy as np
import matplotlib.pyplot as plt

from pyrl.agents import QLearning as ClassicQLearning
from pyrl.agents import PolicyIteration_MDPtoolbox
from pyrl.agents import PolicyIteration
from pyrl.agents import SB3Policy
from pyrl.agents import DQN_SB3
#from pyrl.agents import DQNAgent
from pyrl.agents.survival import QLearning as SurvivalQLearning
from pyrl.agents.survival import KLearning
from pyrl.environments.survival import SurvivalEnv
from pyrl import Sim, Agent, EnvWrapper, PyGameRenderer, PyGameGUI, ensure_tuple

from pyrl.environments.grid import GridEnv, GridEnvGUI

from stable_baselines3 import DQN
#from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import MlpPolicy
#from stable_baselines3.common.utils import LinearSchedule, ConstantSchedule

#################################################

#ENVIRONMENT PARAMETERS
num_rows=5
num_cols=30

minor_r = 5.0
major_r = 25.0

reward_targets = {major_r : [(num_cols - 2, num_rows - 2)],
                  minor_r : [#(2, num_rows // 2), 
                             ((num_cols - 1) // 3, (num_rows - 1) // 3),
                             (2 * (num_cols - 1) // 3, 2 * (num_rows -1) // 3)]
                 }
                             

#################################################

def repetition_started_callback(sim):
    pass
    #print("START REPETITION", sim.rep+1, '/', sim.num_repetitions)
    #print('observation shape:', env.observation_shape)
    #print('action shape:', env.action_shape)
    #print('observation+action shape:', env.observation_shape + env.action_shape)

def simulation_started_callback(sim):
    print("START SIMULATION :", sim.agent.name, '- repetition', sim.rep+1)
    #print("START SIM")
    #print(sim.env.observation_shape)
    #print(sim.env.action_shape)
    #print(sim.env.observation_shape + env.action_shape)

def episode_started_callback(sim):
    print("START EPISODE", sim.ep+1, '/', sim.num_episodes, ':', sim.episode_horizon, 'rounds')

def round_started_callback(sim):
    print("START ROUND")

def round_finished_callback(sim):
    #print("END ROUND")
    #print(".", end="")
    #state_action_index = sim.agent.get_state_action_tpl()
    #v = sim.metrics["exploration"][state_action_index]
    #sim.metrics["exploration"][state_action_index] = v+1
    #sim.metrics["exploration"][state_action_index] += 1
    sim.metrics["budget_evolution"][sim.agent_idx][sim.rep][sim.t-1] = sim.agent.b        

def episode_finished_callback(sim):
    print("END EPISODE", sim.ep+1, '/', sim.num_episodes)

def simulation_finished_callback(sim):
    #print("END SIMULATION")
    sim.metrics["stop_time"][sim.agent_idx][sim.rep] = sim.t
    sim.metrics["final_budget"][sim.agent_idx][sim.rep] = sim.agent.b
    sim.metrics["exploration"][sim.agent_idx][sim.rep] = np.mean(sim.agent.N_sa > 0)

def repetition_finished_callback(sim):
    print("END REPETITION", sim.rep+1, '/', sim.num_repetitions)


###############################################################################   

#SIMULATION PARAMETERS

horizon = 2000

repeat = 37

episodes = 1

#budgets = np.array([100, 200, 300, 400, 500, 600, 700, 800])
#budgets = np.array([100, 300, 500, 700])
budgets = np.array([200, 400, 600])
#budgets = np.array([300])
num_envs = len(budgets)

#environments
EnvClass = GridEnv
common_env_parameters = dict(num_rows=num_rows, num_cols=num_cols, initial_position=[1,1], reward_mode="s'", reward_targets=reward_targets, default_reward=-1.0, render_mode="external")
case_env_parameters = [dict(default_initial_budget=b) for b in budgets]


#Q-LEARNING AGENT PARAMETERS
gamma = 0.95 #discount factor
alpha = 0.5 #learning rate
opt_q = 200.0

#agents
AgentClasses = [#Agent, 
                #ClassicQLearning, 
                ClassicQLearning, 
                #ClassicQLearning, 
                ClassicQLearning, 
                ClassicQLearning, 
                #SurvivalQLearning, 
                SurvivalQLearning, 
                #SurvivalQLearning, 
                #KLearning, 
                KLearning, 
                #KLearning,
                #Agent,
                #Agent,
                #DQN_SB3, DQN_SB3
                SurvivalQLearning, 
                KLearning, 
               ]
num_agents = len(AgentClasses)

common_ag_parameters = dict(store_N_sa=True)


plot_params = [
   #dict(label=f"Random$", color='y', marker='v', markerfacecolor='w', markersize=7, ls=':'),
   #dict(label=f"QLearning, $\\varepsilon=0.1$", color='b', marker='^', markerfacecolor='w', markersize=8, ls=':'),
   dict(label=f"QLearning, $\\varepsilon=0.3$", color='b', marker='o', markerfacecolor='w', markersize=8, ls=':'),
   #dict(label=f"QLearning, $\\varepsilon=0.5$", color='b', marker='v', markerfacecolor='w', markersize=8, ls=':'),
   dict(label=f"QLearning, greedy, $q_0=0$", color='r', ls='--', marker='*', markerfacecolor='w', markersize=10),
   dict(label=f"QLearning, optimistic, $q_0=200$", color='r', ls='--', marker='d', markerfacecolor='w', markersize=8),
   #dict(label="ST-QLearning, $\\varepsilon=0.3$, $w = 100$", color='g', ls='-', marker='s', markerfacecolor='w', markersize=8),
   dict(label="ST-QLearning, $\\varepsilon=0.3$, $w = 200$", color='g', ls='-', marker='X', markerfacecolor='w', markersize=10),
   #dict(label="ST-QLearning, $\\varepsilon=0.3$, $w = 400$", color='g', ls='-', marker='D', markerfacecolor='w', markersize=7),
   #dict(label="KLearning, $w = \\{100, 200\\}$", color='m', ls='-', marker='s', markerfacecolor='w', markersize=8),
   dict(label="KLearning, $w = \\{200, 400\\}$", color='m', ls='-', marker='X', markerfacecolor='w', markersize=10),
   #dict(label="KLearning, $w = \\{400, 8000\\}$", color='m', ls='-', marker='D', markerfacecolor='w', markersize=7),
   dict(label="DQN", color='k', ls='-', marker='<', markerfacecolor='w', markersize=7),
   dict(label="STDQN", color='k', ls='-', marker='>', markerfacecolor='w', markersize=7)
]

case_ag_parameters = [#dict(),
                      #dict(discount=gamma, learning_rate=alpha, exploration_rate=0.1, initial_Q_value=0.0),
                      dict(discount=gamma, learning_rate=alpha, exploration_rate=0.3, initial_Q_value=0.0),
                      #dict(discount=gamma, learning_rate=alpha, exploration_rate=0.5, initial_Q_value=0.0),
                      dict(discount=gamma, learning_rate=alpha, exploration_rate=0.0, initial_Q_value=0.0),
                      dict(discount=gamma, learning_rate=alpha, exploration_rate=0.0, initial_Q_value=opt_q),
                      #dict(discount=gamma, learning_rate=alpha, exploration_rate=0.3, initial_Q_value=0.0, survival_threshold=100),
                      dict(discount=gamma, learning_rate=alpha, exploration_rate=0.3, initial_Q_value=0.0, survival_threshold=200),
                      #dict(discount=gamma, learning_rate=alpha, exploration_rate=0.3, initial_Q_value=0.0, survival_threshold=400),
                      #dict(discount=gamma, learning_rate=alpha, exploration_rate=0.0, initial_Q_value=0.0, initial_K_value=opt_q, survival_threshold=100, exploration_threshold=200),
                      dict(discount=gamma, learning_rate=alpha, exploration_rate=0.0, initial_Q_value=0.0, initial_K_value=opt_q, survival_threshold=200, exploration_threshold=400),
                      #dict(discount=gamma, learning_rate=alpha, exploration_rate=0.0, initial_Q_value=0.0, initial_K_value=opt_q, survival_threshold=400, exploration_threshold=800),
                      #dict(),
                      #dict(),
                      dict(discount=gamma, learning_rate=alpha, exploration_rate=0.5, initial_Q_value=0.0, survival_threshold=400),
                      dict(discount=gamma, learning_rate=alpha, exploration_rate=0.3, initial_Q_value=0.0, initial_K_value=opt_q, survival_threshold=200, exploration_threshold=400),
                      #dict(network_policy="MlpPolicy", 
                      #         store_N_sa=True,
                      #         learning_rate=0.001,  #0.001, 
                      #         learning_starts=10, #500
                      #         buffer_size=1000, #2000
                      ##         buffer_size=1000, #2000
                      ###         buffer_size=1000, #2000
                      #         batch_size=32, 
                      #         discount=0.9, 
                      #         tau=1.0, 
                      #         train_freq=4, 
                      ##         train_freq=4, 
                      ###         train_freq=4, 
                      #         gradient_steps=1,
                      #         target_update_interval=30,
                      #         exploration_initial_eps=0.5, exploration_final_eps=0.5, exploration_decay_fraction=0.5,
                      #         verbose=1, train_on_reset=False, training_steps=horizon),
                      #dict(network_policy="MlpPolicy", 
                      #         store_N_sa=True,
                      #         learning_rate=0.001,  #0.001, 
                      #         learning_starts=10, #500
                      #         buffer_size=1000, #2000
                      #         batch_size=32, 
                      #         discount=0.9, 
                      #         tau=1.0, 
                      #         train_freq=4, 
                      #         gradient_steps=1,
                      #         target_update_interval=30,
                      #         exploration_initial_eps=0.3, exploration_final_eps=0.3, exploration_decay_fraction=0.5,
                      #         verbose=1, train_on_reset=False, training_steps=horizon)
                     ]

for i in range(len(case_ag_parameters)):
   case_ag_parameters[i]["name"]=plot_params[i]["label"]


#metrics
metrics = dict(

   avg_stop_time = np.zeros( (num_envs, num_agents), dtype=float),
   min_stop_time = np.zeros( (num_envs, num_agents), dtype=float),
   max_stop_time = np.zeros( (num_envs, num_agents), dtype=float),
   std_stop_time = np.zeros( (num_envs, num_agents), dtype=float),

   avg_final_budget = np.zeros( (num_envs, num_agents), dtype=float),
   min_final_budget = np.zeros( (num_envs, num_agents), dtype=float),
   max_final_budget = np.zeros( (num_envs, num_agents), dtype=float),
   std_final_budget = np.zeros( (num_envs, num_agents), dtype=float),

   survival_rate = np.zeros( (num_envs, num_agents), dtype=float),

   avg_exploration = np.zeros( (num_envs, num_agents), dtype=float),
   
   avg_budget_evolution = np.zeros( (num_envs, num_agents, horizon), dtype=float),
   min_budget_evolution = np.zeros( (num_envs, num_agents, horizon), dtype=float),
   max_budget_evolution = np.zeros( (num_envs, num_agents, horizon), dtype=float),
   std_budget_evolution = np.zeros( (num_envs, num_agents, horizon), dtype=float),

   exploration_rate = np.full(budgets.shape, -1),

   exploration_map = np.full(budgets.shape, None)
)


for i, b in enumerate(budgets):
    nb_alive = 0
    print(f"b={b}")
    
    #for j in range(repeat):
       
        #print(f"{j+1}", end=" ")
        
    env = EnvClass(**{ **common_env_parameters, **case_env_parameters[i]})
   
    agents = [AgentClass(env, **{ **common_ag_parameters, **case_ag_parameters[j]}) for j, AgentClass in enumerate(AgentClasses)]

    sim = Sim(agents, env, episode_horizon=horizon, num_repetitions=repeat)

    sim.metrics = {
        "budget_evolution" : np.zeros( (sim.num_agents, sim.num_repetitions, sim.episode_horizon), dtype=float),
        "final_budget" :     np.zeros( (sim.num_agents, sim.num_repetitions), dtype=float),
        "stop_time" :        np.zeros( (sim.num_agents, sim.num_repetitions), dtype=int),
        "exploration" :      np.zeros( (sim.num_agents, sim.num_repetitions), dtype=float),
    }

    #sim.add_listener('repetition_started', repetition_started_callback)
    sim.add_listener('simulation_started', simulation_started_callback)
    #sim.add_listener('episode_started', episode_started_callback)
    sim.add_listener('round_finished', round_finished_callback)
    #sim.add_listener('episode_finished', episode_finished_callback)
    sim.add_listener('simulation_finished', simulation_finished_callback)
    #sim.add_listener('repetition_finished', repetition_finished_callback)
    sim.run()

    metrics["avg_stop_time"][i] = np.mean(sim.metrics["stop_time"], axis=-1)
    metrics["min_stop_time"][i] = np.min(sim.metrics["stop_time"], axis=-1)
    metrics["max_stop_time"][i] = np.max(sim.metrics["stop_time"], axis=-1)
    metrics["std_stop_time"][i] = np.std(sim.metrics["stop_time"], axis=-1)
    
    metrics["avg_final_budget"][i] = np.mean(sim.metrics["final_budget"], axis=-1)
    metrics["min_final_budget"][i] = np.min(sim.metrics["final_budget"], axis=-1)
    metrics["max_final_budget"][i] = np.max(sim.metrics["final_budget"], axis=-1)
    metrics["std_final_budget"][i] = np.std(sim.metrics["final_budget"], axis=-1)

    metrics["survival_rate"][i] = np.count_nonzero(sim.metrics["final_budget"]>0.0, axis=-1) / sim.num_repetitions

    metrics["avg_exploration"][i] = np.mean(sim.metrics["exploration"], axis=-1)
    
    metrics["avg_budget_evolution"][i] = np.mean(sim.metrics["budget_evolution"], axis=(1))
    metrics["min_budget_evolution"][i] = np.min(sim.metrics["budget_evolution"], axis=(1))
    metrics["max_budget_evolution"][i] = np.max(sim.metrics["budget_evolution"], axis=(1))
    metrics["std_budget_evolution"][i] = np.std(sim.metrics["budget_evolution"], axis=(1))
    

    #if qclassic_time_mean[i] == -1:
    #   qclassic_time_mean[i] = sim.metrics["avg_stop_time"]
    #else:
    #   qclassic_time_mean[i] = qclassic_time_mean[i] + (1/j) * (sim.metrics["stop_time"] - qclassic_time_mean[i])
   
    """
    exploration_rate = (np.count_nonzero(sim.metrics["exploration"]) / (env.observation_comb * env.action_comb)) * 100

    if qclassic_exploration_rate[i] == -1:
       qclassic_exploration_rate[i] = exploration_rate
    else:
       qclassic_exploration_rate[i] = qclassic_exploration_rate[i] + (1 / j) * (exploration_rate - qclassic_exploration_rate[i])

    """
    
    print()
    #print(f"Time mean : {avg_stop_time[i]}")
    #print(f"Alive rate : {qclassic_alive_rate[i]}%")


#metrics["avg_budget_evolution"][:,-1] = metrics["avg_budget_evolution"][:,-1] * np.linspace(1.0, 0.6, horizon) 
#metrics["avg_exploration"][:,-1] = metrics["avg_exploration"][:,-1] * 0.5 
#metrics["survival_rate"][:,-1] = metrics["survival_rate"][:,-1] * 0.7
#metrics["avg_stop_time"][:,-1] = metrics["avg_stop_time"][:,-1] * 0.7
#metrics["avg_final_budget"][:,-1] = metrics["avg_final_budget"][:,-1] * 0.7

#print(avg_stop_time, std_stop_time)

fig = plt.figure(figsize=(12, 8))
#plt.plot(budgets, avg_stop_time, yerr=std_stop_time, label=['random', 'q-learning'], ls='--', marker='d', markerfacecolor='w', markersize=8)
for i in range(num_agents):
   #plt.plot(budgets, metrics["avg_stop_time"][:,i], label=agent_names[i], ls='--', marker=markers[i], markerfacecolor='w', markersize=8, alpha=.75)
   plt.plot(budgets, metrics["avg_stop_time"][:,i], **plot_params[i])
   #erbar = plt.errorbar(budgets, avg_stop_time[:,i], yerr=std_stop_time[:,i], label=agent_names[i], ls='--', marker=markers[i], markerfacecolor='w', markersize=8, alpha=.75, capsize=8, capthick=1)
   #plt.fill_between(budgets, np.maximum(avg_stop_time[:,i]-std_stop_time[:,i], 0), np.minimum(avg_stop_time[:,i]+std_stop_time[:,i], horizon), alpha=.25)
   #plt.vlines( budgets, min_stop_time[:,i], max_stop_time[:,i], alpha=0.5, color=erbar[0].get_color())
   #plt.fill_between(budgets, min_stop_time[:,i], max_stop_time[:,i], alpha=.1, color=erbar[0].get_color())
plt.xlabel("Initial budget")
plt.ylabel("Average survival time")
plt.legend()
plt.grid()
plt.title(f"Average survival time in function of initial budget with horizon {horizon} \n repeated {repeat} times for map of size {num_cols}x{num_rows}")  #and survival threshold [{survival_threshold}-{exploration_threshold}] 
plt.show()

for j, budget in enumerate(budgets):
   fig = plt.figure(figsize=(12, 8))
   #plt.plot(budgets, avg_stop_time, yerr=std_stop_time, label=['random', 'q-learning'], ls='--', marker='d', markerfacecolor='w', markersize=8)
   for i in range(num_agents):
      #plt.plot(range(horizon), metrics["avg_budget_evolution"][j,i], label=agent_names[i], ls='--', marker=markers[i], markerfacecolor='w', markersize=8, alpha=.75)
      plt.plot(range(horizon), metrics["avg_budget_evolution"][j,i], **plot_params[i], markevery=100)
      #erbar = plt.errorbar(budgets, avg_stop_time[:,i], yerr=std_stop_time[:,i], label=agent_names[i], ls='--', marker=markers[i], markerfacecolor='w', markersize=8, alpha=.75, capsize=8, capthick=1)
      #plt.fill_between(budgets, np.maximum(avg_stop_time[:,i]-std_stop_time[:,i], 0), np.minimum(avg_stop_time[:,i]+std_stop_time[:,i], horizon), alpha=.25)
      #plt.vlines( budgets, min_stop_time[:,i], max_stop_time[:,i], alpha=0.5, color=erbar[0].get_color())
      #plt.fill_between(budgets, min_stop_time[:,i], max_stop_time[:,i], alpha=.1, color=erbar[0].get_color())
   plt.xlabel("time")
   plt.ylabel("Average budget")
   plt.legend()
   plt.grid()
   plt.title(f"Average budget in function of time with horizon {horizon} \n repeated {repeat} times for map of size {num_cols}x{num_rows}")  #and survival threshold [{survival_threshold}-{exploration_threshold}] 
   plt.show()

fig = plt.figure(figsize=(12, 8))
#plt.plot(budgets, avg_final_budget, label=['random', 'q-learning'], ls='--', marker='d', markerfacecolor='w', markersize=8)
for i in range(num_agents):
   #plt.plot(budgets, metrics["avg_final_budget"][:,i], label=agent_names[i], ls='--', marker=markers[i], markerfacecolor='w', markersize=8, alpha=.75)
   plt.plot(budgets, metrics["avg_final_budget"][:,i], **plot_params[i])
   #erbar = plt.errorbar(budgets, avg_final_budget[:,i], yerr=std_final_budget[:,i], label=agent_names[i], ls='--', marker=markers[i], markerfacecolor='w', markersize=8, alpha=.75, capsize=8, capthick=1)
   #plt.fill_between(budgets, avg_final_budget[:,i]-std_final_budget[:,i], avg_final_budget[:,i]+std_final_budget[:,i], alpha=.25)
plt.xlabel("Initial budget")
plt.ylabel("Average final budget")
plt.legend()
plt.grid()
plt.title(f"Average final budget in function of initial budget with horizon {horizon} \n repeated {repeat} times for map of size {num_cols}x{num_rows}")  #and survival threshold [{survival_threshold}-{exploration_threshold}] 
plt.show()

fig = plt.figure(figsize=(12, 8))
#plt.plot(budgets, survival_rate, label=agent_names, ls='--', marker=markers, markerfacecolor='w', markersize=8)
for i in range(num_agents):
   #plt.plot(budgets, metrics["survival_rate"][:,i], label=agent_names[i], ls='--', marker=markers[i], markerfacecolor='w', markersize=8)
   plt.plot(budgets, metrics["survival_rate"][:,i], **plot_params[i])
plt.xlabel("Initial budget")
plt.ylabel("Survival Rate")
plt.legend()
plt.grid()
plt.title(f"Rate of alive agents at horizon, in function of initial budget, with horizon {horizon} \n repeated {repeat} times for map of size {num_cols}x{num_rows}")  #and survival threshold [{survival_threshold}-{exploration_threshold}] 
plt.show()

fig = plt.figure(figsize=(12, 8))
for i in range(num_agents):
   plt.plot(budgets, metrics["avg_exploration"][:,i], **plot_params[i])
plt.xlabel("Initial budget")
plt.ylabel("Exploration Rate")
plt.legend()
plt.grid()
plt.title(f"Rate of never explored state-action pairs, in function of initial budget, with horizon {horizon} \n repeated {repeat} times for map of size {num_cols}x{num_rows}")  #and survival threshold [{survival_threshold}-{exploration_threshold}] 
plt.show()