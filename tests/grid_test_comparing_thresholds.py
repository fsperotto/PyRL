"""
This script create results for OLA 2024


------------
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

import sys

import pickle

#from numba import njit

from argparse import ArgumentParser

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

DEF_COLS = 25
DEF_ROWS = 5
DEF_HORIZON = 2000
DEF_PLOT_MARKEVERY = 1000
DEF_OPT_Q = +100.0
DEF_MAJ_R = +4.0
DEF_MID_R = +4.0
DEF_MIN_R = -1.0
DEF_R_MEAN = 0.0
DEF_R_VAR = 0.1
DEF_GAMMA = 0.95
DEF_INI_BUDGET = 200
#DEF_MID_R_POSITIONS = [1/3, 2/3]
#DEF_R_SPOTS = {(1.,1.):+5.0, (.5,.5):+1.0}
#DEF_R_SPOTS = {(DEF_COLS-2,DEF_COLS-2) : DEF_MAJ_R,
#               (.5,.5) : DEF_MID_R}
DEF_R_SPREAD = 0.1
DEF_R_SPOTS = {(DEF_COLS-2,DEF_ROWS-2) : DEF_MAJ_R,
               (DEF_COLS//4,DEF_ROWS//4) : DEF_MID_R/2,
               (DEF_COLS//2+2,DEF_ROWS//2) : DEF_MID_R}
DEF_R_MATRIX = np.transpose(np.tile(np.linspace(DEF_MIN_R/2, DEF_MIN_R, DEF_COLS), (DEF_ROWS, 1)))
DEF_ALPHA = 0.5
DEF_EPSILON = 0.1


#################################################

def round_finished_callback(sim):
    sim.metrics["budget_evolution"][sim.agent_idx][sim.t-1] = sim.agent.b        
    sim.metrics["exploration_evolution"][sim.agent_idx][sim.t-1] = np.mean(sim.agent.N_sa > 0)


###############################################################################

def run_all(budget, EnvClass, common_env_parameters, AgentClasses, common_ag_parameters, case_ag_parameters, plot_params, horizon = 1000):

   num_agents = len(AgentClasses)
   
   #metrics
   metrics = dict(
      exploration_evolution = np.zeros( (num_agents, horizon), dtype=float),
      budget_evolution = np.zeros( (num_agents, horizon), dtype=float),
   )


   print(f"b={budget}")
    
        
   env = EnvClass(**common_env_parameters)
   
   agents = [AgentClass(env, **{ **common_ag_parameters, **case_ag_parameters[j]}) for j, AgentClass in enumerate(AgentClasses)]

   sim = Sim(agents, env, episode_horizon=horizon)

   sim.metrics = {
        "budget_evolution"      : np.zeros( (sim.num_agents, sim.episode_horizon), dtype=float),
        "exploration_evolution" : np.zeros( (sim.num_agents, sim.episode_horizon), dtype=float),
   }

   sim.add_listener('round_finished', round_finished_callback)
    
   sim.run()

   metrics["exploration_evolution"] = sim.metrics["exploration_evolution"]
   metrics["budget_evolution"] = sim.metrics["budget_evolution"]
    
   print()

   return metrics


###############################################################################


def plot_all(budget, num_agents, metrics, plot_params, horizon, num_cols, num_rows):

   
    fig = plt.figure(figsize=(8, 5))
    for i in range(num_agents):
       plt.plot(range(horizon), metrics["budget_evolution"][i], **plot_params[i], linestyle="-")
    plt.xlabel("time")
    plt.ylabel("budget")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("output/budget_comp.pdf")
    fig.show()

    fig = plt.figure(figsize=(8, 5))
    for i in range(num_agents):
       plt.plot(range(horizon), metrics["exploration_evolution"][i], **plot_params[i], linestyle="-")
    plt.xlabel("time")
    plt.ylabel("exploration")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("output/explo_comp.pdf")
    fig.show()
    

            

                  

###############################################################################

def run(num_rows=DEF_ROWS, num_cols=DEF_COLS, 
        initial_budget=DEF_INI_BUDGET, 
        reward_matrix = DEF_R_MATRIX,
        default_reward:float = DEF_R_MEAN,
        reward_variance:float = DEF_R_VAR,
        reward_spots=DEF_R_SPOTS, reward_spread=DEF_R_SPREAD,
        horizon=DEF_HORIZON, 
        gamma=DEF_GAMMA,
        alpha=DEF_ALPHA,
        epsilon=DEF_EPSILON,
        opt_ini_q=DEF_OPT_Q,         
        #minor_reward = DEF_MID_R, major_reward = DEF_MAJ_R, minor_positions = DEF_MID_R_POSITIONS
        ):
   
   
   #reward_targets = { major_reward : [(num_cols - 2, num_rows - 2)],
   #                   minor_reward : [ ( int((num_cols-1)*p) , int((num_rows-1)*p) ) for p in minor_positions ] }
    

   #SIMULATION PARAMETERS
   
   #environments
   EnvClass = GridEnv
   common_env_parameters = dict(num_rows=num_rows, num_cols=num_cols, initial_position=[1,1], reward_mode="s'", 
                                reward_matrix=reward_matrix,
                                reward_spots=reward_spots, reward_spread=reward_spread,
                                default_reward=default_reward, reward_variance=reward_variance,
                                default_initial_budget=initial_budget,
                                render_mode="external")
   
                        
   common_ag_parameters = dict(store_N_sa=True)
   
   
   AgentClasses = [
                   KLearning,
                   KLearning
                  ]
   
   plot_params = [
      dict(label="STQ, $q_0=0$, $k_0=200$, $\\varepsilon=0.1$, $w = \\{300, 300\\}$", color='r'),
      dict(label="STQ, $q_0=0$, $k_0=200$, $\\varepsilon=0.1$, $w = \\{300, 600\\}$", color='b')
     ]
   
   case_ag_parameters = [
                         dict(discount=gamma, learning_rate=alpha, exploration_rate=epsilon, initial_Q_value=0.0, initial_K_value=opt_ini_q, survival_threshold=100, exploration_threshold=100),
                         dict(discount=gamma, learning_rate=alpha, exploration_rate=epsilon, initial_Q_value=0.0, initial_K_value=opt_ini_q, survival_threshold=100, exploration_threshold=200)
                        ]
   
   
   num_agents = len(AgentClasses)
   print(num_agents)
   
   
   for i in range(len(case_ag_parameters)):
      case_ag_parameters[i]["name"]=plot_params[i]["label"]

    
    
   metrics = run_all(initial_budget, EnvClass, common_env_parameters, AgentClasses, common_ag_parameters, case_ag_parameters, plot_params, horizon=horizon)
   #with open("output/metrics.pickle", "wb") as f:
   #    pickle.dump(metrics, f)
   plot_all(initial_budget, num_agents, metrics, plot_params, horizon, num_cols, num_rows)

   return 0


##################################################################################################

def main() :
    'main entry point'

    # parse arguments
    parser = ArgumentParser(description='Run massive grid simulation')
    
    parser.add_argument("-y", "--num-rows", type=int, default=DEF_ROWS, help="Number of grid rows.")
    parser.add_argument("-x", "--num-cols", type=int, default=DEF_COLS, help="Number of grid columns.")
    parser.add_argument("-b", "--initial-budget", type=list, default=DEF_INI_BUDGET, help="Initial budget.")
    parser.add_argument("-v", "--reward-variance", type=float, default=DEF_R_VAR, help="Default reward variance.")
    parser.add_argument("-m", "--reward-matrix", default=DEF_R_MATRIX, help="Reward matrix.")
    parser.add_argument("-o", "--reward-spots", type=dict, default=DEF_R_SPOTS, help="Reward spots.")
    parser.add_argument("-s", "--reward-spread", type=float, default=DEF_R_SPREAD, help="Reward spreading.")
    parser.add_argument("-t", "--horizon", type=int, default=DEF_HORIZON, help="Max time horizon.")
    parser.add_argument("-g", "--gamma", type=float, default=DEF_GAMMA, help="Discount factor.")
    parser.add_argument("-q", "--opt-ini-q", type=float, default=DEF_OPT_Q, help="Optimistic value for Q initialization.")
    parser.add_argument("-a", "--alpha", type=float, default=DEF_ALPHA, help="Learning rate.")
    parser.add_argument("-e", "--epsilon", type=float, default=DEF_EPSILON, help="Exploration rate.")

    args = parser.parse_args()
    
    run(**vars(args))

##################################################################################################

if __name__ == '__main__':
    sys.exit(main())
    
    
    