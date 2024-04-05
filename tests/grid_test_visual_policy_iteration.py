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

import sys
from argparse import ArgumentParser

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D


from pyrl.agents import PolicyIteration_MDPtoolbox
from pyrl.agents import PolicyIteration
from pyrl import Sim, ensure_tuple

from pyrl.environments.grid import GridEnv, GridEnvGUI

#################################################

DEF_COLS = 25
DEF_ROWS = 5
DEF_HORIZON = 1000
#DEF_OPT_Q = +100.0
DEF_MAJ_R = +10.0
DEF_MID_R = +3.0
DEF_MIN_R = -1.0
DEF_R_MEAN = 0.0
DEF_R_VAR = 1.0
DEF_GAMMA = 0.95
#DEF_ALPHA = 0.5
#DEF_EPSILON = 0.1
DEF_REPETITIONS = 1
DEF_EPISODES = 1
#DEF_INI_BUDGETS = [100, 200, 300, 400, 500, 600, 700, 800]
#DEF_MID_R_POSITIONS = [1/3, 2/3]
DEF_LAUNCH_GUI = False
#DEF_R_SPOTS = {(1.,1.):+5.0, (.5,.5):+1.0}
#DEF_R_SPOTS = {(DEF_COLS-2,DEF_COLS-2) : DEF_MAJ_R,
#               (.5,.5) : DEF_MID_R}
DEF_R_SPREAD = 0.3
DEF_R_SPOTS = {(DEF_COLS-2,DEF_ROWS-2) : DEF_MAJ_R,
               (DEF_COLS//4,DEF_ROWS//4) : DEF_MID_R/2,
               (DEF_COLS//2+2,DEF_ROWS//2) : DEF_MID_R}
DEF_R_MATRIX = np.transpose(np.tile(np.linspace(DEF_MIN_R/2, DEF_MIN_R, DEF_COLS), (DEF_ROWS, 1)))

#################################################


class MDP:
   
   def __init__(self, R, P, observation_space, action_space, R_mode="(fact):s'-->r", P_mode="(fact):sa-->s'"):
      self.R = R
      self.P = P
      self.observation_space = observation_space
      self.action_space = action_space
      self.R_mode=R_mode
      self.P_mode=P_mode
      
   """
   def get_R(self, mode=None):
      if mode is None or mode == self.R_mode:
         return self.R
      else:
         if self.R_mode=="s':r":
            if mode == "s':r":
   """

#################################################

def run(num_rows:int=DEF_ROWS, num_cols:int=DEF_COLS,
              initial_position={(0,0):1.0},
              default_reward:float = DEF_R_MEAN,
              default_reward_variance:float = DEF_R_VAR,
              reward_mode="s'",
              reward_matrix=DEF_R_MATRIX,
              #minor_reward = DEF_MID_R, major_reward = DEF_MAJ_R, minor_positions = DEF_MID_R_POSITIONS,  
              reward_spots=DEF_R_SPOTS,
              reward_spread:float=DEF_R_SPREAD,              
              horizon:int=DEF_HORIZON, num_repetitions:int=DEF_REPETITIONS, num_episodes:int=DEF_EPISODES,
              initial_budget:float=400.0,
              gamma:float=DEF_GAMMA,
              max_policy_iterations:int=50,
              max_value_iterations:int=100,
              theta=0.01,
              launch_gui:bool=False) :

    #reward_targets = { major_reward : [(num_cols - 2, num_rows - 2)],
    #                   minor_reward : [ ( int((num_cols-1)*p) , int((num_rows-1)*p) ) for p in minor_positions ] }


    env = GridEnv(num_rows=num_rows, num_cols=num_cols, 
                  reward_mode=reward_mode,
                  default_reward=default_reward, 
                  reward_matrix=reward_matrix,
                  reward_spots=reward_spots, reward_spread=reward_spread,
                  initial_position=initial_position,                  
                  default_initial_budget=initial_budget,
                  render_mode="external")


    #################################################

    def repetition_started_callback(sim):
        print("START REPETITION", sim.rep+1, '/', sim.num_repetitions)

    def simulation_started_callback(sim):
        print("START SIMULATION :", sim.agent.name, '- repetition', sim.rep+1)

    def episode_started_callback(sim):
        print("START EPISODE", sim.ep+1, '/', sim.num_episodes, ':', sim.episode_horizon, 'rounds')

    def round_started_callback(sim):
        print("START ROUND")

    def round_finished_callback(sim):
        print("END ROUND")

    def episode_finished_callback(sim):
        print("END EPISODE", sim.ep+1, '/', sim.num_episodes)

    def simulation_finished_callback(sim):
       print("END SIMULATION")

       X = range(sim.env.num_cols)
       Y = range(sim.env.num_rows)
       Y, X = np.meshgrid(Y, X)
       
       R = sim.env.get_reward_matrix()

       # Plot R 2D
       #plt.imshow(R, cmap=cm.RdYlGn)
       plt.matshow(R.transpose(), cmap=cm.RdYlGn)
       plt.colorbar( shrink=0.5, aspect=5)
       plt.tight_layout()
       plt.show()
      
       # Plot R 3D
       fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
       ax.plot_surface( X, Y, R, cmap=cm.RdYlGn, linewidth=0, antialiased=True)
       #ax.zaxis.set_major_formatter('{x:.02f}')
       #ax.set_aspect("equal")
       ax.set_box_aspect(aspect=(5, 1, 1))
       ax.yaxis.set_major_locator(MaxNLocator(integer=True))
       ax.invert_yaxis()
       #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 0.5, 0.5, 1]))
       #fig.colorbar(surf, shrink=0.5, aspect=5)

       plt.tight_layout()
       plt.show()

   
       # Plot V
       if hasattr(sim.agent, "V") and sim.agent.V is not None:

          # Plot R 2D
          #plt.imshow(sim.agent.V, cmap=cm.RdYlGn)
          plt.matshow(sim.agent.V.transpose(), cmap=cm.RdYlGn)
          plt.colorbar( shrink=0.5, aspect=5)
          plt.tight_layout()
          plt.show()
          
          fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
          ax.plot_surface( X, Y, sim.agent.V, cmap=cm.RdYlGn, linewidth=0, antialiased=True)
          #ax.set_aspect("equal")
          ax.set_box_aspect(aspect=(5, 1, 1))
          #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
          ax.yaxis.set_major_locator(MaxNLocator(integer=True))
          ax.invert_yaxis()
          #ax.zaxis.set_major_locator(MaxNLocator(integer=True))
          #ax.zaxis.set_major_formatter('{x:.02f}')
          #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 0.5, 0.5, 1]))
      
          plt.tight_layout()
          plt.show()
       

#       # Plot policy
#       if hasattr(sim.agent, "policy") and sim.agent.policy is not None:
#         
#          action_to_dX = np.array([+1, 0, -1, 0], dtype=float),  #0: right,  #1: down,  #2:left,  #3:up
#          action_to_dY = np.array([0, +1, 0, -1], dtype=float),  #0: right,  #1: down,  #2:left,  #3:up
#
#          dX = np.choose(sim.agent.policy.transpose(), action_to_dX)
#          dY = np.choose(sim.agent.policy.transpose(), action_to_dY)
#          
#          fig, ax = plt.subplots(figsize=(6, 6))
#          plt.arrow(X, Y, dX, dY, head_width=0.1)
   
   
    def repetition_finished_callback(sim):
        print("END REPETITION", sim.rep+1, '/', sim.num_repetitions)


    ###################################################
    #reward in the form "factored" + "s'"
    R = env.get_reward_matrix()
    print("shape(R(s'))) =", R.shape)
    #reward in the form "flat" + "s'"
    R = R.reshape(env.observation_comb)
    print("shape(flat(R(s'))) =",R.shape)
    #reward in the form "flat" + "ass'"
    R = np.tile(R, (env.action_comb, env.observation_comb, 1))
    print("shape(flat(R(sas'))) =", R.shape)

    #transition in the form "sas'" + "factored" + "deterministic"
    P = env.get_transition_matrix()
    print("shape(P(sa)-->s') =", P.shape)

    
    ##transition in the form "ass'" + "factored" + "stochastic"
    #XP = np.zeros( env.action_shape + env.observation_shape + env.observation_shape , dtype=float)
    #print(XP.shape)
    #for act in env.action_iterator:
    #   for obs in env.observation_iterator:
    #      next_obs = ensure_tuple(P[obs+act])
    #      XP[act + obs + next_obs] = 1.0
    
    
    #transition in the form "ass'" + "flat" + "stochastic"
    XP = np.zeros( (env.action_comb, env.observation_comb, env.observation_comb) , dtype=float)
    for act_idx, act_tpl in enumerate(env.action_iterator):
       for obs_idx, obs_tpl in enumerate(env.observation_iterator):
          next_obs_tpl = ensure_tuple(P[obs_tpl+act_tpl])
          next_obs_idx = np.ravel_multi_index(next_obs_tpl, env.observation_shape)
          XP[act_idx, obs_idx, next_obs_idx] = 1.0
          
    """
    #transition in the form "sas'" + "flat(sa)/factored(s')" + "deterministic"
    P = P.reshape( (env.observation_comb, env.action_comb, 2) )
    print(P.shape)
    #transition in the form "ass'" + "flat(as)/factored(s')" + "deterministic"
    P = np.swapaxes(P,0,1)
    print(P.shape)
    #transition in the form "ass'" + "flat" + "deterministic"
    #P= np.multiply(P, [env.num_cols,1])
    P= np.multiply(P, [1,env.num_rows])
    print(P.shape)
    P = np.sum(P,axis=2)
    print(P.shape)
    #transition in the form "ass'" + "flat" + "stochastic"
    #P = np.expand_dims(P, axis=-1)
    #print(P.shape)
    #P = np.repeat(P, env.observation_comb, axis=-1)
    #print(P.shape)
    XP = np.zeros( (env.action_comb, env.observation_comb, env.observation_comb) , dtype=float)
    print(XP.shape)
    for act in range(env.action_comb):
       for obs in range(env.observation_comb):
          next_obs = ensure_tuple(P[act, obs])
          XP[act, obs, next_obs] = 1.0
    """

    agent_PI_MDPTB = PolicyIteration_MDPtoolbox(env, R=R, P=XP, discount=gamma, max_iter=max_policy_iterations)

    ###################################################
    #reward in the form "factored" + "s'"
    R = env.get_reward_matrix()
    print(R.shape)
    #reward in the form "factored" + "sas'"
    XR = np.tile( R, env.observation_shape+env.action_shape+tuple([1]*env.observation_ndim) )
    print(XR.shape)

    #transition in the form "sas'" + "factored" + "deterministic"
    P = env.get_transition_matrix()
    print(P.shape)
    #transition in the form "sas'" + "factored" + "probabilistic"
    XP = np.zeros( env.observation_shape+env.action_shape+env.observation_shape , dtype=float)
    print(XP.shape)
    for obs in env.observation_iterator:
       for act in env.action_iterator:
          next_obs = ensure_tuple(P[obs+act])
          XP[obs+act+next_obs] = 1.0

    agent_PI = PolicyIteration(env=env, R=XR, P=XP, discount=gamma, max_policy_iterations=max_policy_iterations, max_value_iterations=max_value_iterations, theta=theta)


    #agents = [agent_PI, agent_PI_MDPTB]
    agents = [agent_PI_MDPTB]

    sim = Sim(agents, env, episode_horizon=horizon, num_repetitions=num_repetitions, num_episodes=num_episodes)

    #sim.add_listener('round_started', round_started_callback)
    #sim.add_listener('round_finished', round_finished_callback)
    sim.add_listener('episode_started', episode_started_callback)
    sim.add_listener('episode_finished', episode_finished_callback)
    sim.add_listener('simulation_started', simulation_started_callback)
    sim.add_listener('simulation_finished', simulation_finished_callback)
    sim.add_listener('repetition_started', repetition_started_callback)
    sim.add_listener('repetition_finished', repetition_finished_callback)

    if launch_gui :   

       grid_elements=[
            #{'pos':0, 'label':'agent', 'data':[{'source':'env', 'attr':'R', 'type':"s'"}]},
            # {'pos':1, 'label':'N_sa', 'source':'agent', 'attr':'N_sa', 'type':'sa', 'color_mode':'inversed_log_grayscale', 'backcolor':None},
             {'pos':1, 'label':'V', 'source':'agent', 'attr':'V', 'type':'s', 'color_mode':'grayscale', 'backcolor':None},
            # {'pos':2, 'label':'Q', 'source':'agent', 'attr':'Q', 'type':'sa', 'color_mode':'grayscale', 'backcolor':None},
            # {'pos':3, 'label':'Q_target', 'source':'agent', 'attr':'Q_target', 'type':'sa', 'color_mode':'grayscale', 'backcolor':None},
             {'pos':2, 'label':'policy', 'source':'agent', 'attr':'policy', 'type':'sa', 'color_mode':'grayscale', 'backcolor':(0,0,0)},
           ]
   
       gui = GridEnvGUI( sim, cell_size=20, fps=50, grid_elements=grid_elements, close_on_finish=False)
   
       gui.launch(give_first_step=True, start_running=False)
       
    else:
       
       #sim.reset()
       sim.run()


##################################################################################################

def main():
    'main entry point'

    # parse arguments
    parser = ArgumentParser(description='Run policy iteration on grid simulation')
    parser.add_argument("-w", "--launch_gui", action='store_true', default=DEF_LAUNCH_GUI, help="Visual simulation (GUI)")
    parser.add_argument("-y", "--num-rows", type=int, default=DEF_ROWS, help="Number of grid rows.")
    parser.add_argument("-x", "--num-cols", type=int, default=DEF_COLS, help="Number of grid columns.")
    parser.add_argument("-b", "--initial_budget", type=float, default=200.0, help="Initial budget.")
    parser.add_argument("-d", "--default-reward", type=float, default=DEF_R_MEAN, help="Default reward mean value.")
    parser.add_argument("-v", "--default-reward-variance", type=float, default=DEF_R_VAR, help="Default reward variance.")
    parser.add_argument("-m", "--reward-matrix", default=DEF_R_MATRIX, help="Reward matrix.")
    parser.add_argument("-o", "--reward-spots", type=dict, default=DEF_R_SPOTS, help="Reward spots.")
    parser.add_argument("-s", "--reward-spread", type=float, default=DEF_R_SPREAD, help="Reward spreading.")
    parser.add_argument("-t", "--horizon", type=int, default=DEF_HORIZON, help="Max time horizon.")
    parser.add_argument("-n", "--num-episodes", type=int, default=DEF_EPISODES, help="Number of episodes.")
    parser.add_argument("-r", "--num-repetitions", type=int, default=DEF_REPETITIONS, help="Number of repetitions.")
    parser.add_argument("-g", "--gamma", type=float, default=DEF_GAMMA, help="Discount factor.")
#    parser.add_argument("-q", "--opt-ini-q", type=float, default=DEF_OPT_Q, help="Optimistic value for Q initialization.")
#    parser.add_argument("-a", "--alpha", type=float, default=DEF_ALPHA, help="Learning rate.")
#    parser.add_argument("-e", "--epsilon", type=float, default=DEF_EPSILON, help="Exploration rate.")
    parser.add_argument("-i", "--max_policy_iterations", type=int, default=30, help="Maximum policy iterations.")
    parser.add_argument("-j", "--max_value_iterations", type=int, default=100, help="Maximum value iterations within a policy iteration.")
    parser.add_argument("-z", "--theta", type=float, default=0.001, help="Maximum value error to consider convergence.")
    #parser.add_argument("--minor_reward", type=float, default=DEF_MID_R, help="Small positive reward value.")
    #parser.add_argument("--major_reward", type=float, default=DEF_MAJ_R, help="Big positive reward value.")
    #parser.add_argument("--minor_positions", type=list, default=DEF_MID_R_POSITIONS, help="Relative positions into the diagonal.")

    args = parser.parse_args()
    
    run(**vars(args))

##################################################################################################

if __name__ == "__main__":
    sys.exit(main())
