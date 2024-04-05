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

import gymnasium
from gymnasium import Wrapper
from gymnasium.wrappers import FlattenObservation

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

import time

#################################################

DEF_COLS = 25
DEF_ROWS = 5
DEF_HORIZON = 10000
DEF_OPT_Q = +100.0
DEF_MAJ_R = +10.0
DEF_MID_R = +3.0
DEF_MIN_R = -1.0
DEF_R_MEAN = 0.0
DEF_R_VAR = 1.0
DEF_GAMMA = 0.95
DEF_ALPHA = 0.5
DEF_EPSILON = 0.1
DEF_REPETITIONS = 5
DEF_EPISODES = 1
DEF_INI_BUDGET = 100
#DEF_MID_R_POSITIONS = [1/3, 2/3]
DEF_LAUNCH_GUI = True

DEF_R_SPREAD = 0.3
DEF_R_SPOTS = {(DEF_COLS-2,DEF_ROWS-2) : DEF_MAJ_R,
               (DEF_COLS//4,DEF_ROWS//4) : DEF_MID_R/2,
               (DEF_COLS//2+2,DEF_ROWS//2) : DEF_MID_R}
DEF_R_MATRIX = np.transpose(np.tile(np.linspace(DEF_MIN_R/2, DEF_MIN_R, DEF_COLS), (DEF_ROWS, 1)))

#################################################


def run(num_rows:int = DEF_ROWS, num_cols:int = DEF_COLS, 
        initial_position:dict = {(0,0):1.0},
        reward_mode:str = "s'",
        reward_matrix = DEF_R_MATRIX,
        default_reward:float = DEF_R_MEAN,
        reward_variance:float = DEF_R_VAR,
        #reward_spots:dict = {(1.,1.):+25.0, (.5,.5):+5.0},
        reward_spots=DEF_R_SPOTS,
        reward_spread:float = DEF_R_SPREAD, 
        initial_budget:float = DEF_INI_BUDGET, 
        horizon:int = DEF_HORIZON, num_episodes:int = DEF_EPISODES, num_repetitions:int = DEF_REPETITIONS, 
        #minor_reward:float = DEF_MID_R, major_reward:float = DEF_MAJ_R, minor_positions:list = DEF_MID_R_POSITIONS,
        gamma=DEF_GAMMA,
        alpha=DEF_ALPHA,
        epsilon=DEF_EPSILON,
        opt_ini_q=DEF_OPT_Q, 
        agents:list = ["PolicyIteration", {"gamma":DEF_GAMMA},
                             "QLearning", {"gamma":DEF_GAMMA, "alpha":DEF_ALPHA, "epsilon":DEF_EPSILON, "initial_q_value":0.0} ]  
        ):

    #SIMULATION PARAMETERS
    #horizon = 800
    #repeat = 1
    #episodes = 1
    #initial_budget = 400

    #CLASSIC Q-LEARNING AGENT PARAMETERS
    initial_Q_value = 0.0
    #gamma = 0.95 #discount factor
    #alpha = 0.5 #learning rate
    epsilon_Q = epsilon #exploration rate

    #SURVIVAL Q-LEARNING AGENT PARAMETERS
    survival_threshold = 100 

    #K-LEARNING AGENT PARAMETERS
    initial_K_value = opt_ini_q
    exploration_threshold = 400
    epsilon_K = 0.00 #exploration rate


    #################################################

    #env = GridEnv(num_cols=3, num_rows=4, reward_matrix=[[-1,-1,5,-1],[0,0,0,0],[-1,-1,-1,100]], reward_mode="s'", render_mode="external")
    #env = GridEnv(num_rows=num_rows, num_cols=num_cols, reward_mode="s'", render_mode="external")


    #reward_targets = {major_r : [(num_cols - 2, num_rows // 2)],
    #                  minor_r : [#(2, num_rows // 2), 
    #                             ((num_cols - 1) // 3, num_rows // 2),
    #                             (2 * (num_cols - 1) // 3, num_rows // 2)]
    #                 }

    #reward_targets = { major_reward : [(num_cols - 2, num_rows - 2)],
    #                   minor_reward : [ ( int((num_cols-1)*p) , int((num_rows-1)*p) ) for p in minor_positions ] }

    env = GridEnv(num_rows=num_rows, num_cols=num_cols, 
                  reward_mode=reward_mode, 
                  default_reward=default_reward, 
                  reward_variance=reward_variance, 
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

    def repetition_finished_callback(sim):
        print("END REPETITION", sim.rep+1, '/', sim.num_repetitions)


    ###################################################

    #reward in the form "factored" + "s'"
    R = env.get_reward_matrix()
    print(R.shape)
    #reward in the form "flat" + "s'"
    R = R.reshape(env.observation_comb)
    print(R.shape)
    #reward in the form "flat" + "ass'"
    R = np.tile(R, (env.action_comb, env.observation_comb, 1))
    print(R.shape)

    #transition in the form "sas'" + "factored" + "deterministic"
    P = env.get_transition_matrix()
    print(P.shape)
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
          next_obs = P[act, obs]
          XP[act, obs, next_obs] = 1.0

    agent_PI_MDPTB = PolicyIteration_MDPtoolbox(env, R=R, P=XP, discount=gamma)
       
    ###################################################
    #reward in the form "factored" + "s'"
    R = env.get_reward_matrix()
    print(R.shape)
    #reward in the form "factored" + "sas'"
    R = np.tile( R, env.observation_shape+env.action_shape+tuple([1]*env.observation_ndim) )
    print(R.shape)

    #transition in the form "sas'" + "factored" + "deterministic"
    P = env.get_transition_matrix()
    print(P.shape)
    XP = np.zeros( env.observation_shape+env.action_shape+env.observation_shape , dtype=float)
    print(XP.shape)
    for obs in env.observation_iterator:
       for act in env.action_iterator:
          next_obs = ensure_tuple(P[obs+act])
          XP[obs+act+next_obs] = 1.0

    agent_PI = PolicyIteration(env=env, R=R, P=XP, discount=gamma, max_policy_iterations=50)

    agent_random = Agent(env=env, store_N_sa=True)

    agent_Q = ClassicQLearning(env=env,
                               store_N_sa=True,
                               discount=gamma,
                               learning_rate=alpha,
                               exploration_rate=epsilon_Q,
                               initial_Q_value=initial_Q_value)

    agent_ST_Q = SurvivalQLearning(env=env, 
                               store_N_sa=True,
                               discount=gamma,
                               learning_rate=alpha,
                               exploration_rate=epsilon_Q,
                               initial_Q_value=initial_Q_value,
                               survival_threshold=survival_threshold)

    agent_K = KLearning(env=env, 
                               store_N_sa=True,
                               discount=gamma,
                               learning_rate=alpha,
                               exploration_rate=epsilon_K,
                               initial_Q_value=initial_Q_value,
                               initial_K_value=initial_K_value,
                               survival_threshold=survival_threshold,
                               exploration_threshold=exploration_threshold)





    #sch = ConstantSchedule(1.0)
    #ann = MlpPolicy(env.observation_space, env.action_space, lr_schedule=sch)

    """
     
             observation_space (Space) – Observation space
             action_space (Discrete) – Action space
             lr_schedule (Callable[[float], float]) – Learning rate schedule (could be constant)
             net_arch (Optional[List[int]]) – The specification of the policy and value networks.
             activation_fn (Type[Module]) – Activation function
             features_extractor_class (Type[BaseFeaturesExtractor]) – Features extractor to use.
             features_extractor_kwargs (Optional[Dict[str, Any]]) – Keyword arguments to pass to the features extractor.
             normalize_images (bool) – Whether to normalize images or not, dividing by 255.0 (True by default)
             optimizer_class (Type[Optimizer]) – The optimizer to use, th.optim.Adam by default   
    """         
             
    """
        :param buffer_size: size of the replay buffer
        :param learning_starts: how many steps of the model to collect transitions for before learning starts
        :param batch_size: Minibatch size for each gradient update
        :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
        :param gamma: the discount factor
        :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
            like ``(5, "step")`` or ``(2, "episode")``.
        :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout.
        :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
            If ``None``, it will be automatically selected.
        :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
        :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
            at a cost of more complexity.
            See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        :param target_update_interval: update the target network every ``target_update_interval``
            environment steps.
        :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
        :param exploration_initial_eps: initial value of random action probability
        :param exploration_final_eps: final value of random action probability
        :param max_grad_norm: The maximum value for the gradient clipping
    """

    """
    # instantiate the untrained model to endow the agent
    model = DQN("MlpPolicy", env, 
                learning_rate=0.001, learning_starts=500, 
                buffer_size=2000, batch_size=32, 
                gamma=0.9, tau=1.0, train_freq=4, gradient_steps=1,
                target_update_interval=100,
                exploration_fraction=0.5, exploration_initial_eps=0.3, exploration_final_eps=0.1,
                verbose=1)

    print(model.policy)

    # Train the agent and display a progress bar
    #model.learn(total_timesteps=2000, progress_bar=True)

    agent_DQN_SB3 = SB3Policy(env, model, training_steps=20000)
    """

    #env = FlattenObservation(env)

    agent_DQN_SB3 = DQN_SB3(env, network_policy="MlpPolicy",
             store_N_sa=True,
             learning_rate=0.001,  #0.001, 
             learning_starts=50, #500
             buffer_size=600, #2000
             batch_size=80, 
             discount=0.95, 
             tau=0.5, 
             train_freq=4, 
             gradient_steps=1,
             target_update_interval=40, #100,
             exploration_initial_eps=0.3, exploration_final_eps=0.3, exploration_decay_fraction=0.5,
             verbose=1, train_on_reset=False, training_steps=horizon,
             store_Q=True, store_policy=True)


    #agent = agent_Q
    #agent = agent_ST_Q
    #agent = agent_K
    agent = [agent_PI, agent_Q, agent_ST_Q, agent_K]   # agent_random, agent_PI, agent_DQN_SB3, agent_PI_MDPTB, 
    #agent = [agent_DQN_SB3]

    #window = GridEnvRender(env, agent, cell_size=35)
    #env._render_frame = window.refresh


    sim = Sim(agent, env, episode_horizon=horizon, num_repetitions=num_repetitions, num_episodes=num_episodes)

    #sim.add_listener('round_started', round_started_callback)
    #sim.add_listener('round_finished', round_finished_callback)
    sim.add_listener('episode_started', episode_started_callback)
    sim.add_listener('episode_finished', episode_finished_callback)
    sim.add_listener('simulation_started', simulation_started_callback)
    sim.add_listener('simulation_finished', simulation_finished_callback)
    sim.add_listener('repetition_started', repetition_started_callback)
    sim.add_listener('repetition_finished', repetition_finished_callback)

    grid_elements=[
         #{'pos':0, 'label':'agent', 'data':[{'source':'env', 'attr':'R', 'type':"s'"}]},
          {'pos':1, 'label':'N_sa', 'source':'agent', 'attr':'N_sa', 'type':'sa', 'color_mode':'inversed_log_grayscale', 'backcolor':None},
         #{'pos':2, 'label':'V', 'source':'agent', 'attr':'V', 'type':'s', 'color_mode':'grayscale', 'backcolor':None},
          {'pos':2, 'label':'Q', 'source':'agent', 'attr':'Q', 'type':'sa', 'color_mode':'grayscale', 'backcolor':None},
          {'pos':3, 'label':'Q_target', 'source':'agent', 'attr':'Q_target', 'type':'sa', 'color_mode':'grayscale', 'backcolor':None},
          {'pos':4, 'label':'policy', 'source':'agent', 'attr':'policy', 'type':'sa', 'color_mode':'grayscale', 'backcolor':(0,0,0)},
        ]

    #gui = PyGameGUI(sim)
    gui = GridEnvGUI( sim, cell_size=20, fps=50, grid_elements=grid_elements, close_on_finish=False)

    gui.launch(give_first_step=True, start_running=True)

    #gui.close()


##################################################################################################

def main():
    'main entry point'

    # parse arguments
    parser = ArgumentParser(description='Run visual grid simulation')
    
    #parser.add_argument("-w", "--launch_gui", action='store_true', default=DEF_LAUNCH_GUI, help="Visual simulation (GUI)")
    parser.add_argument("-y", "--num-rows", type=int, default=DEF_ROWS, help="Number of grid rows.")
    parser.add_argument("-x", "--num-cols", type=int, default=DEF_COLS, help="Number of grid columns.")
    parser.add_argument("-b", "--initial_budget", type=float, default=DEF_INI_BUDGET, help="Initial budget.")
    parser.add_argument("-d", "--default-reward", type=float, default=DEF_R_MEAN, help="Default reward mean value.")
    parser.add_argument("-v", "--reward-variance", type=float, default=DEF_R_VAR, help="Default reward variance.")
    parser.add_argument("-m", "--reward-matrix", default=DEF_R_MATRIX, help="Reward matrix.")
    parser.add_argument("-o", "--reward-spots", type=dict, default=DEF_R_SPOTS, help="Reward spots.")
    parser.add_argument("-s", "--reward-spread", type=float, default=DEF_R_SPREAD, help="Reward spreading.")
    parser.add_argument("-t", "--horizon", type=int, default=DEF_HORIZON, help="Max time horizon.")
    parser.add_argument("-n", "--num-episodes", type=int, default=DEF_EPISODES, help="Number of episodes.")
    parser.add_argument("-r", "--num-repetitions", type=int, default=DEF_REPETITIONS, help="Number of repetitions.")
    parser.add_argument("-g", "--gamma", type=float, default=DEF_GAMMA, help="Discount factor.")
    parser.add_argument("-q", "--opt-ini-q", type=float, default=DEF_OPT_Q, help="Optimistic value for Q initialization.")
    parser.add_argument("-a", "--alpha", type=float, default=DEF_ALPHA, help="Learning rate.")
    parser.add_argument("-e", "--epsilon", type=float, default=DEF_EPSILON, help="Exploration rate.")
    #parser.add_argument("-i", "--max_policy_iterations", type=int, default=30, help="Maximum policy iterations.")
    #parser.add_argument("-j", "--max_value_iterations", type=int, default=100, help="Maximum value iterations within a policy iteration.")
    #parser.add_argument("-z", "--theta", type=float, default=0.001, help="Maximum value error to consider convergence.")
    #parser.add_argument("--minor_reward", type=float, default=DEF_MID_R, help="Small positive reward value.")
    #parser.add_argument("--major_reward", type=float, default=DEF_MAJ_R, help="Big positive reward value.")
    #parser.add_argument("--minor_positions", type=list, default=DEF_MID_R_POSITIONS, help="Relative positions into the diagonal.")

    args = parser.parse_args()
    
    run(**vars(args))
    
    return 0

##################################################################################################

if __name__ == '__main__':
    sys.exit(main())
