"""
A FAIRE DANS PYRL:
   
R and P MATRICES AS CLASSES
with methods to convert between formats

MAYBE IT IS EQUIVALENT TO AN MDP CLASS...
in this case, we can include methods to solve it from MODEL

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
major_r = 50.0

#SIMULATION PARAMETERS

horizon = 800

initial_budget = 300

repeat = 2

episodes = 1


#################################################

#env = GridEnv(num_cols=3, num_rows=4, reward_matrix=[[-1,-1,5,-1],[0,0,0,0],[-1,-1,-1,100]], reward_mode="s'", render_mode="external")
#env = GridEnv(num_rows=num_rows, num_cols=num_cols, reward_mode="s'", render_mode="external")

reward_targets = {major_r : [(num_cols - 2, num_rows // 2)],
                  minor_r : [(2, num_rows // 2), ((num_cols - 1) // 2, num_rows // 2)]}

env = GridEnv(num_rows=num_rows, num_cols=num_cols, 
              reward_mode="s'", reward_targets=reward_targets, default_reward=-1.0, 
              #reward_spread=0.3,
              default_initial_budget=initial_budget,
              render_mode="external")


#################################################

RUN_VISUAL_SIM = True

if RUN_VISUAL_SIM:
   
   def repetition_started_callback(sim):
       print("START REPETITION", sim.rep+1, '/', sim.num_repetitions)
       print('observation shape:', env.observation_shape)
       print('action shape:', env.action_shape)
       print('observation+action shape:', env.observation_shape + env.action_shape)
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
       #sim.metrics["budget"][sim.t-1] = sim.agent.b        
   
   ###################################################
   
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
   
   agent_DQN_SB3 = DQN_SB3(env, network_policy="MlpPolicy", 
            learning_rate=0.001,  #0.001, 
            learning_starts=100, #500
            buffer_size=1000, #2000
            batch_size=32, 
            discount=0.9, 
            tau=1.0, 
            train_freq=4, 
            gradient_steps=1,
            target_update_interval=100,
            exploration_initial_eps=0.3, exploration_final_eps=0.1, exploration_decay_fraction=0.5,
            verbose=1, train_on_reset=False, training_steps=horizon)
   
   agent = [agent_DQN_SB3] 
   
   #window = GridEnvRender(env, agent, cell_size=35)
   #env._render_frame = window.refresh
   
   
   sim = Sim(agent, env, episode_horizon=horizon, num_repetitions=repeat, num_episodes=episodes)
   
   #sim.add_listener('round_started', round_started_callback)
   #sim.add_listener('round_finished', round_finished_callback)
   sim.add_listener('episode_started', episode_started_callback)
   sim.add_listener('episode_finished', episode_finished_callback)
   sim.add_listener('simulation_started', simulation_started_callback)
   sim.add_listener('simulation_finished', simulation_finished_callback)
   sim.add_listener('repetition_started', repetition_started_callback)
   sim.add_listener('repetition_finished', repetition_finished_callback)
   
   
   #gui = PyGameGUI(sim)
   gui = GridEnvGUI(sim, cell_size=25, fps=50, close_on_finish=False, 
                    grid_elements=[
                       #{'pos':0, 'label':'agent', 'data':[{'source':'env', 'attr':'R', 'type':"s'"}]},
                       {'pos':1, 'label':'N', 'source':'agent', 'attr':'N', 'type':'sa', 'color_mode':'inversed_log_grayscale', 'backcolor':None},
                       #{'pos':2, 'label':'V', 'source':'agent', 'attr':'V', 'type':'s', 'color_mode':'grayscale', 'backcolor':None},
                       {'pos':2, 'label':'Q', 'source':'agent', 'attr':'Q', 'type':'sa', 'color_mode':'grayscale', 'backcolor':None},
                       {'pos':3, 'label':'Q_target', 'source':'agent', 'attr':'Q_target', 'type':'sa', 'color_mode':'grayscale', 'backcolor':None},
                       {'pos':4, 'label':'policy', 'source':'agent', 'attr':'policy', 'type':'sa', 'color_mode':'grayscale', 'backcolor':(0,0,0)},
                    ],
                   )
   
   gui.launch(give_first_step=True, start_running=True)
   
   #gui.close()


###################################################




"""
# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    print(obs, action, end=' ')
    obs, rewards, dones, info = vec_env.step(action)
    print(rewards)
    #vec_env.render("human")
"""