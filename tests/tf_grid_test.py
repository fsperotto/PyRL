import numpy as np
from pyrl.agents import DQNAgent as ClassicDQN
from pyrl import Sim
from pyrl.environments.tf_grid import TFGridEnv, GridEnvRender
from tensorforce.environments import Environment

#################################################

num_rows=3
num_cols=30
minor_r = 5.0
major_r = 100.0
reward_targets = {major_r : [(num_cols - 2, num_rows // 2)],
                  minor_r : [(3*(num_cols - 1) // 5, num_rows // 2), ((num_cols - 1) // 3, num_rows // 2)]}
horizon = 2000
initial_budget = 400
repeat = 10
initial_Q_value = 0.0
gamma = 0.95 #discount factor
alpha = 0.5 #learning rate
survival_threshold = 250 
initial_K_value = 200
exploration_threshold = 500
points = 6
repeat = 3
initial_budgets = np.linspace(100, horizon, points, dtype=int)
batch_size = 256
replay_capacity = 6000
exploration_rate = 0.9
cell_size = 25
learning_rate = 0.3

def simulation_started_callback(sim, env, agent):
    print("START SIM")
    sim.metrics = dict(
        time = 0,
        exploration = np.zeros(np.prod(env.observation_space.nvec) + env.action_space.n),
        budget = np.zeros((sim.episode_horizon,), dtype=int)
    )

def simulation_finished_callback(sim, env, agent):
    print("END SIM")

def episode_started_callback(sim, env, agent):
    print("START EPISODE")

def episode_finished_callback(sim, env, agent):
    print("END EPISODE")

def round_started_callback(sim, env, agent):
    pass

def round_finished_callback(sim, env, agent):
    print("END ROUND")
    sim.metrics["time"] = sim.metrics["time"] + 1
    state_action_index = tuple(np.concatenate( (agent.get_state(), agent.get_action()) ) )
    state_action_index = tuple(agent.get_state_action())
    v = sim.metrics["exploration"].item(state_action_index)
    sim.metrics["exploration"].itemset(state_action_index, v+1)
    sim.metrics["budget"][sim.t-1] = agent.b        

###################################################

env = Environment.create(
    environment=TFGridEnv(num_rows=num_rows,
                        num_cols=num_cols, 
                        reward_mode="s'",
                        reward_targets=reward_targets,
                        default_reward=-1.0,
                        render_mode="external"
                        ),
    max_episode_timesteps=horizon,
)

agent_DQN = ClassicDQN(environment=env,
                       memory=replay_capacity,
                       batch_size=batch_size,
                       initial_budget=initial_budget,
                       exploration=exploration_rate,
                       discount=gamma,
                       max_episode_timesteps=horizon,
                       tracking="all",
                       learning_rate=learning_rate
                       )

window = GridEnvRender(env, agent_DQN, cell_size=cell_size)

env._render_frame = window.refresh

print("TEST CLASSIC DQN AGENT")

sim = Sim(agent_DQN,
          env,
          episode_horizon=horizon,
          num_simulations=repeat,
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