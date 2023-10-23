import gymnasium as gym

from pyrl import Sim, Agent, EnvWrapper #, System

#################################################

def start_episode_callback(sim):
   print("START EPISODE", sim.ep+1, '/', sim.num_episodes)

def end_episode_callback(sim):
   print(sim.t, 'rounds')
   
#################################################

horizon = 20
num_episodes = 1


envs_props = [
   
         {'name':'CliffWalking-v0', 'render_mode':'human', 'horizon':10},
         {'name':'FrozenLake-v1', 'render_mode':'human', 'horizon':horizon},
         {'name':'Taxi-v3', 'render_mode':'human', 'horizon':10},
         #{'name':'Blackjack-v1', 'render_mode':'human', 'horizon':10},
         {'name':'Acrobot-v1', 'render_mode':'human', 'horizon':horizon},
         {'name':'MountainCar-v0', 'render_mode':'human', 'horizon':horizon},
         {'name':'CartPole-v1', 'render_mode':'human', 'horizon':horizon},
         {'name':'Pendulum-v1', 'render_mode':'human', 'horizon':horizon},
         {'name':'BipedalWalker-v3', 'render_mode':'human', 'horizon':horizon},
         {'name':'CarRacing-v2', 'render_mode':'human', 'horizon':horizon},
         {'name':'LunarLander-v2', 'render_mode':'human', 'horizon':horizon},
         {'name':'SpaceInvaders-v4', 'render_mode':'human', 'horizon':horizon},
         {'name':'Enduro-v4', 'render_mode':'human', 'horizon':horizon},
         {'name':'ALE/Atlantis-v5', 'render_mode':'human', 'horizon':horizon},
         {'name':'ALE/Tetris-v5', 'render_mode':'human', 'horizon':horizon},
         {'name':'ALE/Enduro-v5', 'render_mode':'human', 'horizon':horizon},
         {'name':'ALE/SpaceInvaders-v5', 'render_mode':'human', 'horizon':horizon},
                     
   ]

#################################################

for env_props in envs_props:
   print('***', env_props['name'])
   #env = EnvWrapper(gym.make(env_props['name'], render_mode=env_props['render_mode']))
   env = EnvWrapper(gym.make(env_props['name'], render_mode=env_props['render_mode']))
   agent = Agent(env)
   s, info = env.reset()
   a = agent.reset(s)
   for t in range(10):
      s, r, terminated, truncated, info = env.step(a)
      a = agent.step(s, r, terminated, truncated)
   env.close()
   #env.env = None
   #env = None

#################################################

   
for env_props in envs_props:
   
   print('***', env_props['name'])
   
   env = EnvWrapper(gym.make(env_props['name'], render_mode=env_props['render_mode']))

   #agent = Agent(observation_space = env.observation_space, action_space = env.action_space)
   agent = Agent(env)

   sim = Sim( agent, env, num_episodes=num_episodes, episode_horizon=env_props['horizon'])

   sim.add_listener('episode_started', start_episode_callback )
   sim.add_listener('episode_finished', end_episode_callback )

   sim.run()
   

#################################################

