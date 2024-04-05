from pyrl import Agent
import numpy as np
from collections.abc import Iterable
from pyrl import pyrl_space
from stable_baselines3 import DQN
from stable_baselines3.common.logger import Logger

class DQNAgent(Agent):
    """
        Stable Baselines3 Deep Q-Network Agent.
    """
    
    def __init__(self, env, initial_observation=None, initial_budget=1000, store_N=True, policy="MlpPolicy", learning_rate=0.0001, buffer_size=1000, 
                 learning_starts=64, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, 
                 gradient_steps=1, replay_buffer_class=None, replay_buffer_kwargs=None, 
                 optimize_memory_usage=False, target_update_interval=100, exploration_fraction=0.1, 
                 exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10, 
                 stats_window_size=100, tensorboard_log=None, policy_kwargs=None, verbose=0, 
                 seed=None, device='auto', _init_setup_model=True
                 ):
        
        self.agent = DQN(env=env,
                        policy=policy,
                        learning_rate=learning_rate,
                        buffer_size=buffer_size,
                        learning_starts=learning_starts,
                        batch_size=batch_size,
                        tau=tau,
                        gamma=gamma,
                        train_freq=train_freq,
                        gradient_steps=gradient_steps,
                        replay_buffer_class=replay_buffer_class,
                        replay_buffer_kwargs=replay_buffer_kwargs,
                        optimize_memory_usage=optimize_memory_usage,
                        target_update_interval=target_update_interval,
                        exploration_fraction=exploration_fraction,
                        exploration_initial_eps=exploration_initial_eps,
                        exploration_final_eps=exploration_final_eps,
                        max_grad_norm=max_grad_norm,
                        stats_window_size=stats_window_size,
                        tensorboard_log=tensorboard_log,
                        policy_kwargs=policy_kwargs,
                        verbose=verbose,
                        seed=seed,
                        device=device,
                        _init_setup_model=_init_setup_model
                        )
        self.initial_agent = self.agent
        
        self.environment = env
        self.s = initial_observation
        self.initial_budget = initial_budget
        self.b = self.initial_budget
        
        self.batch_size = batch_size
        
        #observations (what the agent perceives from the environment state)
        self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb, self.obs_idx_factors = pyrl_space(self.environment.observation_space)
        #actions
        self.action_space, self.action_shape, self.num_act_var, self.num_act_comb, self.act_idx_factors = pyrl_space(self.environment.action_space)
        
        self.initial_Q_value = 0
        self.store_N = store_N
        
        
    def reset(self, s, reset_knowledge=True):
        self.t = 0 #time, or number of elapsed rounds 
        self.s = s  if isinstance(s, Iterable)  else  [s] #memory of the current state and last received reward
        self.s = s
        self.r = 0.0   
        self.b = self.initial_budget
        self.a = self.environment.action_space.sample() #next chosen action
        logger = Logger("logs/", output_formats='stdout')

        # Set the logger for the model
        self.agent.set_logger(logger)
        if reset_knowledge:
            self.agent = self.initial_agent
            self.Q = np.random.sample(self.observation_shape + self.action_shape)
            if self.store_N:
               self.N = np.zeros(self.observation_shape + self.action_shape, dtype=int)
        
        return self.a
        
    def choose_action(self):
        states = self.s
        self.a, _states = self.agent.predict(observation=states)
        
        return self.a.item()

    def observe(self, s, r, terminated=False, truncated=False):
        """
            Memorize the observed state and received reward.
        """
        prev_s = self.s
        self.s = s  if isinstance(s, Iterable)  else  [s]
        self.r = r
        self.t += 1
        self.b += r
        self.agent.replay_buffer.add(obs=prev_s, next_obs=self.s, action=self.a, reward=self.r, done=terminated, infos=[{}])
        print(prev_s, self.s)
        self.N[self.s[0], self.s[1]] += 1
        
    def learn(self):
        self.agent.train(gradient_steps=1)
        
