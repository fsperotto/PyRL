from collections.abc import Iterable
from pyrl import Agent, pyrl_space
from tensorforce import Agent as TensorforceAgent
import numpy as np
import random
import torch


class SurvivalA2CAgent(Agent):
    """
        Survival Advantage Actor Critic Agent
    """
    
    def __init__(
        self, env, observation_space, action_space, batch_size, initial_observation=None,
        initial_budget=1000, survival_threshold=250, exploration_threshold=500, max_episode_timesteps=None, network='auto', store_N=True,
        use_beta_distribution=False, memory='minimum', update_frequency=1.0, learning_rate=1e-3,
        horizon=1, gamma=0.99, reward_processing=None, return_processing=None,
        advantage_processing=None, predict_terminal_values=False, critic='auto', critic_optimizer=1.0,
        state_preprocessing='linear_normalization', exploration_rate=0.0, variable_noise=0.0,
        l2_regularization=0.0, entropy_regularization=0.0, parallel_interactions=1,
        config=None, saver=None, summarizer=None, tracking=None, recorder=None,
        # **kwargs
    ):

        #observations (what the agent perceives from the environment state)
        self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = pyrl_space(observation_space)        
        #actions
        self.action_space, self.action_shape, self.num_act_var, self.num_act_comb = pyrl_space(action_space)

        self.initial_observation = initial_observation
        self.agent = TensorforceAgent.create(
                        agent='a2c',
                        environment=env,
                        batch_size=batch_size,
                        exploration=exploration_rate,
                        max_episode_timesteps=max_episode_timesteps,
                        network=network,
                        use_beta_distribution=use_beta_distribution,
                        update_frequency=update_frequency,
                        learning_rate=learning_rate,
                        horizon=horizon,
                        discount=gamma,
                        reward_processing=reward_processing,
                        return_processing=return_processing,
                        advantage_processing=advantage_processing,
                        predict_terminal_values=predict_terminal_values,
                        critic=critic,
                        critic_optimizer=critic_optimizer,
                        state_preprocessing=state_preprocessing,
                        variable_noise=variable_noise,
                        l2_regularization=l2_regularization,
                        entropy_regularization=entropy_regularization,
                        parallel_interactions=parallel_interactions,
                        config=config,
                        saver=saver,
                        summarizer=summarizer,
                        tracking=tracking,
                        recorder=recorder,
                        memory=memory,
                        )
        
        self.episode_terminal = False
        self.initial_budget = initial_budget
        self.store_N = store_N
        self.truncated = None
        self.exploration_threshold = exploration_threshold
        self.survival_threshold = survival_threshold
        self.exploration_rate = exploration_rate
        self.gamma = gamma

        self.initial_K_value = 200

    
    def reset(self, s, reset_knowledge=True, reset_budget=True, learning=True):
        
        self.t = 0 # time, or number of elapsed rounds
        self.s = s  if isinstance(s, Iterable)  else  [s] # memory of the current state and last received reward
        self.s = s
        self.r = 0.0
        
        if reset_knowledge:
            self.agent.reset()
            self.Q = np.random.sample(self.observation_shape + self.action_shape)
            self.K = np.full(self.observation_shape + self.action_shape, self.initial_K_value)            
            
            if self.store_N:
               self.N = np.zeros(self.observation_shape + self.action_shape, dtype=int)
            
        self.b = self.initial_budget
        self.a = self.action_space.sample() # next chosen action
        self.recharge_mode = False
        
        
    def initial_internals(self):
        self.internals = self.agent.initial_internals()
        return self.internals
    
    def choose_action(self) -> list :
        states = self.s
        sample = random.random()
        
        if self.recharge_mode:
            # No Exploration, only Exploitation
            self.a = self.agent.act(states=states, deterministic=True)
            return self.a
        
        else:
            self.a = self.agent.act(states=states, deterministic=False)
            return self.a
        
    def observe(self, s, r, terminal=False, truncated=None):
        self.s = s
        self.r = r
        if self.r > 0: self.r = self.r * 1000
        self.truncated = truncated
        self.t += 1
        self.b += r
        
        if not self.recharge_mode and self.b < self.survival_threshold:
            self.recharge_mode = True
        
        if self.recharge_mode and self.b > self.exploration_threshold:
            self.recharge_mode = False
        
        self.agent.observe(self.r, terminal=terminal)
                
        self.N[self.s[0], self.s[1]] += 1
        
    def experience(self, states, actions, terminal, reward):
        self.agent.experience(states, actions, terminal, reward)

    def update(self):
        self.agent.update()
        
    def close(self):
        self.agent.close()