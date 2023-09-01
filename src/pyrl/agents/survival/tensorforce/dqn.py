from pyrl import Agent
import numpy as np
from collections.abc import Iterable
from tensorforce.agents import Agent as TFAgent
from pyrl import pyrl_space

class SurvivalDQNAgent(Agent):
    """
        Survival Deep Q-Network Agent.
    """
    
    def __init__(self, env, observation_space, action_space, memory, batch_size, initial_observation=None,
                initial_budget=1000, store_N=True, exploration_threshold=500, survival_threshold=250, max_episode_timesteps=None, network='auto', update_frequency=0.25,
                start_updating=None, learning_rate=0.001, huber_loss=None, horizon=1,
                discount=0.99, reward_processing=None, return_processing=None,
                predict_terminal_values=False, target_update_weight=1.0, target_sync_frequency=1,
                state_preprocessing='linear_normalization', exploration_rate=0.5, variable_noise=0.0,
                l2_regularization=0.0, entropy_regularization=0.0, parallel_interactions=1,
                config=None, saver=None, summarizer=None,
                tracking=None, recorder=None, **kwargs 
                 ):
        
        #observations (what the agent perceives from the environment state)
        self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = pyrl_space(observation_space)        
        #actions
        self.action_space, self.action_shape, self.num_act_var, self.num_act_comb = pyrl_space(action_space)
        
        self.exploration_rate = exploration_rate
        self.agent = TFAgent.create(
                agent='dqn',
                environment=env,
                memory=memory,
                network=network,
                batch_size=batch_size,
                max_episode_timesteps=max_episode_timesteps,
                exploration=self.exploration_rate,
                update_frequency=update_frequency,
                start_updating=start_updating,
                learning_rate=learning_rate,
                huber_loss=huber_loss,
                horizon=horizon,
                discount=discount,
                reward_processing=reward_processing,
                return_processing=return_processing,
                predict_terminal_values=predict_terminal_values,
                target_update_weight=target_update_weight,
                target_sync_frequency=target_sync_frequency,
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
            )

        self.s = initial_observation
        self.initial_budget = initial_budget
        self.b = self.initial_budget
        
        self.exploration_threshold = exploration_threshold
        self.survival_threshold = survival_threshold
        
        self.initial_Q_value = 0
        self.store_N = store_N
        self.truncated = None
        
    def reset(self, s, reset_knowledge=True, reset_budget=True, learning=True):
        self.t = 0 #time, or number of elapsed rounds 
        self.s = s  if isinstance(s, Iterable)  else  [s] #memory of the current state and last received reward
        self.r = 0.0   
        self.b = self.initial_budget
        self.a = self.action_space.sample() #next chosen action

        if reset_knowledge:
            self.agent.reset()
            self.Q = np.random.sample(self.observation_shape + self.action_shape)
            if self.store_N:
                self.N = np.zeros(self.observation_shape + self.action_shape, dtype=int)

        self.recharge_mode = False
        
    def choose_action(self):
        states = self.s
        if self.recharge_mode:
            # No Exploration, only Exploitation
            self.a = self.agent.act(states, deterministic=True)
        else:
            self.a = self.agent.act(states, deterministic=False)

        return self.a

    def observe(self, s, r, terminal=False, truncated=False):
        """
            Memorize the observed state and received reward.
        """
        self.s = s
        self.r = r
        
        if self.r > 0 : self.reward_scale = 1
        else: self.reward_scale = 1
        
        self.truncated = truncated
        self.t += 1
        self.b += r
        self.agent.observe(self.r, terminal=terminal)
        self.N[self.s[0], self.s[1]] += 1

        if not self.recharge_mode and self.b < self.survival_threshold:
            self.recharge_mode = True
        
        if self.recharge_mode and self.b > self.exploration_threshold:
            self.recharge_mode = False
        
        
    def learn(self):
        pass
