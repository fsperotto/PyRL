from pyrl import Agent
import numpy as np
from collections.abc import Iterable
from tensorforce.agents import Agent as TFAgent
from pyrl import pyrl_space

class DQNAgent(Agent):
    """
        Deep Q-Network Agent.
    """
    
    def __init__(self, environment, memory, batch_size, initial_observation=None,
                initial_budget=1000, max_episode_timesteps=None, network='auto', update_frequency=0.25,
                start_updating=None, learning_rate=0.001, huber_loss=None, horizon=1,
                discount=0.99, reward_processing=None, return_processing=None,
                predict_terminal_values=False, target_update_weight=1.0, target_sync_frequency=1,
                state_preprocessing='linear_normalization', exploration=0.5, variable_noise=0.0,
                l2_regularization=0.0, entropy_regularization=0.0, parallel_interactions=1,
                config=None, saver=None, summarizer=None,
                tracking=None, recorder=None, **kwargs 
                 ):
        
        self.exploration= exploration
        self.agent = TFAgent.create(
                agent='dqn',
                environment=environment,
                memory=memory,
                network=network,
                batch_size=batch_size,
                max_episode_timesteps=max_episode_timesteps,
                exploration=self.exploration,
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
        self.environment = environment
        self.s = initial_observation
        self.initial_budget = initial_budget
        self.b = self.initial_budget
        
        #observations (what the agent perceives from the environment state)
        self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = pyrl_space(self.environment.observation_space)
        #actions
        self.action_space, self.action_shape, self.num_act_var, self.num_act_comb = pyrl_space(self.environment.action_space)
        
        self.initial_Q_value = 0
        
    def reset(self, s, reset_knowledge=True):
        self.t = 0 #time, or number of elapsed rounds 
        self.s = s  if isinstance(s, Iterable)  else  [s] #memory of the current state and last received reward
        self.r = 0.0   
        self.b = self.initial_budget
        self.a = self.environment.action_space.sample() #next chosen action
        self.N = np.zeros([self.environment.num_cols, self.environment.num_rows])
        self.N[self.s[0], self.s[1]] += 1
        if reset_knowledge:
            self.agent.reset()
        self.Q = np.full((self.environment.num_cols, self.environment.num_rows), self.initial_Q_value, dtype=float)
        self.Q[self.s[1], self.s[0]] = self.agent.tracked_tensors()['agent/policy/action-values'].max()
        
        return self.a
        
    def act(self, states):
        self.a = self.agent.act(states)
        
        return self.a

    def observe(self, s, r, terminated=False, truncated=False):
        """
            Memorize the observed state and received reward.
        """
        self.s = s
        self.r = r
        if self.r != -1: self.r = self.r * 1000
        self.t += 1
        self.b += r
        self.N[self.s[0], self.s[1]] += 1      
        self.agent.observe(self.r, terminal=terminated)
        self.Q[self.s[1], self.s[0]] = self.agent.tracked_tensors()['agent/policy/action-values'].max()
        
    def learn(self):
        pass
