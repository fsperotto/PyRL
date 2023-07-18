import pyrl
from pyrl import Agent
from pyrl.replay_buffer.replay_buffer import ReplayMemory
from collections import namedtuple, deque
from tensorforce.agents import DeepQNetwork

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections.abc import Iterable

from tensorforce.agents import Agent as TFAgent
import numpy as np
import tensorflow as tf

class SurvivalDQNAgent(Agent):
    """
        Survival Deep Q-Network Agent.
    """
    
    def __init__(self, environment, policy, memory, batch_size, initial_observation=None,
                initial_budget=1000, survival_threshold=100, max_episode_timesteps=None,
                exploration_threshold=100, network='auto', update_frequency=0.25, optimizer='adamW',
                start_updating=None, learning_rate=0.001, huber_loss=None, horizon=1,
                discount=0.99, reward_processing=None, return_processing=None,
                predict_terminal_values=False, target_update_weight=1.0, target_sync_frequency=1,
                state_preprocessing='linear_normalization', exploration=0.5, variable_noise=0.0,
                l2_regularization=0.0, entropy_regularization=0.0, parallel_interactions=1,
                config=None, saver=None, summarizer=None,
                tracking=None, recorder=None, **kwargs 
                 ):
        

        self.exploration = exploration
        
        self.environment = environment

        self.s = initial_observation
        
        self.initial_budget = initial_budget
        self.b = self.initial_budget
        self.survival_threshold = survival_threshold
        self.exploration_threshold = exploration_threshold
    
        self.agent = TFAgent.create(
                agent='dqn',
                environment=environment,
                memory=memory,
                batch_size=batch_size,
                max_episode_timesteps=max_episode_timesteps,
                exploration=self.exploration,
                start_updating=start_updating,
                update_frequency=update_frequency,
                target_sync_frequency=target_sync_frequency,
                discount=discount
            )
        
        self.agent.policy = policy
        
    def reset(self, s, reset_knowledge=True):
        #time, or number of elapsed rounds 
        self.t = 0
        #memory of the current state and last received reward
        self.s = s  if isinstance(s, Iterable)  else  [s]
        self.r = 0.0   
        self.steps_done = 0 # used for epsilon decay over the timesteps and episodes
        self.b = self.initial_budget
        
        self.recharge_mode = False
        
        #next chosen action
        self.a = self.environment.action_space.sample()
        self.agent.reset()
        
        return self.a
        
    def act(self, states):
        # self.steps_done += 1
        if self.recharge_mode:
            # No Exploration, only Exploitation
            self.a = self.agent.act(states, deterministic=True)
        else:
            self.a = self.agent.act(states, deterministic=False)
            
        return self.a
 
    def observe(self, s, r, terminated=False, truncated=False):
        """
            Memorize the observed state and received reward.
        """
        self.s = s    
        self.r = r
        self.t += 1
        self.b += r
        
        if not self.recharge_mode and self.b < self.survival_threshold:
            self.recharge_mode = True
        
        if self.recharge_mode and self.b > self.exploration_threshold:
            self.recharge_mode = False
        
        self.agent.observe(self.r, terminal=terminated)
        
    def learn(self):
        pass
