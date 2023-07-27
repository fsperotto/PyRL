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
                agent='dqn', environment=environment, memory=memory, batch_size=batch_size,
                max_episode_timesteps=max_episode_timesteps, exploration=self.exploration,
                start_updating=batch_size
                # discount=discount
            )
        self.environment = environment
        self.s = initial_observation
        self.initial_budget = initial_budget
        self.b = self.initial_budget
        
    def reset(self, s, reset_knowledge=True):
        #time, or number of elapsed rounds 
        self.t = 0
        #memory of the current state and last received reward
        self.s = s  if isinstance(s, Iterable)  else  [s]
        self.r = 0.0   
        # self.steps_done = 0 # used for epsilon decay over the timesteps and episodes
        self.b = self.initial_budget
        #next chosen action
        self.a = self.environment.action_space.sample()
        self.agent.reset()
        
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
        self.t += 1
        self.b += r
        # exploration = (1 / math.log(self.steps_done + 2))
        # self.agent.exploration = exploration        
        self.agent.observe(self.r, terminal=terminated)
    
    def learn(self):
        pass
