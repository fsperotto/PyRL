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

class SurvivalDQNAgent(Agent):
    """
        Survival Deep Q-Network Agent.
    """
    
    def __init__(self, environment, memory, batch_size, initial_observation=None,
                initial_budget=1000, threshold=100, max_episode_timesteps=None, network='auto', update_frequency=0.25,
                start_updating=None, learning_rate=0.001, huber_loss=None, horizon=1,
                discount=0.99, reward_processing=None, return_processing=None,
                predict_terminal_values=False, target_update_weight=1.0, target_sync_frequency=1,
                state_preprocessing='linear_normalization', exploration=0.0, variable_noise=0.0,
                l2_regularization=0.0, entropy_regularization=0.0, parallel_interactions=1,
                config=None, saver=None, summarizer=None,
                tracking=None, recorder=None, **kwargs 
                 ):
        
        
        # self.steps_done = 0 # used for epsilon decay over the timesteps and episodes
        
        
        # def custom_epsilon_decay():
        #     if self.steps_done == 0:
        #         epsilon = 0.0
        #     else:
        #         epsilon = (1 - (1 / math.log(self.steps_done + 2)))

        #     print(epsilon)
            
        #     return epsilon
        exploration = 0.7
        
        self.environment = environment

        self.s = initial_observation
        
        self.initial_budget = initial_budget
        self.b = self.initial_budget
        self.threshold = threshold
        
        # exploration = (1 / math.log(3))
        
        self.agent = TFAgent.create(
                agent='dqn', environment=environment, memory=memory, batch_size=batch_size,
                max_episode_timesteps=max_episode_timesteps,
                exploration=exploration
            )
        
        
        # self.agent.exploration = self.exploration_policy
        
    
        
    def reset(self, s, reset_knowledge=True):
        #time, or number of elapsed rounds 
        self.t = 0
        #memory of the current state and last received reward
        self.s = s  if isinstance(s, Iterable)  else  [s]
        self.r = 0.0   
        self.steps_done = 0 # used for epsilon decay over the timesteps and episodes
        self.b = self.initial_budget
        #next chosen action
        self.a = self.environment.action_space.sample()
        self.agent.reset()
        
        return self.a
        
    def act(self, states):
        # sample = random.random()
        # # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        # eps_threshold = (1 - (1 / math.log(self.steps_done + 2)))
        self.steps_done += 1
        
        if self.b <= self.threshold:
            self.a = self.agent.act(states, deterministic=True)
        else:
            # self.agent.exploration = (1 - (1 / math.log(self.steps_done + 2)))
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

        # if self.b < self.threshold:
        #     exploration = 0.0
        # else:
        #     exploration = 0.5
            
        # self.agent.exploration = exploration        
        self.agent.observe(self.r, terminal=terminated)
    
    # def exploration_policy(self, agent, states):
    #     epsilon = (1 - (1 / math.log(self.steps_done + 2)))

    #     if np.random.rand() < epsilon:
    #         return np.random.randint(self.environment.action_space)
    #     else:
    #         return agent.act(states)
    
    def custom_epsilon_decay(self):
        epsilon = (1 - (1 / math.log(self.steps_done + 2)))
        
        return epsilon
    
    
    def learn(self):
        pass
