# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:16:05 2024

@author: fperotto
"""

from collections.abc import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnvMAB(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, arms, h=2000, r_min=0.0, r_max=1.0, b_0=20.0, ruinable=True, prev_draw=True, seed=None):
        
        super().__init__()
        
        self.rnd_gen = np.random.default_rng(seed=seed)
        
        #a list of arms from pyrl.arms
        # A : arms (1 ... i ... k)   - i or idx_arm
        self.arms = arms if isinstance(arms, Iterable) else [arms]
        
        #domain of rewards ( by default on [0, 1] )
        if r_min > r_max:
            r_min, r_max = r_max, r_min
        self.r_min = r_min
        self.r_max = r_max
        self.r_amp = r_max - r_min

        #time-horizon (0, 1 ... t ... h)
        #max number of rounds 
        self.h = h
        self.T = range(self.h)          #range for time (0 ... h-1)
        self.T1 = range(1, self.h+1)    #range for time (1 ... h)
        self.T01 = range(0, self.h+1)   #range for time (0, 1 ... h)
        
        #number of arms
        self.k = len(self.arms)
        self.K = range(self.k)          #range for arms (0 ... k-1)
        self.K1 = range(1,self.k+1)     #range for arms (1 ... k)

        #arms properties
        self.mu_i = np.array([a.mean for a in self.arms]) # * self.d.r_amp + self.d.r_min     #means
        self.i_star = np.argmax(self.mu_i)                #best arm index
        self.i_worst = np.argmin(self.mu_i)               #worst arm index
        self.mu_star = np.max(self.mu_i)                  #best mean
        self.mu_worst = np.min(self.mu_i)                 #worst mean

        #budget
        self.b_0 = b_0   
        self.ruinable = ruinable
        self.b = None

        #in order to optimize running, all the arms random elements can be drawn at the beginning
        self.prev_draw = prev_draw
        self.drawn_reward_i_t = None

        # Define action and observation space, that must be gym.spaces objects
        # each arm is a discrete action:
        self.action_space = spaces.Discrete(len(arms))
        
        # in mab, there is no observation, or the state is unique
        self.observation_space = spaces.Discrete(1)
        
        self.r = None   #last received reward
        
        self.terminated = False
        self.truncated = False
        
        self.ruined = False
        
        self.t = -1


    def reset(self, seed=None):
        
        #initial round
        self.t = 0

        self.b = self.b_0

        self.terminated = False
        self.truncated = False
        
        self.ruined = self.ruinable and self.b <= 0.0
        
        if seed is not None:
            self.rnd_gen = np.random.default_rng(seed=seed)
            
        if self.prev_draw:
            seed_t = self.rnd_gen.random(self.h)     #luck is the same for every arm in a same round
            self.drawn_reward_i_t = np.array([arm.convert(chances_arr=seed_t) for arm in self.arms]) #seed to reward
            
        #return observation, info
        return 0, None
    
    
    def step(self, action):
        
        #next round
        self.t += 1
        
        #no choice, no action
        if action == -1:
            self.r = 0.0
        else:
            # The arm played gives reward
            if self.drawn_reward_i_t is not None:
                self.r = self.drawn_reward_i_t[action, self.t]
            else:
                self.r = self.arms[action].draw()
        
        #update budget
        self.b += self.r
        self.ruined = self.ruinable and self.b <= 0.0

        #termination conditions        
        self.terminated = self.ruined or self.t >= self.h
        
        #return observation, reward, terminated, truncated, info
        return 0, self.r, self.terminated, self.truncated, None



    def render(self):
        pass


    def close(self):
        pass