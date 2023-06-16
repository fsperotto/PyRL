import pyrl
from pyrl import Agent
from pyrl.replay_buffer.replay_buffer import ReplayMemory
from collections import namedtuple, deque

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections.abc import Iterable

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent(Agent):
    """
        Deep Q-Network Agent.

    """
    
    def __init__(self, observation_space, action_space, initial_observation=None,
                 eps_start=0.9, eps_end=0.05, eps_decay=1000,
                 replay_buffer=ReplayMemory(1000), batch_size=128, gamma=0.99,
                 tau=0.005, 
                 ):
        super().__init__(observation_space, action_space, initial_observation=None)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print((self.observation_space.shape[0], self.action_space.n))
        self.policy_net = DQN(self.observation_space.shape[0], self.action_space.n).to(self.device)
        self.target_net = DQN(self.observation_space.shape[0], self.action_space.n).to(self.device)
        
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = 1e-4
        self.tau = tau
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.steps_done = 0 # used for epsilon decay over the timesteps and episodes
        
    def reset(self, s, reset_knowledge=True):
        #time, or number of elapsed rounds 
        self.t = 0
        #memory of the current state and last received reward
        # self.s = s  if isinstance(s, Iterable)  else  [s]
        self.s = s
        # self.s = torch.tensor(s  if isinstance(s, Iterable)  else  [s], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.r = 0.0
        #next chosen action
        #self.a = [None for _ in range(self.num_action_vars)] 
        self.a = self.action_space.sample()
        
    def act(self):
        # global steps_done

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                self.a = self.policy_net(self.s).max(1)[1].view(1, 1)
                return self.a
        else:
            self.a = torch.tensor([[self.action_space.sample()]], device=self.device, dtype=torch.long)
            return self.a

    def observe(self, s, r):
        """
            Memorize the observed state and received reward.
        """
        # self.prev_s = self.s
        # self.s = s  if isinstance(s, Iterable)  else  [s]
        self.s = s
        
        self.r = r
        
        # Store the transition in memory
        # self.replay_buffer.push(self.prev_s, self.a, self.s, self.r)
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # print(batch.next_state)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
       
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)