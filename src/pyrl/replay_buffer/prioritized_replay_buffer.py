import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class PrioritizedReplayMemory:
    def __init__(self, capacity, prob_alpha=0.7, reward_priority_scale=1.0, epsilon=1e-6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.reward_priority_scale = reward_priority_scale
        self.epsilon = epsilon
    
    def push(self, state, action, next_state, reward, done):
        assert state.ndim == next_state.ndim
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, next_state, reward, done))
        else:
            self.buffer[self.pos] = (Transition(state, action, next_state, reward, done))
        
        # Set priority to be the absolute value of reward scaled by reward_priority_scale
        self.priorities[self.pos] = abs(reward * self.reward_priority_scale) + self.epsilon
        self.pos = (self.pos + 1) % self.capacity
    
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs = probs / probs.sum()
                
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def clear(self):
        self.buffer.clear()
        self.pos        = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        
    def __len__(self):
        return len(self.buffer)
