import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class PrioritizedReplayMemory:
    def __init__(self, capacity, prob_alpha=0.7, reward_priority_scale=1.0, epsilon=1e-6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = deque([], maxlen=capacity)
        self.pos        = 0
        self.priorities = deque([], maxlen=capacity)
        self.reward_priority_scale = reward_priority_scale
        self.epsilon = epsilon
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        
        max_prio = max(self.priorities) if self.buffer else 1.0        
        self.buffer.append((state, action, reward, next_state, done))                
        self.priorities.append(max_prio)
        self.pos = (self.pos + 1) % self.capacity
    
    
    def sample(self, batch_size, beta=0.4):        
        prios = list(self.priorities)
        probs  = [prio ** self.prob_alpha for prio in prios]
        probs = np.array(probs)
        probs = probs / probs.sum()
                
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights = (total * probs[indices])
        weights  = [weight ** (-beta) for weight in weights]
        weights = np.array(weights)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def clear(self):
        self.buffer.clear()
        self.pos        = 0
        self.priorities = deque([], maxlen=self.capacity)
        
    def __len__(self):
        return len(self.buffer)
