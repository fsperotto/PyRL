from typing import Callable
import numpy as np
#from gymnasium.spaces import Space

from pyrl.agents.survival import QLearning

class KLearning(QLearning):

    #--------------------------------------------------------------    
    def __init__(self, env, *, 
                 default_action=None,
                 discount=0.9, learning_rate=0.1, 
                 exploration_rate=None, should_explore: Callable = None, 
                 initial_Q: np.ndarray = None, initial_Q_value: float = None, 
                 store_V = False, store_policy = True,
                 survival_threshold=10, exploration_threshold=None, 
                 initial_K_value: float = 0,
                 name:str="K-Learning",
                 remember_prev_a=False,
                 store_N_sa=False, store_N_saz=False, store_N_z=False, store_N_a=False
                ):
        
        super().__init__(env=env, 
                         default_action=default_action,
                         discount=discount, learning_rate=learning_rate,
                         exploration_rate=exploration_rate,
                         should_explore=should_explore,
                         survival_threshold=survival_threshold,
                         initial_Q=initial_Q, initial_Q_value=initial_Q_value,
                         remember_prev_a=remember_prev_a,
                         store_N_sa=store_N_sa, store_N_saz=store_N_saz, store_N_z=store_N_z, store_N_a=store_N_a, 
                         store_V=store_V, store_policy=store_policy,
                         name=name)
       
        self.initial_K_value = initial_K_value

        if exploration_threshold is None:
           self.exploration_threshold = survival_threshold
        else:
           self.exploration_threshold = exploration_threshold
        
        self.K = None
        self.recharge_mode = False

    #--------------------------------------------------------------    
    def _choose(self):
        
        #if self.recharge_mode and maxq > 0:
        if self.recharge_mode :
        
            maxq = self.Q[self.get_state_tpl()].max()
            a = np.random.choice(np.flatnonzero(self.Q[self.get_state_tpl()] == maxq))
        
        else:
        
            if self.should_explore(self):

               a = self.action_space.sample()
               
            else:
           
               maxk = self.K[self.get_state_tpl()].max()
               #maxk = self.K[self.s].max()
               a = np.random.choice(np.flatnonzero(self.K[self.get_state_tpl()] == maxk))
               #a = np.random.choice(np.flatnonzero(self.K[self.s] == maxk))
        
        return a
    
    #--------------------------------------------------------------    
    def reset(self, initial_observation, *, initial_budget=None, reset_budget=True, 
              reset_knowledge=True, learning_mode='off-policy'):
       
        if reset_knowledge:
           self.K = np.full(self.observation_shape + self.action_shape, self.initial_K_value)
        self.recharge_mode = False

        super().reset(initial_observation=initial_observation, 
                      reset_knowledge=reset_knowledge,
                      initial_budget=initial_budget,
                      reset_budget=reset_budget,
                      learning_mode=learning_mode)
    
    #--------------------------------------------------------------    
    def _observe(self, state: int, reward: float, terminated: bool = False, truncated: bool = False) -> None:
        
        super()._observe(state, reward, terminated, truncated)

        if not self.recharge_mode and self.b <= self.survival_threshold:
            self.recharge_mode = True
        
        if self.recharge_mode and self.b > self.exploration_threshold:
            self.recharge_mode = False

    #--------------------------------------------------------------    
    def _learn(self):
        super()._learn()
        #self.K[self.get_state_tpl(self.prev_s) + self.get_action()] = (1 - self.learning_rate) * self.K[self.last_state, self.last_action] + self.learning_rate * (self.current_reward + self.discount * self.K[self.current_state, :].max())
        index_sa = self.get_state_tpl(self.prev_s) + self.get_action_tpl()
        index_next_s = self.get_state_tpl()
        K_sa = self.K[index_sa]
        K_next_s = self.K[index_next_s]
        new_k = (1 - self.learning_rate) * K_sa + self.learning_rate * (self.r + self.discount * K_next_s.max())
        self.K[index_sa] = new_k
        return new_k
        