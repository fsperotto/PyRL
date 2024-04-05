# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:27:06 2023

@author: fperotto
"""


from typing import Iterable, Callable, TypeVar, Generic

import numpy as np
import math
#from gymnasium.spaces import Space

from pyrl import Agent, ensure_tuple, Env

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

#from stable_baselines3.common.utils import obs_as_tensor
import torch as th


###############################################################################

class DQN_SB3(Agent):
    """The agent wrapper for DQN class from stable_baselines_3 """

    #--------------------------------------------------------------    
    def __init__(self, env,
                 default_action=None,
                 network_policy="MlpPolicy",
                 discount=0.9, 
                 learning_rate=0.001, learning_starts=500,
                 buffer_size=2000, batch_size=32,
                 tau=1.0, train_freq=4, gradient_steps=1,
                 target_update_interval=100,
                 exploration_initial_eps=0.3, exploration_final_eps=0.1, exploration_decay_fraction=0.2,
                 should_explore=None, 
                 train_on_reset=False, training_steps=200000,
                 verbose=0,
                 remember_prev_a=False,
                 store_Q=False, store_policy=False,
                 store_N_sa=False, store_N_saz=False, store_N_z=False, store_N_a=False,
                 name="DQN_SB3"):

        super().__init__(env, 
                         default_action=default_action,
                         remember_prev_s=True, remember_prev_a=remember_prev_a,
                         store_N_sa=store_N_sa, store_N_saz=store_N_saz, store_N_z=store_N_z, store_N_a=store_N_a,
                         name=name)
        
        self.train_on_reset = train_on_reset

        self.network_policy = network_policy
        self.discount=discount
        self.learning_rate=learning_rate
        self.training_steps=training_steps
        self.learning_starts = learning_starts       
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.buffer_size=buffer_size
        self.tau=tau
        self.train_freq=train_freq 
        self.target_update_interval=target_update_interval
        self.exploration_initial_eps=exploration_initial_eps
        self.exploration_final_eps=exploration_final_eps
        self.exploration_decay_fraction=exploration_decay_fraction
        self.verbose=verbose
        
        # instantiate the untrained model to endow the agent
        self.model = None

        self.store_Q = store_Q
        if store_Q:
           # Q(s, a) table
           self.Q = None
           self.Q_target = None
           
        self.store_policy = store_policy
        if store_policy:
           #policy
           self.policy = None
           self.policy_target = None


    #--------------------------------------------------------------    
    def reset(self, initial_observation, *,
              reset_knowledge=True, learning_mode='off-policy',
              reset_budget=True, initial_budget=None):
        """
        Reset $t, r, s, a$, and can also reset the learned knowledge.

            Parameters:
                s (list): the initial state of the environment, observed by the agent
                reset_knowledge (bool) = True : if the agent should be completely reseted

            Returns:
                action : list representing the joint action chosen by the controller
        """
    
        if reset_knowledge:

           self.model = DQN( self.network_policy, self.env, 
               learning_rate=self.learning_rate, learning_starts=self.learning_starts, 
               buffer_size=self.buffer_size, batch_size=self.batch_size, 
               gamma=self.discount, tau=self.tau, train_freq=self.train_freq, gradient_steps=self.gradient_steps,
               target_update_interval=self.target_update_interval,
               exploration_initial_eps=self.exploration_initial_eps, exploration_final_eps=self.exploration_final_eps,
               exploration_fraction=self.exploration_decay_fraction, 
               verbose=self.verbose)
           
           if self.store_Q:
              self.Q = np.zeros( self.observation_shape+self.action_shape, dtype=float)
              self.Q_target = np.zeros( self.observation_shape+self.action_shape, dtype=float)
           if self.store_policy:
              self.policy=np.zeros(self.observation_shape+self.action_shape, dtype=float)
              self.policy_target=np.zeros(self.observation_shape+self.action_shape, dtype=float)
           
           #need to initialize the logger, the rest is not useful
           total_timesteps, callback = self.model._setup_learn(
               0, #total_timesteps,
               None, #callback,
               False, #reset_num_timesteps,
               "run", #tb_log_name,
               False #progress_bar,
           )
           
           if self.train_on_reset:
              self._plan()
           

        super().reset(initial_observation=initial_observation, 
                      reset_knowledge=reset_knowledge,
                      reset_budget=reset_budget, initial_budget=initial_budget,
                      learning_mode=learning_mode)


    #--------------------------------------------------------------    

    def _update_Q(self):
         
       if self.store_Q:
          for obs in self.observation_iterator:

             obs_th = self.model.q_net.obs_to_tensor(np.array(obs))[0]
             #dis = model.policy.get_distribution(obs_th)
             #probs = dis.distribution.probs
             #probs_np = probs.detach().numpy()
             self.Q[obs] = self.model.q_net(obs_th).numpy(force=True)

             obs_th = self.model.q_net_target.obs_to_tensor(np.array(obs))[0]
             self.Q_target[obs] = self.model.q_net_target(obs_th).numpy(force=True)

    #--------------------------------------------------------------    

    def _update_policy(self):

       if self.store_policy:
          #self.policy=np.zeros(self.observation_shape+self.action_shape, dtype=float)
          self.policy.fill(0.0)
          for obs in self.observation_iterator:
             action = self.model.predict(obs, deterministic=True)
             action = ensure_tuple(action)
             self.policy[obs + action] = 1.0
       
       
    #--------------------------------------------------------------    
    def _choose(self):
        
        if self.t < self.learning_starts:
           
           return super()._choose()
           
        else:
           
           #obs = self.get_state_tpl()
           obs = np.array([self.s])
   
           #action, _states = self.model.predict(obs, deterministic=True)
           action, _states = self.model.predict(obs, deterministic=False)  #exploration
           
           return action[0]
        


    #--------------------------------------------------------------    
    def _plan(self, total_timesteps=None, progress_bar=True) -> None :
       
       if total_timesteps is None:
          total_timesteps = self.training_steps
          
       # Train the agent and display a progress bar
       self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

       self._update_Q()
       self._update_policy()

    #--------------------------------------------------------------    
    def _learn(self) -> None:

         # Store data in replay buffer (normalized action and unnormalized observation)
         #self.model._store_transition(self.model.replay_buffer, [self.a], [self.s], [self.r], [self.terminated or self.truncated], [{}])
         self.model.replay_buffer.add(
            np.array([self.prev_s]), 
            np.array([self.s]), 
            np.array([self.a]), 
            np.array([self.r]),
            np.array([self.terminated or self.truncated]),
            np.array([{}])
         )

         if self.learning_starts is not None and self.t > self.learning_starts:

             # If no `gradient_steps` is specified,
             # do as many gradients steps as steps performed during the rollout
             #gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
             # Special case when the user passes `gradient_steps=0`
             if self.gradient_steps > 0:
                 self.model.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)

             self._update_Q()
             self._update_policy()

         self.model._on_step()
         
         
###############################################################################

class STDQN_SB3(DQN_SB3):
    """The agent STDQN class using stable_baselines_3 """

    #--------------------------------------------------------------    
    def __init__(self, env,
                 default_action=None,
                 survival_threshold=100, exploration_threshold=200, 
                 initial_K_value: float = 0,
                 network_policy="MlpPolicy",
                 discount=0.9, 
                 learning_rate=0.001, learning_starts=500,
                 buffer_size=2000, batch_size=32,
                 tau=1.0, train_freq=4, gradient_steps=1,
                 target_update_interval=100,
                 exploration_initial_eps=0.3, exploration_final_eps=0.1, exploration_decay_fraction=0.2,
                 should_explore=None, 
                 training_steps=200000,
                 verbose=0,
                 remember_prev_a=False,
                 store_Q=False, store_policy=False,
                 store_N_sa=False, store_N_saz=False, store_N_z=False, store_N_a=False,
                 name="STDQN_SB3"):

        super().__init__(env, 
                         default_action=default_action,
                         remember_prev_a=remember_prev_a,
                         network_policy=network_policy,
                         discount=discount, 
                         learning_rate=learning_rate, learning_starts=learning_starts,
                         buffer_size=buffer_size, batch_size=batch_size,
                         tau=tau, train_freq=train_freq, gradient_steps=gradient_steps,
                         target_update_interval=target_update_interval,
                         exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, exploration_decay_fraction=exploration_decay_fraction,
                         should_explore=should_explore, 
                         train_on_reset=False, training_steps=training_steps,
                         verbose=verbose,
                         store_Q=store_Q, store_policy=store_policy,
                         store_N_sa=store_N_sa, store_N_saz=store_N_saz, store_N_z=store_N_z, store_N_a=store_N_a,
                         name=name)
         
        # the untrained neutral model is already instantiated
        self.modelQ = self.model 
        # instantiate the untrained optimistic model to endow the agent
        self.modelK = DQN(network_policy, env, 
               learning_rate=learning_rate, learning_starts=learning_starts, 
               buffer_size=buffer_size, batch_size=batch_size, 
               gamma=discount, tau=tau, train_freq=train_freq, gradient_steps=gradient_steps,
               target_update_interval=target_update_interval,
               exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps,
               exploration_fraction=exploration_decay_fraction, 
               verbose=verbose)
        
        self.modelK.replay_buffer = self.modelQ.replay_buffer 

        self.initial_K_value = initial_K_value

        if exploration_threshold is None:
           self.exploration_threshold = survival_threshold
        else:
           self.exploration_threshold = exploration_threshold
        
        self.recharge_mode = False


        if self.store_Q:
           # K(s, a) table
           self.K = None
           
        if self.store_policy:
           #policy
           self.policyQ = self.policy
           self.policyK = None

    #--------------------------------------------------------------    
    def reset(self, initial_observation, *,
              reset_knowledge=True, learning_mode='off-policy',
              reset_budget=True, initial_budget=None):
        """
        Reset $t, r, s, a$, and can also reset the learned knowledge.

            Parameters:
                s (list): the initial state of the environment, observed by the agent
                reset_knowledge (bool) = True : if the agent should be completely reseted

            Returns:
                action : list representing the joint action chosen by the controller
        """
    
        if reset_knowledge:

           if self.store_Q:
              self.K = np.zeros( self.observation_shape+self.action_shape, dtype=float)
           if self.store_policy:
              self.policyK=np.zeros(self.observation_shape+self.action_shape, dtype=float)
           
           #need to initialize the logger, the rest is not useful
           total_timesteps, callback = self.modelK._setup_learn(
               0, #total_timesteps,
               None, #callback,
               False, #reset_num_timesteps,
               "run", #tb_log_name,
               False #progress_bar,
           )
           
           if self.train_on_reset:
              self._plan()
           

        super().reset(initial_observation=initial_observation, 
                      reset_knowledge=reset_knowledge,
                      reset_budget=reset_budget, initial_budget=initial_budget,
                      learning_mode=learning_mode)

        if self.store_policy:
           self.policyQ=self.policy

    #--------------------------------------------------------------    

    def _update_Q(self):
         
       super()._update_Q()

       if self.store_Q:
          for obs in self.observation_iterator:
             obs_th = self.modelK.q_net.obs_to_tensor(np.array(obs))[0]
             #dis = model.policy.get_distribution(obs_th)
             #probs = dis.distribution.probs
             #probs_np = probs.detach().numpy()
             self.K[obs] = self.modelK.q_net(obs_th).numpy(force=True)

    #--------------------------------------------------------------    

    def _update_policy(self):

       super()._update_policy()
       
       if self.store_policy:
          #self.policy=np.zeros(self.observation_shape+self.action_shape, dtype=float)
          self.policyK.fill(0.0)
          for obs in self.observation_iterator:
             action = self.modelK.predict(obs, deterministic=True)
             action = ensure_tuple(action)
             self.policyK[obs + action] = 1.0
       
         
    #--------------------------------------------------------------    
    def _choose(self):
        
        if self.t < self.learning_starts :
           #decision = "random"
           return self.action_space.sample()
        else :
           obs = np.array([self.s])
           if self.recharge_mode :  #and maxq > 0:
              #decision = "safe"
              action, _states = self.modelQ.predict(obs, deterministic=True)  #greedy
              return action[0]
           else:
              #decision = "normal"
              action, _states = self.modelK.predict(obs, deterministic=False)  #exploration
              return action[0]
    
         
    #--------------------------------------------------------------    
    def _learn(self) -> None:
       
         # Store data in replay buffer (normalized action and unnormalized observation)
         #self.model._store_transition(self.model.replay_buffer, [self.a], [self.s], [self.r], [self.terminated or self.truncated], [{}])
         self.model.replay_buffer.add(
            np.array([self.prev_s]), 
            np.array([self.s]), 
            np.array([self.a]), 
            np.array([self.r]),
            np.array([self.terminated or self.truncated]),
            np.array([{}])
         )

         if self.learning_starts is not None and self.t > self.learning_starts:
             if self.gradient_steps > 0:
                 self.modelQ.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)
                 self.modelK.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)
             self._update_Q()
             self._update_policy()

         self.modelQ._on_step()
         self.modelK._on_step()
