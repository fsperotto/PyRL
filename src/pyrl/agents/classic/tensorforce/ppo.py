from collections.abc import Iterable
from pyrl import Agent, pyrl_space
from tensorforce import Agent as TensorforceAgent
import numpy as np


class PPOAgent(Agent):
    """
        Tensorforce     Proximal Policy Optimization Agent
    """
    
    def __init__(
        self, env, observation_space, action_space, batch_size, initial_observation=None,
        initial_budget=1000, max_episode_timesteps=None,  network='auto', store_N=True,
        use_beta_distribution=False, memory='minimum', update_frequency=1.0, learning_rate=0.001, 
        multi_step=10, subsampling_fraction=0.33, likelihood_ratio_clipping=0.25, gamma=0.99, 
        reward_processing=None, return_processing=None, advantage_processing=None, predict_terminal_values=False, 
        baseline=None, baseline_optimizer=None, state_preprocessing='linear_normalization', exploration_rate=None, 
        variable_noise=0.0, l2_regularization=0.0, entropy_regularization=0.0, parallel_interactions=1, 
        config=None, saver=None, summarizer=None, tracking=None, recorder=None, **kwargs):

        #observations (what the agent perceives from the environment state)
        self.observation_space, self.observation_shape, self.num_obs_var, self.num_obs_comb = pyrl_space(observation_space)        
        #actions
        self.action_space, self.action_shape, self.num_act_var, self.num_act_comb = pyrl_space(action_space)

        self.initial_observation = initial_observation
        self.agent = TensorforceAgent.create(
                        agent='ppo',
                        environment=env,
                        batch_size=batch_size,
                        exploration=exploration_rate,
                        max_episode_timesteps=max_episode_timesteps,
                        network=network,
                        use_beta_distribution=use_beta_distribution,
                        memory=memory,
                        update_frequency=update_frequency,
                        learning_rate=learning_rate,
                        multi_step=multi_step,
                        subsampling_fraction=subsampling_fraction,
                        likelihood_ratio_clipping=likelihood_ratio_clipping,
                        discount=gamma,
                        reward_processing=reward_processing,
                        return_processing=return_processing,
                        advantage_processing=advantage_processing,
                        predict_terminal_values=predict_terminal_values,
                        baseline=baseline,
                        baseline_optimizer=baseline_optimizer, 
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
                        **kwargs
                        )
                    
        
        self.episode_terminal = False
        self.initial_budget = initial_budget
        self.store_N = store_N
        self.truncated = None

    
    def reset(self, s, reset_knowledge=True, reset_budget=True, learning=True):
        
        self.t = 0 # time, or number of elapsed rounds
        self.s = s  if isinstance(s, Iterable)  else  [s] # memory of the current state and last received reward
        self.s = s
        self.r = 0.0
        
        if reset_knowledge:
            self.agent.reset()
            self.Q = np.random.sample(self.observation_shape + self.action_shape)
                
            if self.store_N:
               self.N = np.zeros(self.observation_shape + self.action_shape, dtype=int)
            
        self.b = self.initial_budget
        self.a = self.action_space.sample() # next chosen action
        
        
    def initial_internals(self):
        self.internals = self.agent.initial_internals()
        return self.internals
    
    def choose_action(self) -> list :
        states = self.s
        self.a = self.agent.act(states=states)
        
        return self.a 
        
    def observe(self, s, r, terminal=False, truncated=None):
        self.s = s
        self.r = r
        self.truncated = truncated
        self.t += 1
        self.b += r
        self.agent.observe(self.r, terminal=terminal)
        self.N[self.s[0], self.s[1]] += 1
        
    def experience(self, states, actions, terminal, reward):
        self.agent.experience(states, actions, terminal, reward)

    def update(self):
        self.agent.update()
        
    def close(self):
        self.agent.close()