from pyrl import Agent
from collections.abc import Iterable
from tensorforce.agents import Agent as TFAgent

class SurvivalA2CAgent(Agent):
    """
        Advantage Actor Critic Agent.
    """
    
    def __init__(self, environment, batch_size, survival_threshold=250,
                 exploration_threshold=500, memory='minimum', initial_observation=None,
                initial_budget=1000, max_episode_timesteps=None, network='auto', 
                use_beta_distribution=False, update_frequency=1.0, learning_rate=0.001,
                horizon=5000, discount=0.99, reward_processing=None, return_processing=None,
                advantage_processing=None, predict_terminal_values=False, critic='auto',
                critic_optimizer=1.0, state_preprocessing='linear_normalization', exploration=0.5,
                variable_noise=0.0, l2_regularization=0.0, entropy_regularization=0.0,
                parallel_interactions=1, config=None, saver=None, summarizer=None, tracking=None,
                recorder=None, **kwargs
                ):
                
        self.agent = TFAgent.create(
                agent='a2c',
                environment=environment,
                memory=memory,
                batch_size=batch_size,
                max_episode_timesteps=max_episode_timesteps,
                exploration=exploration,
                network=network, 
                use_beta_distribution=use_beta_distribution,
                update_frequency=update_frequency,
                learning_rate=learning_rate,
                horizon=horizon,
                discount=discount,
                reward_processing=reward_processing,
                return_processing=return_processing,
                advantage_processing=advantage_processing,
                predict_terminal_values=predict_terminal_values,
                critic=critic,
                critic_optimizer=critic_optimizer,
                state_preprocessing=state_preprocessing,
                variable_noise=variable_noise,
                l2_regularization=l2_regularization,
                entropy_regularization=entropy_regularization,
                parallel_interactions=parallel_interactions,
                config=config,
                saver=saver,
                summarizer=summarizer,
                tracking=tracking,
                recorder=recorder
            )
        self.environment = environment
        self.s = initial_observation
        self.initial_budget = initial_budget
        self.b = self.initial_budget
        
        self.survival_threshold = survival_threshold
        self.exploration_threshold = exploration_threshold
        
    def reset(self, s, reset_knowledge=True): 
        self.t = 0 #time, or number of elapsed rounds
        self.s = s  if isinstance(s, Iterable)  else  [s] #memory of the current state and last received reward
        self.r = 0.0   
        self.b = self.initial_budget
        self.a = self.environment.action_space.sample() #next chosen action
        self.recharge_mode = False
        if reset_knowledge:
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
        
        if not self.recharge_mode and self.b < self.survival_threshold:
            self.recharge_mode = True
        
        if self.recharge_mode and self.b > self.exploration_threshold:
            self.recharge_mode = False
        
        self.agent.observe(self.r, terminal=terminated)
    
    def learn(self):
        pass
