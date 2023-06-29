# from tensorforce.agents import AdvantageActorCritic
from collections.abc import Iterable
from pyrl import Agent
from tensorforce import Agent as TensorforceAgent

class A2CAgent(Agent):
    """
        Advantage Actor Critic Agent
    """
    
    def __init__(
        self, env, observation_space, action_space, batch_size, initial_observation=None,
        max_episode_timesteps=None,
        network='auto', use_beta_distribution=False,
        memory='minimum',
        update_frequency=1.0, learning_rate=1e-3,
        horizon=1, discount=0.99, reward_processing=None, return_processing=None,
        advantage_processing=None,
        predict_terminal_values=False,
        critic='auto', critic_optimizer=1.0,
        state_preprocessing='linear_normalization',
        exploration=0.0, variable_noise=0.0,
        l2_regularization=0.0, entropy_regularization=0.0,
        parallel_interactions=1,
        config=None, saver=None, summarizer=None, tracking=None, recorder=None,
        # **kwargs
    ):
        # super().__init__(observation_space, action_space, initial_observation=initial_observation)
        self.observation_space = observation_space
        self.action_space = action_space
        self.initial_observation = initial_observation
        # global agent
        self.agent = TensorforceAgent.create(
                        agent='a2c',
                        environment=env,
                        batch_size=batch_size,
                        # max_episode_timesteps=100,
                        # network=network,
                        # use_beta_distribution=use_beta_distribution,
                        # update_frequency=update_frequency,
                        # learning_rate=learning_rate,
                        # horizon=horizon,
                        # discount=discount,
                        # reward_processing=reward_processing,
                        # return_processing=return_processing,
                        # advantage_processing=advantage_processing,
                        # predict_terminal_values=predict_terminal_values,
                        # critic=critic,
                        # critic_optimizer=critic_optimizer,
                        # state_preprocessing=state_preprocessing,
                        # exploration=exploration,
                        # variable_noise=variable_noise,
                        # l2_regularization=l2_regularization,
                        # entropy_regularization=entropy_regularization,
                        # parallel_interactions=parallel_interactions,
                        # config=config,
                        # saver=saver,
                        # summarizer=summarizer,
                        # tracking=tracking,
                        # recorder=recorder,
                        # memory=memory,
                        )
        
        self.episode_terminal = False
        
    
    def reset(self, initial_observation, reset_knowledge=True):
        #time, or number of elapsed rounds 
        self.t = 0   
        #memory of the current state and last received reward
        self.s = initial_observation  if isinstance(initial_observation, Iterable)  else  [initial_observation]
        self.r = 0.0
        #next chosen action
        #self.a = [None for _ in range(self.num_action_vars)] 
        self.a = self.action_space.sample()
        # self.agent.reset()
        # super().reset(initial_observation, reset_knowledge=reset_knowledge)
        # print(self.agent)
        
    def initial_internals(self):
        self.internals = self.agent.initial_internals()
        return self.internals
    
    def act(self) -> list :
        # internals = self.internals
        states = self.s
        self.a = self.agent.act(states=states) 
        return self.a 
        
    def observe(self, s, r, terminal=False):
        # super().observe(s=s, r=r, terminal=terminal)
        self.s = s
        self.r = r
        self.agent.observe(reward=r, terminal=terminal)
        
    # def experience(self, states, internals, actions, terminal, reward):
    #     self.agent.experience(states, internals, actions, terminal, reward)

    def experience(self, states, actions, terminal, reward):
        self.agent.experience(states, actions, terminal, reward)

    def update(self):
        self.agent.update()
        
    def close(self):
        self.agent.close()