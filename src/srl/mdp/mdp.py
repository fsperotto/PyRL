import math
from math import sin, cos, atan, sqrt
import numpy as np

from random import sample, randint, randrange, random

############################################################################

class RandomAgent:
    
    def __init__(self, problem):
        self.problem = problem
        
    def choose(self):
        return [randint(-1,+1) for n in self.problem.art_num_states]

    def update(self):
        pass
    
    def act(self):
        self.choose()
        self.problem.update(self.action)
        self.update()
    
############################################################################

class QLearningAgent:

    def __init__(self, problem, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.problem = problem
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.modified = False
        self.state = self.problem.cur_flat_state()
        self.action = self.problem.cur_flat_action()
        self.reward = self.problem.last_reward
        self.q_values = np.zeros(shape=(self.problem.num_flat_states, self.problem.num_flat_actions) )
        
    #Q-Learning updating
    def update(self):
        
        self.modified = False

        prev_state = self.state
        self.state = self.problem.cur_flat_state()
        self.reward = self.problem.last_reward
        
        old_q_value = self.q_values[prev_state, self.action]
        new_q_value = self.reward + (self.gamma * max(self.q_values[self.state]))

        self.q_values[prev_state, self.action] = (1-self.alpha) * old_q_value + self.alpha * new_q_value
        
        #greedy test
        #self.q_values[prev_state, self.action] = new_q_value
        
#        diff = new_q_value - old_q_value
#    
#        if (diff != 0.0):
#            #flag modification
#            self.modified = True
#            new_q_value = old_q_value + self.alpha * diff
#            self.q_values[prev_state, self.action] = new_q_value
#            #updateSpecificValueSfromSA(iPrevState);

    def choose(self):
        self.state = self.problem.cur_flat_state()
        if random() > self.epsilon:
            #best_q_value = max(self.q_values[self.state])
            #self.action = self.q_values[self.state].index(best_q_value)
            self.action = np.argmax(self.q_values[self.state])
            return self.problem.to_art_action(self.action)
        else:
            return [randint(-1,+1) for n in self.problem.art_num_states]

    def act(self):
        self.choose()
        self.problem.update(self.action)
        self.update()
    
############################################################################

class PolicyIterationAgent:

    def __init__(self, problem, gamma=0.9):
        self.problem = problem
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.modified = False
        self.state = self.problem.cur_flat_state()
        self.action = self.problem.cur_flat_action()
        self.reward = self.problem.last_reward
        self.v_values = np.zeros(shape=(self.problem.num_flat_states) )
        self.stats = {}
        self.learn()

    def learn(self):
        #initial policy
        self.policy = np.zeros(shape=(self.problem.num_flat_states), dtype=int)
        is_policy_changing = True
        #iterate until definitive policy
        self.stats['count_policy_iterations'] = 0
        self.stats['count_evaluate_iterations'] = []
        while is_policy_changing:
            self.stats['count_policy_iterations'] += 1
            self.stats['count_evaluate_iterations'] += [0]
            #evaluate policy
            delta = float('inf')
            theta = 0.0000001
            while delta > theta:
                self.stats['count_evaluate_iterations'][-1] += 1
                delta = 0.0
                for s in range(self.problem.num_flat_states):
                    a = self.policy[s]
                    v = self.v_values[s]
                    next_s = self.problem.next_flat_state(s, a)
                    next_v = self.v_values[next_s]
                    r = self.problem.expected_reward_flat(s, a, next_s)
                    self.v_values[s] = r + self.gamma * next_v
                    delta = max(delta, abs(v - self.v_values[s]))
            #policy improvement
            is_policy_changing = False
            for s in range(self.problem.num_flat_states):
                new_v = -float('inf')
                for a in range(self.problem.num_flat_actions):
                    next_s = self.problem.next_flat_state(s, a)
                    next_v = self.v_values[next_s]
                    r = self.problem.expected_reward_flat(s, a, next_s)
                    v = r + self.gamma * next_v
                    if v > new_v:
                        new_v = v
                        new_a = a
                if new_a != self.policy[s]:
                    self.policy[s] = new_a
                    is_policy_changing = True
        print('policy iterations:', self.stats['count_policy_iterations'], '; evaluate iterations:', self.stats['count_policy_iterations'])
                
    def update(self):
        pass

    def choose(self):
        self.state = self.problem.cur_flat_state()
        self.action = self.policy[self.state]
        #self.action = np.argmax(self.q_values[self.state])
        #return self.problem.to_art_action(self.action)
        return self.action

    def act(self):
        self.choose()
        self.problem.update(self.action)
        self.update()

       