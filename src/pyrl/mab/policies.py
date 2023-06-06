#Import Dependencies
import numpy as np
from numpy.random import binomial, randint, uniform, rand, normal, choice
from random import choices
import numpy.ma as ma
from math import sqrt, log, inf
from scipy.stats import beta, norm
from scipy.integrate import quadrature as integral
from collections import Iterable
from numba import jit
from itertools import accumulate as acc
from decimal import Decimal


################################################################################

class BasePolicy():
    """ Base class for any policy."""

    #def __new__(cls, k, **kwargs):
    #    #if type(self) == BasePolicy:
    #    if cls == BasePolicy:
    #        raise Exception("[ERROR] cannot instantiate AbstractPolicy")
    #    else:
    #        return super().__new__(cls)
        
    def __init__(self, k, **kwargs):
        """k is the number of arms
           w=0 is the number of times each arm must be played before true start
           label=None is the name of the strategy
           style=None is a dict of plotting properties: label, linecolor, linestyle, linewidth, marker...
        """
        
        #number of arms
        if (not isinstance(k, int)) or (k < 2):
            raise Exception("[ERROR] k must be an integer greater or equal 2.")
        self.k = k 
        
        #list of prioritary decision rules (executed before base rules, depending on conditions)
        #if empty, go directly to the base rule
        self.pred_rules = []

        #base decision rule (empty in this abstract implementation)
        self.base_rule = None
        
        #number of times each arm is tried before impact budget
        if 'w' in kwargs and kwargs['w'] is not None:
            self.w = kwargs['w']
            if (not isinstance(self.w, int)) or (self.w < 0):
                raise Exception("[ERROR] w must be an integer greater or equal 0.")
            # first decision rule, mandatory initial rounds, if any
            if (self.w > 0):
                self.pred_rules.append(self.mandatory_choice)
        else:
            #default
            self.w = 0        #disabled

        #initial budget
        if 'b_0' in kwargs and kwargs['b_0'] is not None:
            self.b_0 = kwargs['b_0']
            if (self.b_0 <= 0.0):
                raise Exception("[ERROR] when defined, b_0 must be greater than 0.0.")
            #set flag
            self.is_budgeted = True
        else:
            #default
            self.b_0 = None    #disabled
            self.is_budgeted = False

        #safety-threshold
        if 'omega' in kwargs and kwargs['omega'] is not None:
            self.omega = kwargs['omega']
            if (not self.is_budgeted):
                raise Exception("[ERROR] b_0 must be defined for using omega.")
            if (self.omega <= 0.0):
                raise Exception("[ERROR] when defined, omega must be greater than 0.0.")
            #set flag
            self.is_alarmable = True
            # include safety-threshold (alarmed) policy in the priority rules, if defined
            self.pred_rules.append(self.alarmed_choice)
        else:
            #default
            self.omega = None   #disabled
            self.is_alarmable = False

        #property that says if is ruined when budget is over
        if 'ruinable' in kwargs and kwargs['ruinable'] is not None:
            self.is_ruinable = kwargs['ruinable']
        else:
            #default
            self.is_ruinable = self.is_budgeted

        # include ruined policy, if the agent is ruinable
        if self.is_ruinable:
            self.pred_rules.append(self.ruined_choice)
            
        #how equivalent utility arms are preferred
        if 'tiebreak_mode' in kwargs and kwargs['tiebreak_mode'] in ['random', 'first', 'last', 'sequential', 'reversed']:
            self.tiebreak_mode = kwargs['tiebreak_mode']
        else:
            self.tiebreak_mode = 'sequential'

        #index of the first arm if play the arms in sequence, negative if in inverse order, otherwise None
        if 'first_i' in kwargs and kwargs['first_i'] is not None:
            if (not isinstance(kwargs['first_i'], int)) or (kwargs['first_i'] == 0) or (abs(kwargs['first_i']) >= k) :
                raise Exception("[ERROR] first_i must be an integer betwen 1 and k, or between -1 and -k for inverse order.")
            self.inversed_order = kwargs['first_i'] < 0
            self.first_i = abs(kwargs['first_i'])-1
        else:
            #default
            self.inversed_order = False
            self.first_i = 0
            
        # internal states
        self.t = 0     #internal time-step (current round)
        self.s = 0.0   #total cumulated rewards
        self.n_i = np.zeros(k, dtype=int)  #number of pulls of each arm
        self.s_i = np.zeros(k)  # cumulated rewards for each arm
        self.mu_i = np.zeros(k)  # mean for each arm
        self.i = None  #last pulled arm
        self.r = None  #last reward

        self.b = self.b_0       #budget
        self.is_ruined = False
            
        #define label (name of the strategy)
        if 'label' in kwargs:
            self.label = kwargs['label']
        else:
            self.label = None

        #for plotting
        if 'style' in kwargs:
            self.style = kwargs['style']
        else:
            self.style = {}
        
    def set_base_rule(self, rule, prob=None, cycle=None):
        #the rules are drawn from a determined probability
        self.base_prob = None
        self.base_acc_cycle = None
        self.cycle_len = None
        self.base_rule = rule
        if prob is not None:
            if (cycle is not None):
                raise Exception("[ERROR] prob cannot be defined at same time than cycle.")
            if (not isinstance(prob, list)) or (not isinstance(rule, list)) or (len(prob) != len(rule)):
                raise Exception("[ERROR] when using epsilon, prob and rule must be lists of the same size.")
            self.base_prob = prob
        elif cycle is not None:
            if (not isinstance(cycle, list)) or (not isinstance(rule, list)) or (len(cycle) != len(rule)):
                raise Exception("[ERROR] when using cycles, cycle and rule must be lists of the same size.")
            self.base_acc_cycle = list(acc(cycle))
            self.cycle_len = sum(cycle)
        
    def reset(self):
        """ Start the game (fill counters and statistics with 0)."""
        self.t = 0     #internal time-step (current round)
        self.s = 0.0   #total cumulated rewards
        self.n_i.fill(0)  #number of pulls of each arm
        self.s_i.fill(0.0)  # cumulated rewards for each arm
        self.mu_i.fill(0.0)  # mean for each arm
        self.i = None  #last pulled arm
        self.r = None  #last reward
        self.b = self.b_0   #budget
        self.is_ruined = False

    def observe(self, r):
        """ Receive reward, increase t, pulls, and update """
        self.r = r       #
        self.update()    #update all counters and statistics

    def update(self):
        #update internal state (based on last reward)
        self.t += 1 
        self.n_i[self.i] += 1
        self.s_i[self.i] += self.r
        self.mu_i[self.i] = self.s_i[self.i] / self.n_i[self.i]
        if (self.w > 0) and (self.t >= self.w * self.k) :
            self.s += self.r
        if (self.is_budgeted):
            self.b = self.b_0 + self.s
            self.is_ruined = (self.is_ruinable and self.b <= 0.0)

    def tiebreak_choice(self, bests):
        if self.tiebreak_mode == 'first':
            return bests[0]
        elif self.tiebreak_mode == 'last':
            return bests[-1]
        elif self.tiebreak_mode == 'sequential':
            return bests[self.t % len(bests)]
        elif self.tiebreak_mode == 'reversed':
            return bests[len(bests) - 1 - (self.t % len(bests))]
        else:  #'random'
            return choice(bests)
            
    def no_choice(self):
        #return always -1 (no choice)
        return -1

    def random_choice(self):
        # uniform random choice among the arms
        return randint(self.k)

    def sequential_choice(self):
        # play arms in order repeatedly
        #suppose that first_i (between 0 and k-1) and inversed_order are defined
        if self.inversed_order:
            return (self.k-1) - ((self.t + self.first_i) % self.k)
        else:
            return (self.t + self.first_i) % self.k
            
    def mandatory_choice(self):
        if (self.t < (self.k * self.w)):
            # play each arm w times, in order
            return self.sequential_choice()
        else:
            return None

    def greedy_choice(self):
        #uniform choice among the arms with maximal mean
        bests = np.flatnonzero(self.mu_i == np.max(self.mu_i))
        return self.tiebreak_choice(bests)

    def greedy_sequential_choice(self):
        #first choice among the arms with maximal mean
        if self.inversed_order:
            return (self.k-1) - np.argmax(self.mu_i[::-1])
        else:
            return np.argmax(self.mu_i)
    
    def alarmed_choice(self):
        #low budget
        #suppose that is_budgeted and is_alarmable are both true
        if (self.b <= self.omega) and (np.max(self.mu_i) > 0.0):
            return self.greedy_choice()
        #sufficient budget
        else:
            return None

    def ruined_choice(self):
        #if budgeted and ruined, return always -1 (no choice)
        if (self.b <= 0.0):
            return -1
        else:
            return None
    
    
    def choose(self):
        
        #try the prioritarian rules in order, if the rule make a decision, return it
        for rule in self.pred_rules:
            self.i = rule()
            if self.i is not None:
                return self.i
        
        #if rules are chosen from a distribution
        if self.base_prob is not None:
            rule = choice(self.base_rule, p=self.base_prob)
            self.i = rule()
            
        #if rules are cyclical in a defined number of rounds
        elif self.base_acc_cycle is not None:
            idx_rule = next(j for j,t in enumerate(self.base_acc_cycle) if self.t%self.cycle_len < t)
            rule = self.base_rule[idx_rule]
            self.i = rule()
        
        #otherwise uses the base rule
        else:
            self.i = self.base_rule()
        
        return self.i

    def __str__(self):
        if self.label is not None:
            return self.label
        else:
            lbl = self.default_label()
            if self.is_alarmable:
                lbl = "ST-" + lbl
            return lbl

    def prob_ruin_inf(self, means, minr=None, maxr=None, variances=None):
        return None

    def prob_ruin_at(self, t, means, minr=None, maxr=None, variances=None):
        return None

    def prob_ruin_until(self, t, means, minr=None, maxr=None, variances=None):
        return None

    def expected_instant_reward(self, t, means, minr=None, maxr=None, variances=None):
        return None

    def expected_cumulated_reward(self, h, means, minr=None, maxr=None, variances=None):
        return None

    def expected_budget(self, h, means, minr=None, maxr=None, variances=None):
        return None

    def expected_instant_regret(self, t, means, minr=None, maxr=None, variances=None):
        return None

    def expected_cumulated_regret(self, t, means, minr=None, maxr=None, variances=None):
        return None
    

################################################################################

def zcomb(n:int, k:int):
    #k and n must be naturals
    if (n >= k) and (k >= 0):
        return comb(n, k)
    else:
        return 0

################

def exact_ruin_dist(t, b, p=0.5):
    if (abs(b)>t) or ((t+b)%2):
        return 0
    else:
        #centered catalan triangle
        k = (t+b)//2
        n = k-1 #(t+b-2)//2
        m = (t-b)//2
        c = max(0, zcomb(n+m, m) - zcomb(n+m, m-1))
        try:
            return c * p**(t-k) * (1-p)**k
        except:
            return Decimal(c) * Decimal(p)**(t-k) * Decimal((1-p))**k

################

def ruin_dist(h, b, p=0.5):
    s = 0.0
    for t in range(h):
        s += exact_ruin_dist(t, b, p)
    return s    

###############

def prob_ruin_inf(p):
    return ((1-p)/p)**b_0  if  p > 0.5  else  1.0

###############

class FixedPolicy(BasePolicy):

    def __init__(self, k, fixed_i=1, **kwargs):
        
        super().__init__(k, **kwargs)

        if (not isinstance(fixed_i, int)) or (fixed_i <= 0) or (fixed_i > k) :
            raise Exception("[ERROR] fixed_i must be an integer betwen 1 and k.")
        
        #index if always play the same arm, otherwise None
        #note: convert the parameter between 1..k to 0..k-1
        self.fixed_i = fixed_i-1
        
        self.set_base_rule(self.fixed_choice)

    def fixed_choice(self):
        return self.fixed_i
        
    def default_label(self):
        return "Fixed ($i=" + str(self.fixed_i+1) + "$)"

    def prob_ruin_at(self, means, t=None, minr=None, maxr=None, variances=None, dist_type='bernoulli'):
        if self.is_ruinable and min_r < 0.0 and t is not None and t < float('inf'):
            if dist_type == 'bernoulli' and min_r == -1.0 and max_r == 1.0:
                return exact_ruin_dist(t, self.b_0, p=p_arr[self.fixed_i])
            else:
                return None
        else:
            return 0.0

    def prob_ruin_until(self, means, h=None, minr=None, maxr=None, variances=None, dist_type='bernoulli'):
        if self.is_ruinable and min_r < 0.0:
            if dist_type == 'bernoulli':
                p = p_arr[self.fixed_i]
                if h is None or h == float('inf'):
                    if p > 0.5:
                        return ((1-p)/p)**self.b_0
                    else:
                        return 1.0
                else:
                    return ruin_dist(t, self.b_0, p)
        else:
            return 0.0

    def expected_instant_reward(self, t, means, minr=None, maxr=None, variances=None, dist_type='bernoulli'):
        if self.is_ruinable:
            return (1.0 - self.prob_ruin_until(t)) * means[self.fixed_i]
        else:
            return means[self.fixed_i]

    def expected_cumulated_reward(self, h, means, minr=None, maxr=None, variances=None):
        return None

    def expected_budget(self, h, means, minr=None, maxr=None, variances=None):
        return None

    #the expected regret if the distribution parameters are known
    def expected_instant_regret(self, t, means, minr=None, maxr=None, variances=None):
        return max(means) - means[self.fixed_i]

    def expected_cumulated_regret(self, t, means, minr=None, maxr=None, variances=None):
        return None
        

################################################################################

class GreedyPolicy(BasePolicy):
    r""" Class that implements the empirical means method
    The naive Empirical Means policy for bounded bandits: like UCB but without a bias correction term. 
    Note that it is equal to UCBalpha with alpha=0, only quicker.
    choose an arm with maximal index (uniformly at random):
    .. math:: A(t) \sim U(\arg\max_{1 \leq k \leq K} I_k(t)).
    .. note:: In almost all cases, there is a unique arm with maximal index, so we loose a lot of time with this generic code, but I couldn't find a way to be more efficient without loosing generality.
    """
    def __init__(self, k, **kwargs):
        
        super().__init__(k, **kwargs)
        
        # base choice rule is greedy based on the empirical means
        self.set_base_rule(self.greedy_choice)
        
    def default_label(self):
        return "Greedy Empirical Means"
        
        
################################################################################

class RandomPolicy(BasePolicy):

    def __init__(self, k, **kwargs):
    
        super().__init__(k, **kwargs)
        
        # base choice rule is random
        self.set_base_rule(self.random_choice)
        
    def default_label(self):
        return  "Uniform Random"

    def expected_reward(self, h, means, minr=None, maxr=None, variances=None):
        return np.mean(means)

    def expected_regret(self, h, means, minr=None, maxr=None, variances=None):
        return max(means) - np.mean(means)


################################################################################

class EpsilonGreedyPolicy(BasePolicy):
    
    def __init__(self, k, eps=0.1, **kwargs):
    
        super().__init__(k, **kwargs)
        
        self.eps = eps
        
        # base choice rule is random with probability epsilon, and greedy otherwise
        self.set_base_rule([self.random_choice, self.greedy_choice], prob=[self.eps, 1.0-eps])

    def default_label(self):
        return  r"$\varepsilon$-Greedy ($\varepsilon=" + str(round(self.eps,2)) + "$)"
        
################################################################################

class SequentialEpsilonGreedyPolicy(BasePolicy):
    
    def __init__(self, k, eps=0.1, **kwargs):
    
        super().__init__(k, **kwargs)
        
        self.eps = eps
        
        # base choice rule is sequential (uniform) with probability epsilon, and greedy otherwise
        self.set_base_rule([self.sequential_choice, self.greedy_choice], prob=[eps, 1.0-eps])

    def default_label(self):
        return r"Sequential-$\varepsilon$-Greedy ($\varepsilon=" + str(round(self.eps,2)) + "$)"
            
################################################################################

class SequentialPolicy(BasePolicy):
    
    def __init__(self, k, **kwargs):
    
        super().__init__(k, **kwargs)
        
        # base choice rule is random with probability epsilon, and greedy otherwise
        self.set_base_rule(self.sequential_choice)

    def default_label(self):
        return  "Sequential"
            
################################################################################

class RandomExploreThenExploitPolicy(BasePolicy):
    
    def __init__(self, k, h_explore=100, h_exploit=inf, **kwargs):
    
        super().__init__(k, **kwargs)

        # base choice rule is random then greedy
        self.set_base_rule([self.random_choice, self.greedy_choice], cycle=[h_explore, h_exploit])

    def default_label(self):
        return   f"Explore-Then-Exploit" # ({self.base_cycle})"

################################################################################
      
ExploreThenExploitPolicy = RandomExploreThenExploitPolicy
      
################################################################################

class SequentialExploreThenExploitPolicy(BasePolicy):
    
    def __init__(self, k, h_explore=100, h_exploit=inf, **kwargs):
    
        super().__init__(k, **kwargs)

        # base choice rule is random then greedy
        self.set_base_rule([self.sequential_choice, self.greedy_choice], cycle=[h_explore, h_exploit])

    def default_label(self):
        return   f"Sequential-Explore-Then-Exploit" # ({self.base_cycle})"
            
################################################################################

class PositiveGreedyPolicy(BasePolicy):

    def __init__(self, k, max_pot_h=20, **kwargs):
    
        super().__init__(k, **kwargs)

        #max horizon to search potentials
        self.max_pot_h = max_pot_h
        
        # base choice rule is random then greedy
        self.set_base_rule(self.positive_choice)

    def default_label(self):
        return   f"Positive Greedy" # ({self.base_cycle})"

    def potential_choice(self):
        #uniform choice among the arms with maximal mean considering a new supplementary success
        for j in range(1, self.max_pot_h):
            pot_i = (self.s_i+j) / (self.n_i+j)
            mu_star_pot = np.max(pot_i)
            bests = np.flatnonzero(pot_i == mu_star_pot)
            n_min = np.min(self.n_i[bests])
            #n_max = np.max(self.n_i[bests])
            less  = np.flatnonzero(self.n_i == n_min)
            bests = np.intersect1d(bests, less)
            if mu_star_pot > 0:
                break
        return self.tiebreak_choice(bests)

    def positive_choice(self):
        #greedy estimate is positive
        if np.max(self.mu_i) > 0.0:
            return self.greedy_choice()
        #otherwise try best potential
        else:
            return self.potential_choice()
            #return self.random_choice()

################################################################################

class IndexedPolicy(BasePolicy):

    def __init__(self, k, v_0=0.0, **kwargs):
        
        super().__init__(k, **kwargs)

        if type(self) == IndexedPolicy:
            raise Exception("[ERROR] cannot instantiate AbstractBudgetedIndexedPolicy")
        
        # parameters
        self.v_0 = v_0   #initial value (index or utility) for the arms

        self.set_base_rule(self.utilitarian_choice)

        # internal state
        self.v_i = np.full(self.k, self.v_0)  # value (index or utility) for each arm
        
    def reset(self):
        super().reset()
        self.v_i = np.full(self.k, self.v_0)  # value (index or utility) for each arm
        
    def observe(self, r):
        """ Receive reward, increase t, pulls, and update """
        super().observe(r)
        self.evaluate()
        
    def evaluate(self):
        #evaluate utility (if any)
        #self.v_i[self.i] = ...
        pass
        
    def utilitarian_choice(self):
        #uniform choice among the arms with maximal utility
        bests = np.flatnonzero(self.v_i == np.max(self.v_i))
        return choice(bests)

        
################################################################################

class EmpiricalSumPolicy(IndexedPolicy):
    r""" The empirical sum policy.
    - At every time step, the arm with max total sum is chosen. 
    --> It is a possible greedy policy for zero centered reward domains.
    """
    def __init__(self, k, **kwargs):
        
        super().__init__(k, v_0=0.0, **kwargs)

        self.v_i = self.s_i
        
    def default_label(self):
        return   "Greedy Empirical Sum"

    def evaluate(self):
        #utility is the cumulated sum of rewards
        #self.v_i[self.i] = self.s_i[self.i]
        pass
            

################################################################################

class SoftMaxPolicy(IndexedPolicy):
    r"""The Boltzmann , label=NoneExploration (Softmax) index policy, with a constant temperature :math:`\eta_t`.
    - Reference: [Algorithms for the multi-armed bandit problem, V.Kuleshov & D.Precup, JMLR, 2008, §2.1](http://www.cs.mcgill.ca/~vkules/bandits.pdf) and [Boltzmann Exploration Done Right, N.Cesa-Bianchi & C.Gentile & G.Lugosi & G.Neu, arXiv 2017](https://arxiv.org/pdf/1705.10257.pdf).
    - Very similar to Exp3 but uses a Boltzmann distribution.
    Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://sbubeck.com/SurveyBCB12.pdf)
    """

    def __init__(self, k, eta=0.1, **kwargs):
        
        super().__init__(k,  **kwargs)
        
        if eta <= 0:
            raise Exception("[ERROR] the temperature parameter for softmax must be greater than 0.")
            
        self.eta = eta # np.sqrt(np.log(k) / k)
        
        # base choice rule is softmax
        self.set_base_rule(self.softmax_choice)
                  
    def default_label(self):
        return f"SoftMax ($\\eta={round(self.eta,2)}$)"
            
    def evaluate(self):
        r"""Update the trusts probabilities according to the Softmax (ie Boltzmann) distribution on accumulated rewards, and with the temperature :math:`\eta_t`.
        .. math::
        \mathrm{trusts}'_k(t+1) &= \exp\left( \frac{X_k(t)}{\eta_t N_k(t)} \right) \\
        \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).
        If :math:`X_k(t) = \sum_{\sigma=1}^{t} 1(A(\sigma) = k) r_k(\sigma)` is the sum of rewards from arm k.
        """
        self.v_i[self.i] = np.exp(self.mu_i[self.i] / self.eta)
    
    def softmax_choice(self):
        # Calculate Softmax probabilities
        sum_v = sum(self.v_i)
        probs = self.v_i / sum_v
        # Use categorical_draw to pick arm
        return choice(self.k, p=probs)
    

################################################################################

class UCBPolicy(IndexedPolicy):

    def __init__(self, k, r_min=0.0, r_max=1.0, alpha=0.5, alpha_prime=None, **kwargs):
        super().__init__(k,  v_0=float('+inf'), **kwargs)
        if alpha_prime is not None:
            self.alpha_prime = alpha_prime
            self.alpha = alpha / (r_max - r_min)**2
        else:
            self.alpha = alpha
            self.alpha_prime = (r_max - r_min)**2 * alpha
        
    def default_label(self):
        return   r"UCB ($\alpha=" + str(round(self.alpha,2)) + ", \alpha'= " + str(round(self.alpha_prime,2)) + "$)"
        
    def evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:
        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha \log(t)}{N_k(t)}}.
        """
        #calculate utility following UCB formula
        for i in range(self.k): 
            if self.n_i[i] == 0:
                self.v_i[i] = float('+inf')
            else:
                self.v_i[i] = self.mu_i[i] + sqrt(self.alpha_prime * log(self.t) / self.n_i[i])

################################################################################

class GamblerUCBPolicy(UCBPolicy):

    def default_label(self):
        return   r"Gambler UCB ($\alpha=" + str(round(self.alpha,2)) + ", \alpha'= " + str(round(self.alpha_prime,2)) + "$)"
        
    def evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:
        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{\alpha \log(t)}{N_k(t)}}.
        """
        #calculate utility following UCB formula
        #calculate utility following UCB formula
        t = max(1, self.t)
        b = max(1, self.b)
        #factor is near from zero when budget is very low, and approaches 1 when budget is high
        f = (1 - (1 / b))
        #f = sqrt(1 - (1 / b))
        #f = b/t
        for i in range(self.k): 
            if self.n_i[i] == 0:
                self.v_i[i] = float('+inf')
            else:
                #v1
                self.v_i[i] = self.mu_i[i] + sqrt(self.alpha_prime * log(b) / self.n_i[i])
                #v2
                #y = f*t + (1-f)*b
                #self.v_i[i] = self.mu_i[i] + sqrt(self.alpha_prime * log(y) / self.n_i[i])
                #v3
                #self.v_i[i] = self.mu_i[i] + sqrt(f * self.alpha_prime * log(t) / self.n_i[i])
                #v4
                #w_i = t / self.n_i[i]
                #z_i = b * w_i
                #self.v_i[i] = self.mu_i[i] + sqrt(self.alpha_prime * log(b) / z_i)


################################################################################

class ThompsonPolicy(IndexedPolicy):
    """The Thompson (Bayesian) index policy.
    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    - Prior is initially flat, i.e., :math:`a=\alpha_0=1` and :math:`b=\beta_0=1`.
    - Reference: [Thompson - Biometrika, 1933].
    """

    def __init__(self, k, prior=1.0, r_min=-1.0, r_max=+1.0, **kwargs):

        super().__init__(k, **kwargs)
        
        self.r_min = r_min
        self.r_max = r_max
        self.r_amp = r_max-r_min
        self.prior = prior   #prior factor for Beta dist parameters (generally 1.0 for uniform, or 0.5 for uninformative)
        
    def default_label(self):
        return "Thompson-Sampling"
                  
    def evaluate(self):
        """ Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, 
        giving :math:`S_k(t)` rewards of 1, by sampling from the Beta posterior:
        .. math::
        A(t) &\sim U(\arg\max_{1 \leq k \leq K} I_k(t)),\\
        I_k(t) &\sim \mathrm{Beta}(1 + \tilde{S_k}(t), 1 + \tilde{N_k}(t) - \tilde{S_k}(t)).
        """
        for i in range(self.k):
            #v_alpha = self.s_i[i] + 1
            #v_beta = self.n_i[i] - self.s_i[i] + 1
            #correct alpha and beta parameters (successes and failures) considering reward bounds
            x = max(0.0, (self.s_i[i] - (self.n_i[i] * self.r_min)) / self.r_amp)
            y = max(0.0, self.n_i[i] - x)
            self.v_i[i] = beta.rvs(x + self.prior, y + self.prior)

################################################################################

class GamblerThompsonPolicy(ThompsonPolicy):
    r"""The Thompson (Bayesian) index policy.
    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    - Prior is initially flat, i.e., :math:`a=\alpha_0=1` and :math:`b=\beta_0=1`.
    - Reference: [Thompson - Biometrika, 1933].
    """

    def default_label(self):
        return   "Gambler-Thompson-Sampling"
                  
    def evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, 
        giving :math:`S_k(t)` rewards of 1, by sampling from the Beta posterior:
        .. math::
        A(t) &\sim U(\arg\max_{1 \leq k \leq K} I_k(t)),\\
        I_k(t) &\sim \mathrm{Beta}(1 + \tilde{S_k}(t), 1 + \tilde{N_k}(t) - \tilde{S_k}(t)).
        """
        c = 0.0001
        #factor (for prior) is near from zero when budget is very low, and approaches 1 when budget is high
        weighted_prior = (1 - (1 / (self.b+1+c))) * self.prior
        
        #draw samples from posterior
        for i in range(self.k):
            #v_alpha = self.s_i[i] + 1
            #v_beta = self.n_i[i] - self.s_i[i] + 1
            #correct alpha and beta parameters (successes and failures) considering reward bounds
            x = max(0.0, (self.s_i[i] - (self.n_i[i] * self.r_min)) / self.r_amp)
            y = max(0.0, self.n_i[i] - x)
            self.v_i[i] = beta.rvs(x + weighted_prior, y + weighted_prior)

################################################################################

class BayesUCBPolicy(IndexedPolicy):
    """ The Bernoulli-Bayes-UCB policy.
    - uses a Beta conjugate prior for Bernoulli likelihood.
    - Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def __init__(self, k, prior=1.0, r_min=-1.0, r_max=+1.0, **kwargs):

        super().__init__(k, **kwargs)
        
        self.r_min = r_min
        self.r_max = r_max
        self.r_amp = r_max-r_min
        self.prior = prior   #prior factor for Beta dist parameters (generally 1.0 for uniform, or 0.5 for uninformative)

    def default_label(self):
        return  "Bayes-UCB (Beta)"

    def evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by taking the :math:`1 - \frac{1}{t}` quantile from the Beta posterior:
        .. math:: I_k(t) = \mathrm{Quantile}\left(\mathrm{Beta}(1 + S_k(t), 1 + N_k(t) - S_k(t)), 1 - \frac{1}{t}\right).
        """
        t = max(2, self.t)
        q = 1 - (1 / t)

        #i = self.i_last
        for i in range(self.k): 
            #correct alpha and beta parameters (successes and failures) considering reward bounds
            x = max(0.0, (self.s_i[i] - (self.n_i[i] * self.r_min)) / self.r_amp)
            y = max(0.0, self.n_i[i] - x)
            self.v_i[i] = beta.ppf(q, x + self.prior, y + self.prior)

################################################################################

class GamblerBayesUCBPolicy(BayesUCBPolicy):
    """ The Bernoulli-Bayes-UCB policy.
    - uses a Beta conjugate prior for Bernoulli likelihood.
    - Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def set_label(self, label=None):
        self.label = label or "Gambler-Bayes-UCB (Beta)"

    def evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by taking the :math:`1 - \frac{1}{t}` quantile from the Beta posterior:
        .. math:: I_k(t) = \mathrm{Quantile}\left(\mathrm{Beta}(1 + S_k(t), 1 + N_k(t) - S_k(t)), 1 - \frac{1}{t}\right).
        """
        t = max(1, self.t)
        b = max(1, self.b)

        #original
        #q = (1-(1/t))

        #v1
        q = (1-(1/b))

        #v2
        #q = (1-(1/t)) * (1-(1/b))
        
        #v3
        #factor is near from zero when budget is very low, and approaches 1 when budget is high
        #f = (1-(1/b))
        #f = sqrt(1-(1/b))
        #q = (1-f)*(1-(1/t)) + f*(1-(1/b))

        #i = self.i_last
        for i in range(self.k): 
            #correct alpha and beta parameters (successes and failures) considering reward bounds
            x = max(0.0, (self.s_i[i] - (self.n_i[i] * self.r_min)) / self.r_amp)
            y = max(0.0, self.n_i[i] - x)
            self.v_i[i] = beta.ppf(q, x + self.prior, y + self.prior)

################################################################################

class BernKLUCBPolicy(IndexedPolicy):

    def __init__(self, k, r_min=-1.0, r_max=+1.0, **kwargs):

        super().__init__(k)

        if (label is None) and (type(self) is BernKLUCBPolicy):
            self.label = f"KL-UCB (Bern)"
            if omega is not None:
                self.label = "ST-" + self.label

    #@jit
    def _klBern(self, x, y):
        r""" Kullback-Leibler divergence for Bernoulli distributions.
        .. math:: \mathrm{KL}(\mathcal{B}(x), \mathcal{B}(y)) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}).
        """
        eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]
        x = min(max(x, eps), 1 - eps)
        y = min(max(y, eps), 1 - eps)
        return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))

    #@jit
    def _klucbBern(self, x, d, precision=1e-6):
        """ KL-UCB index computation for Bernoulli distributions, using :func:`klucb`."""
        upperbound = min(1., self._klucbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
        return self._klucb(x, d, upperbound, precision)

    #@jit
    def _klucbGauss(self, x, d, sig2x=0.25):
        """ KL-UCB index computation for Gaussian distributions.
        - Note that it does not require any search.
        .. warning:: it works only if the good variance constant is given.
        .. warning:: Using :class:`Policies.klUCB` (and variants) with :func:`klucbGauss` is equivalent to use :class:`Policies.UCB`, so prefer the simpler version.
        """
        return x + sqrt(abs(2 * sig2x * d))

    #@jit
    def _klucb(self, x, d, upperbound, precision=1e-6, lowerbound=float('-inf'), max_iterations=50):
        r""" The generic KL-UCB index computation.
        - ``x``: value of the cum reward,
        - ``d``: upper bound on the divergence,
        - ``kl``: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
        - ``upperbound``, ``lowerbound=float('-inf')``: the known bound of the values ``x``,
        - ``precision=1e-6``: the threshold from where to stop the research,
        - ``max_iterations=50``: max number of iterations of the loop (safer to bound it to reduce time complexity).
        .. math:: \mathrm{klucb}(x, d) \simeq \sup_{\mathrm{lowerbound} \leq y \leq \mathrm{upperbound}} \{ y : \mathrm{kl}(x, y) < d \}.
        .. note:: It uses a **bisection search**, and one call to ``kl`` for each step of the bisection search.
        For example, for :func:`klucbBern`, the two steps are to first compute an upperbound (as precise as possible) and the compute the kl-UCB index:
        >>> x, d = 0.9, 0.2   # mean x, exploration term d
        >>> upperbound = min(1., klucbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
        """
        v = max(x, lowerbound)
        u = upperbound
        i = 0
        while ((i < max_iterations) and (u - v > precision)):
            i += 1
            m = (v + u) * 0.5
            if self._klBern(x, m) > d:
                u = m
            else:
                v = m
        return (v + u) * 0.5

    def evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:
        .. math:: \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
        U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(t)}{N_k(t)} \right\},\\
        I_k(t) &= U_k(t).
        If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1).
        """
        c = 1.0
        #tolerance = 1e-4
        #i = self.i_last
        for i in range(self.k): 
            n_i = self.n_i[i]
            mu_i = self.mu_i[i]
            if n_i == 0:
                self.v_i[i] = float('+inf')
            else:
                self.v_i[i] = self._klucbBern(mu_i, c * log(self.t) / n_i)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# TO FIX !!!!!
################################################################################

"""
class GaussianBayesUCBPolicy(IndexPolicy):
    # The Gaussian-Bayes-UCB policy.
    #- uses a Normal Inverse Gamma conjugate prior for Normal likelihood.
    #

    def __init__(self, k, v_ini=None, w=1, label=None):
        super().__init__(k, v_ini=v_ini, w=w, label=label)
        if label is None:
            self.label = "Bayes-UCB (Normal)"

    def observe(self, r):
        #Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards, by taking the :math:`1 - \frac{1}{t}` quantile from the Normal-Inverse-Gamma posterior:
        #.. math:: I_k(t) = \mathrm{Quantile}\left(\mathrm{NIG}(1 + S_k(t), 1 + N_k(t) - S_k(t)), 1 - \frac{1}{t}\right).
        #

        super().observe(r)

        #t = self.t
        #q = 1. - (1. / (1 + t))
        t = max(1.0, self.t)
        q = 1 - (1 / t)

        #i = self.i_last
        for i in range(self.k): 
            #q = 1. - (1. / (1 + self.n_i[i]))
            alp = self.s_i[i] + 1
            bet = self.n_i[i] - self.s_i[i] + 1
            self.v_i[i] = beta.ppf(q, alp, bet)

################################################################################

# class for the marab algorithm
class MaRaBPolicy(IndexPolicy):

    def __init__(self, k, v_ini=None, w=1, label=None, alpha=0.05, c=1e-6):
        super().__init__(k, v_ini=v_ini, w=w, label=label)
        self.alpha = alpha
        self.c = c
        self.reward_samples = [np.array([0.0]) for a in range(k)]
        if label is None:
            self.label = f"Empirical-MARAB ($\alpha={self.alpha}$)"

    def reset(self):
        super().reset()
        self.reward_samples = [np.array([0.0]) for a in range(self.k)]

    def observe(self, r):
        super().observe(r)
        self.reward_samples[self.i_last] = np.sort(np.append(self.reward_samples[self.i_last], [r]))
        for i in range(self.k): 
            # calculating empirical cvar
            e = np.ceil(self.alpha*self.n_i[i]).astype(int)
            empirical_cvar = self.reward_samples[i][:e].mean()
            # calculating lower confidence bound
            lcb = np.sqrt(np.log(np.ceil(self.alpha*self.t))/self.n_i[i])
            # adding score to scores list
            self.v_i[i] = empirical_cvar - self.c * lcb

################################################################################


class AlarmedAlphaUCBPolicy(AlphaUCBPolicy, AlarmedPolicy):

    def __init__(self, k, v_ini=None, w=1, label=None, alpha=2.0, b_0=None, omega=None):
        AlphaUCBPolicy.__init__(self, k, v_ini=v_ini, w=w, alpha=alpha, label=label)
        AlarmedPolicy.__init__(self, k, b_0=b_0, omega=omega)
        if label is None:
            self.label = f"Alarmed-UCB($\omega={self.omega}$)"

    def reset(self):
        AlphaUCBPolicy.reset(self)
        AlarmedPolicy.reset(self)

    def observe(self, r):
        AlphaUCBPolicy.observe(self, r)
        AlarmedPolicy.observe(self, r)



class AlarmedBernKLUCBPolicy(BernKLUCBPolicy, AlarmedPolicy):

    def __init__(self, k, v_ini=None, w=1, label=None, b_0=None, omega=1.0):
        BernKLUCBPolicy.__init__(self, k, v_ini=v_ini, w=w, label=label)
        AlarmedPolicy.__init__(self, k, b_0=b_0, w=w)
        self.omega = omega   #safety-critical warning threshold for budget level
        if label is None:
            self.label = f"Alarmed-KL-UCB($\omega={self.omega}$)"

    def reset(self):
        BernKLUCBPolicy.reset(self)
        AlarmedPolicy.reset(self)

    def observe(self, r):
        BernKLUCBPolicy.observe(self, r)
        AlarmedPolicy.observe(self, r)



class AlarmedEpsilonGreedyPolicy(EpsilonGreedyPolicy, AlarmedPolicy):

    def __init__(self, k, w=1, label=None, b_0=None, omega=1.0, eps=0.9):
        EpsilonGreedyPolicy.__init__(self, k, w=w, label=label, eps=eps)
        AlarmedPolicy.__init__(self, k, b_0=b_0, w=w)
        self.omega = omega   #safety-critical warning threshold for budget level
        if label is None:
            self.label = f"Alarmed-$\\epsilon$-greedy($\\epsilon=" + str(round(self.eps,2)) + "\omega=" + str(round(self.omega, 2)) + "$)"

    def reset(self):
        EpsilonGreedyPolicy.reset(self)
        AlarmedPolicy.reset(self)

    def observe(self, r):
        EpsilonGreedyPolicy.observe(self, r)
        AlarmedPolicy.observe(self, r)



#####################################################


class BanditGamblerPolicy(EmpiricalMeansPolicy, BudgetedPolicy):

    def __init__(self, k, v_ini=None, w=1, label=None, b_0=None):
        #super().__init__(k, v_ini=v_ini, w=w, d=d, b_0=b_0)
        EmpiricalMeansPolicy.__init__(self, k, v_ini=v_ini, w=w, label=label)
        BudgetedPolicy.__init__(self, k, b_0=b_0, w=w)
        if label is None:
            self.label = "Bandit-Gambler"

    #@jit
    def ruin_estimated_prob(self, i):
        n_i = self.n_i[i]
        x_i = self.s_i[i]
        y_i = n_i - self.s_i[i]
        b = max(1.0, self.b)
        return beta.cdf(0.5, x_i+1, y_i+1) + integral(lambda p, x, y, b : ((1-p)/p)**b * beta.pdf(p, x+1, y+1), 0.5, 1.0, (x_i, y_i, b))[0]

    def surv_estimated_prob(self, i):
        n_i = self.n_i[i]
        x_i = self.s_i[i]
        y_i = n_i - x_i
        b = max(1.0, self.b)
        return integral(lambda p, x, y, b : (1-((1-p)/p)**b) * beta.pdf(p, x+1, y+1), 0.5, 1.0, (x_i, y_i, b))[0]

    def reset(self):
        #super().reset()
        EmpiricalMeansPolicy.reset(self)
        BudgetedPolicy.reset(self)

    def _update(self, r):
        #super()._update(r)
        EmpiricalMeansPolicy._update(self, r)
        BudgetedPolicy._update(self, r)

    def _evaluate(self):
        i = self.i_last
        #self.v_i[i] = 1.0 - self.ruin_estimated_prob(i)
        self.v_i[i] = self.surv_estimated_prob(i)


################################################################################

class BanditGamblerUCBPolicy(BanditGamblerPolicy):

    def __init__(self, k, v_ini=None, w=1, label=None, b_0=None):
        super().__init__(k, v_ini=v_ini, w=w, label=label)
        Budgeted.__init__(self, k, b_0=b_0, w=w)
        if label is None:
            self.label = "Bandit-Gambler-UCB"

    def _evaluate(self):
        for i in range(self.k):
            self.v_i[i] = 1.0 - self.ruin_estimated_prob(i)

    def ruin_estimated_prob(self, i):
        b = max(1.0, self.b)
        factor = np.log(self.t)/self.t
        n_i = self.n_i[i]
        x_i = self.s_i[i]
        y_i = n_i - self.s_i[i]
        return beta.cdf(0.5, x_i+1, y_i+1) + integral(lambda p, x, y, b : ((1-p)/p)**b * beta.pdf(p, x*factor+1, y*factor+1), 0.5, 1.0, (x_i, y_i, b))[0]

################################################################################

class PositiveGamblerUCB(EmpiricalMeansPolicy, Budgeted):

    def __init__(self, k, v_ini=None, w=1, label=None, b_0=None):
        #super().__init__(k, v_ini=v_ini, w=w, d=d, b_0=b_0)
        EmpiricalMeansPolicy.__init__(self, k, v_ini=v_ini, w=w, label=label)
        Budgeted.__init__(self, k, b_0=b_0, w=w)
        if label is None:
            self.label = "Positive-Gambler"

    def reset(self):
        #super().reset()
        EmpiricalMeansPolicy.reset(self)
        Budgeted.reset(self)

    def _update(self, r):
        #super()._update(r)
        EmpiricalMeansPolicy._update(self, r)
        Budgeted._update(self, r)

    def _evaluate(self):
        t = self.t
        b = max(1.0, self.b)
        for i in range(self.k):
            n_i = self.n_i[i]
            mu_i = self.mu_i[i]
            x_i = self.s_i[i]
            y_i = n_i - self.s_i[i]
            if self.n_i[i] == 0:
                self.v_i[i] = float('+inf')
            else:
                self.v_i[i] = 1 - beta.cdf(0.5, x_i+1, y_i+1) + sqrt((2 * log(b)) / n_i)

################################################################################

class BGP(BanditGamblerPolicy):

    #@jit
    def surv_estimated_prob(self, i):
        n_i = self.n_i[i]
        x_i = self.s_i[i]
        y_i = n_i - x_i
        b = max(1.0, self.b)
        #return integral(lambda p, x, y, b : (1-((1-p)/p)**b) * beta.pdf(p, x+1, y+1), 0.5, 1.0, (x_i, y_i, b))[0]
        return integral(lambda p, x, y, b : p * beta.pdf(p, x+1, y+1), 0.5, 1.0, (x_i, y_i, b))[0]    #simplify to linear

################################################################################

class BG_UCB(BanditGamblerPolicy):

    def __init__(self, k, v_ini=None, w=1, label=None, b_0=None):
        super().__init__(k, v_ini=v_ini, w=w, label=label, b_0=b_0)
        if label is None:
            self.label = "Bandit-Gambler-UCB"

    def ruin_estimated_prob(self, i):
        n_i = self.n_i[i]
        x_i = self.s_i[i]
        y_i = n_i - x_i
        b = max(1.0, self.b)
        factor = np.log(self.t)/self.t
        return beta.cdf(0.5, x_i+1, y_i+1) + integral(lambda p, x, y, b : ((1-p)/p)**b * beta.pdf(p, x*factor+1, y*factor+1), 0.5, 1.0, (x_i, y_i, b))[0]

################################################################################

class BG_Pos(BanditGamblerPolicy):

    def __init__(self, k, v_ini=None, w=1, label=None, b_0=None):
        super().__init__(k, v_ini=v_ini, w=w, label=label, b_0=b_0)
        if label is None:
            self.label = "Positive-Gambler"

    def _evaluate(self):
        i = self.i_last
        n_i = self.n_i[i]
        mu_i = self.s_i[i] / n_i
        t = self.t
        x_i = self.s_i[i]
        y_i = n_i - self.s_i[i]
        b = max(1.0, self.b)
        if self.n_i[i] == 0:
            self.v_i[i] = float('+inf')
        else:
            self.v_i[i] = 1 - beta.cdf(0.5, x_i+1, y_i+1) + sqrt((2 * log(b)) / n_i)

"""