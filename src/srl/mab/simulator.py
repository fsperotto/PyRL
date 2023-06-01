#Import Dependencies
import numpy as np
from numpy.random import binomial, randint, uniform, rand, normal, choice
import random
from random import choices
import numpy.ma as ma
from math import sqrt, log
from scipy.stats import beta, norm
from scipy.integrate import quadrature as integral
from scipy.ndimage.filters import uniform_filter1d
from collections import Iterable
from math import sqrt, log
from numba import jit
from itertools import accumulate as acc
from multiprocess import Pool
import psutil
from tqdm.notebook import tqdm
from IPython.display import display
import matplotlib.pyplot as plt


################################################################################


class SMAB():
    """ Base survival MAB process. """

    def __init__(self, arms, algs, h, b_0, n=1, w=1, r_min=0.0, r_max=1.0, run=False, prev_draw=True, use_multiprocess=False, save_only_means=True):
        """
         A : List of Arms
         G : List of Algorithms
         h : max time-horizon
         d : rewards domain
         n : number of repetitions
         w : number of times each arm must be played at beginning
         b_0 : initial budget
        """

        #domain of rewards ( by default on [0, 1] )
        if r_min > r_max:
            r_min, r_max = r_max, r_min
        self.r_min = r_min
        self.r_max = r_max
        self.r_amp = r_max - r_min

        #time-horizon (0, 1 ... t ... h)
        self.h = h   #time-horizon
        self.T = range(self.h)          #range for time (0 ... h-1)
        self.T1 = range(1, self.h+1)    #range for time (1 ... h)
        self.T01 = range(0, self.h+1)   #range for time (0, 1 ... h)

        # A : arms (1 ... i ... k)   - i or idx_arm
        self.arms = arms if isinstance(arms, Iterable) else [arms]

        #number of arms
        self.k = len(self.arms)
        self.K = range(self.k)          #range for arms (0 ... k-1)
        self.K1 = range(1,self.k+1)     #range for arms (1 ... k)

        #arms properties
        self.mu_i = np.array([a.mean for a in self.arms]) # * self.d.r_amp + self.d.r_min     #means
        self.i_star = np.argmax(self.mu_i)                #best arm index
        self.i_worst = np.argmin(self.mu_i)               #worst arm index
        self.mu_star = np.max(self.mu_i)                  #best mean
        self.mu_worst = np.min(self.mu_i)                 #worst mean

        #budget
        self.b_0 = b_0   
        
        #algorithms (1 ... g ... m)
        self.algs = algs if isinstance(algs, Iterable) else [algs]
        self.m = len(self.algs)

        #repetitions (1 ... j ... n)   - j or idx_rep
        self.n = n

        #window
        self.w = w
        
        self.save_only_means = save_only_means
        self.prev_draw = prev_draw
        self.use_multiprocess = use_multiprocess

        #run
        if run:
            self.run()


    def run_episode(self, g, drawn_reward_i_t=None):
        # g is the index of the tested algorithm
        # j is the index of the current repetition
        # drawn_reward_i_t is the previous draw result, if prev_draw
        # complete the matrices H and R for the row g, j

        #get the algorithm
        alg = self.algs[g]
        
        # Initialize
        alg.reset()

        # Allocate matrix for Rewards and History of selected Actions (2d matrices [t x g])
        received_rewards_t = np.empty((self.h), dtype=float)  #successes
        chosen_actions_t = np.empty((self.h), dtype=int)      #history of actions
        
        # Loop on time
        #for t in tqdm(self.T, desc=tqdm_desc_it, leave=tqdm_leave, disable=(tqdm_disable or self.n > 1 or self.m > 1) ):
        for t in range(self.h):
            # The algorithm chooses the arm to play
            i = alg.choose()
            #no choice, no action
            if i == -1:
                x = 0.0
            else:
                # The arm played gives reward
                if drawn_reward_i_t is not None:
                    x = drawn_reward_i_t[i, t]
                else:
                    x = self.A[i].draw()
            # The reward is returned to the algorithm
            alg.observe(x)
            # Save both
            received_rewards_t[t] = x
            chosen_actions_t[t] = i
            #self.received_rewards_j_g_t[j, g, t] = x
            #self.chosen_actions_j_g_t[j, g, t] = i
            
        return received_rewards_t, chosen_actions_t

    
    def run_repetition(self, j, random_seed=None):

        # Allocate matrix for Rewards and History of selected Actions (2d matrices [t x g])
        received_rewards_g_t = np.empty((self.m, self.h), dtype=float)  #successes
        chosen_actions_g_t = np.empty((self.m, self.h), dtype=int)    #history of actions
        
        drawn_reward_i_t = None
        if self.prev_draw:
            seed_t = rand(self.h)     #luck is the same for every arm in a same round
            drawn_reward_i_t = np.array([arm.convert(chances_arr=seed_t) for arm in self.arms]) #seed to reward

        # For each algorithm
        #for g, alg in enumerate(tqdm(self.algs, desc=tqdm_desc_alg, leave=tqdm_leave, disable=(tqdm_disable or self.m == 1))):
        for g in range(self.m):
            received_rewards_g_t[g], chosen_actions_g_t[g] = self.run_episode(g, drawn_reward_i_t)

        return received_rewards_g_t, chosen_actions_g_t
            
        
    def run(self, tqdm_desc_it="iterations", tqdm_desc_alg="algorithms", tqdm_desc_rep="repetitions", tqdm_leave=False, tqdm_disable=False, num_threads=1, smooth_window=None):

        #time-horizon (1 ... t ... h)
        #arms (1 ... i ... k)
        #repetitions (1 ... j ... n)
        #algorithms (1 ... g ... m)

        num_cpus = psutil.cpu_count(logical=False)        
        
        # Allocate Rewards and History of selected Actions (3d matrices [t x g x j])
        # R : history of rewards for each repetition j, alg g, round t
        self.received_rewards_j_g_t = np.empty((self.n, self.m, self.h), dtype=float)   
        # H : history of actions for each repetition j, alg g, round t
        self.chosen_actions_j_g_t = np.empty((self.n, self.m, self.h), dtype=int)

        # previously draw the reward for every arm at each round,
        # then all algorithms can experience the same chance
        #previous_draw_t_j = None
        #if prev_draw:
        #    previous_draw_t_j = rand(self.h, self.n)     #luck is the same for every arm in a same round and repetition

        # For each repetition
        #for j in tqdm(range(self.n), desc=tqdm_desc_rep, leave=(tqdm_leave and self.m == 1), disable=(tqdm_disable or self.n == 1)):
        #for j in tqdm(range(self.n), desc=tqdm_desc_rep, leave=tqdm_leave, disable=(tqdm_disable or self.n == 1)):
        
        if self.use_multiprocess:
            
            #with Pool(num_cpus) as pool:
            #    for j in range(self.n):
            #        pool.apply_async(self.run_repetition, args=(j), callback=get_repetition_result)
    
            #WORKING!!!!    
            #with Pool(num_cpus) as pool:
            #        resulting_map = pool.map(self.run_repetition, range(self.n))
            #    self.received_rewards_j_g_t = np.array([v[0] for v in resulting_map])
            #    self.chosen_actions_j_g_t = np.array([v[1] for v in resulting_map])
            
            with Pool(num_cpus) as pool:
                for j, result in enumerate(tqdm(pool.imap(self.run_repetition, range(self.n)), total=self.n)):
                    self.received_rewards_j_g_t[j], self.chosen_actions_j_g_t[j] = result
            
            #with Pool(num_cpus) as pool:
            #    #self.received_rewards_j_g_t[j], self.chosen_actions_j_g_t[j] = pool.map(self.run_repetition, range(self.n))
            #    #tqdm(p.imap(self.run_repetition, range(self.n)))
            #    for j in range(self.n):
            #        self.received_rewards_j_g_t[j], self.chosen_actions_j_g_t[j] = pool.apply(self.run_repetition, (j))
            #    #multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
                
        else:
            for j in tqdm(range(self.n)):
                self.received_rewards_j_g_t[j], self.chosen_actions_j_g_t[j] = self.run_repetition(j)
            

        #simple names
        R = self.received_rewards_j_g_t
        H = self.chosen_actions_j_g_t

        #the rewards at beginning are forced to zero if w>0
        #R[:,:,:self.w*self.k] = 0.0
        
        #Translate Rewards following Domain
        #R = X * self.d.r_amp + self.d.r_min
        X = R
        
        #actions history, with initial action index being 1, not 0
        self.H1 = H+1

        #actions map (bool 4d matrix)
        H_a = np.array([[[[True if (self.H1[j,g,t]==i) else False for t in self.T] for i in self.K1] for g in range(self.m)] for j in range(self.n)], dtype='bool')

        #progressive actions count (int 4d matrix [t x j x i x a])
        self.N_a = np.cumsum(H_a, axis=3)

        #averaged progressive actions count (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MN_a = np.mean(self.N_a, axis=0)

        #progressive actions frequency (float 4d matrix [t x j x i x a])
        self.F_a = self.N_a / self.T1

        #averaged progressive actions frequency (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MF_a = np.mean(self.F_a, axis=0)

        #final arm pull count (int 3d matrix [j x i x a])
        #n_a = N_a[:,:,:,self.h-1]
        n_a = self.N_a[:,:,:,-1]

        #averaged final arm pull count (float 2d matrix [j x a]) #averaged over repetitions
        self.mn_a = np.mean(n_a, axis=0)

        #final arm pull frequency (float 3d matrix [j x i x a])
        self.f_a = self.F_a[:,:,:,-1]

        #averaged final arm pull frequency (float 2d matrix [j x a]) #averaged over repetitions
        self.mf_a = np.mean(self.f_a, axis=0)

        #progressive cumulative rewards (float 3d matrix [t x j x i])
        self.SR = np.cumsum(R, axis=2, dtype='float')

        #averaged progressive cumulative rewards (float 2d matrix [t x j]) #averaged over repetitions
        self.average_cumulative_rewards_jt = self.MSR = np.mean(self.SR, axis=0)

        #final rewards (float 2d matrix [j x i])
        self.sr = self.SR[:,:,-1]

        #averaged final rewards (float 1d matrix [j]) #averaged over repetitions
        self.msr = np.mean(self.sr, axis=0)
        #and standard deviation
        self.dsr = np.std(self.sr, axis=0)

        #progressive average rewards (float 3d matrix [t x j x i]) #averaged over time
        self.MR = self.SR / self.T1

        #averaged progressive average rewards (float 2d matrix [t x j]) #averaged over time and repetitions
        self.averaged_mean_reward_jt = self.MMR = np.mean(self.MR, axis=0)

        #regret (float 3d matrix [t x j x i])
        L = self.mu_star - R

        #averaged regret (float 2d matrix [t x j])
        #self.ML = np.mean(L, axis=0)
        #progressive average regret (float 3d matrix [t x j x i]) #averaged over time
        self.ML = self.mu_star - self.MR

        #averaged average regret (float 2d matrix [t x j]) #averaged over time and repetitions
        self.average_mean_regret_jt = self.MML = np.mean(self.ML, axis=0)
        #self.average_mean_regret_jt = self.MML  = self.mu_star - self.MMR
        
        #cumulated regret (float 3d matrix [t x j x i])
        self.SL = np.cumsum(L, axis=2, dtype='float')

        #averaged cumulated regret (float 2d matrix [t x j]) #averaged over repetitions
        self.average_cumulative_regret_jt = self.MSL = np.mean(self.SL, axis=0)

        #final cumulated regret (float 2d matrix [j x i])
        sl = self.SL[:,:,-1]

        #averaged final cumulated regret (float 1d matrix [j]) #averaged over repetitions
        self.msl = np.mean(sl, axis=0)
        #and standard deviation
        self.dsl = np.std(sl, axis=0)
        
        #rewards map (float 4d matrix [t x j x i x a])
        R_a = np.array([[[[R[j,g,t] if (self.H1[j,g,t]==i) else 0.0 for t in self.T] for i in self.K1] for g in range(self.m)] for j in range(self.n)], dtype='float')

        #averaged rewards map (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MR_a = np.mean(R_a, axis=0)

        #progressive rewards map (int 4d matrix [t x j x i x a])
        SR_a = np.cumsum(R_a, axis=3)

        #averaged progressive rewards map (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MSR_a = np.mean(SR_a, axis=0)

        #final rewards per action (float 3d matrix [j x i x a])
        sr_a = SR_a[:,:,:,-1]

        #averaged final rewards per action (float 2d matrix [j x a]) #averaged over repetitions
        self.msr_a = np.mean(sr_a, axis=0)

        #reward proportion per action (float 3d matrix [j x i x a])
        self.fr_a = sr_a / self.SR[:,:,-1,np.newaxis]

        #averaged proportion per action (float 2d matrix [j x a]) #averaged over repetitions
        self.mfr_a = np.mean(self.fr_a, axis=0)

        #progressive budget (float 3d matrix [t x j x i])
        # i.e. the progressive cumulative rewards plus initial budget
        B = self.SR + self.b_0

        ##progressive on negative counter of episodes (float 3d matrix [t x j])
        ## i.e. the number of episodes where, at each time t, alg j is running on negative budget
        #N = np.sum(B >= 0, axis=0)

        #averaged progressive budget (float 2d matrix [t x j]) #averaged over repetitions
        #self.MB = np.mean(B, axis=0)
        self.MB = self.MSR + self.b_0

        #final budget (float 2d matrix [j x i])
        b = B[:,:,-1]

        #averaged final budget (float 1d matrix [j]) #averaged over repetitions
        self.mb = np.mean(b, axis=0)

        #time map on non-positive budget (int 3d matrix [t x j x i])
        #TNB = np.array([[[1 if(v<=0) else 0 for v in B_ij] for B_ij in B_i] for B_i in B])
        TNB = (B <= 0).astype(int)
        
        #time dead map (int 3d matrix [t x j x i])
        TD = np.maximum.accumulate(TNB, axis=2)


        #time alive map (int 3d matrix [t x j x i])
        TS = 1 - TD

        #progressive death counter of episodes (float 3d matrix [t x j])
        DC = np.sum(TD, axis=0)

        #final death counter
        dc = DC[:,-1]

        #progressive survival rate of episodes (float 3d matrix [t x j])
        #MS = 1 - np.mean(TD, axis=0)
        self.MS = np.mean(TS, axis=0)

        #final survival counter
        self.ms = self.MS[:,-1]

        ####################################################################

        #history considering ruin (zero after ruin)
        self.H1R = np.multiply(self.H1, TS)
        HR_a = np.array([[[[True if (self.H1R[j,g,t]==i) else False for t in self.T] for i in self.K1] for g in range(self.m)] for j in range(self.n)], dtype='bool')
        self.NR_a = np.cumsum(HR_a, axis=3)
        self.MNR_a = np.mean(self.NR_a, axis=0)
        self.FR_a = self.NR_a / self.T1
        self.MFR_a = np.mean(self.FR_a, axis=0)

        ####################################################################

        #progressive budget considering ruin (float 3d matrix [t x j x i])
        # i.e. the progressive cumulative rewards plus initial budget
        #_RB = ma.masked_less_equal(_B, 0.0).filled(0.0)
        #_RB = np.maximum(B, 0.0)
        RB = np.multiply(B, TS)

        self.MRB = np.mean(RB, axis=0)

        ####################################################################

        #rewards considering ruin (zero after ruin)
        RR = np.multiply(R, TS)
        
        #progressive cumulative rewards (float 3d matrix [t x j x i])
        self.SRR = np.cumsum(RR, axis=2, dtype='float')

        #progressive average rewards (float 3d matrix [t x j x i]) #averaged over time
        self.MRR = self.SRR / self.T1

        #averaged progressive average rewards (float 2d matrix [t x j]) #averaged over time and repetitions
        self.MMRR = np.mean(self.MRR, axis=0)
        
        ####################################################################
        
        #progressive penalized mean budget (float 3d matrix [t x j x i])
        # i.e. the progressive mean budget multiplied by survival rate
        self.MPB = np.multiply(self.MB, self.MS)

        #empirical relative regret
        #self.MRL = 1 - (self.MRB / self.MRB[self.oracle_idx,:])
        self.MRL = 1 - (self.MRB / self.mu_star)
        
        #immediate immortal reward (averaged over episodes)
        self.avgR = np.mean(R, axis=0)

        #immediate mortal reward (averaged over episodes)
        self.avgRR = np.mean(RR, axis=0)

            
    """ 
    Plot a line graph
    """
    def plot(self, Y, X=None, 
             algs_idx=None, h=None, 
             line_properties=None, #dict: label, linestyle, linecolor, linewidth, marker, markercolor, markersize, markevery, alpha, visible...
             xlabel="$t$", ylabel=None, reorder='desc', showlast='legend', title=None, 
             filename=None, figsize=None, show=True, smooth_window=None, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
            #fig = plt.figure(figsize=figsize)
        
        if isinstance(Y, str):
            if (Y=='immortal_precision'):
                X = self.T1
                Y = self.MF_a[:,self.i_star]
                if ylabel is None:
                    ylabel = 'immortal precision (averaged over repetitions, does not stop on ruin)'
                if title is None:
                    title="Precision (without ruin)"
            elif (Y=='mortal_precision'):
                X = self.T1
                Y = self.MFR_a[:,self.i_star]
                if ylabel is None:
                    ylabel = 'mortal precision (averaged over repetitions, stop on ruin)'
                if title is None:
                    title="Precision (considering ruin)"
            elif Y=='sum_reward':
                X = self.T01
                Z = np.reshape(np.zeros(self.m, dtype='float'), [self.m, 1])
                Y = np.block([Z, self.MSR])
                if ylabel is None:
                    ylabel = 'immortal cumulated reward (averaged over repetitions, does not stop on ruin)'
                if title is None:
                    title="Cumulated Reward (without ruin)"
            elif Y=='immortal_budget':
                X = self.T01
                Z = np.reshape(np.repeat(self.b_0, self.m), [self.m, 1])
                Y = np.block([Z, self.MB])
                if ylabel is None:
                    ylabel = 'immortal budget (averaged over repetitions, does not stop on ruin)'
                if title is None:
                    title="Budget (without ruin)"
            elif Y=='mortal_budget':
                X = self.T01
                Z = np.reshape(np.repeat(self.b_0, self.m), [self.m, 1])
                Y = np.block([Z, self.MRB])
                if ylabel is None:
                    ylabel = 'mortal budget (averaged over repetitions, zero if ruin)'
                if title is None:
                    title="Budget (considering ruin)"
            elif Y=='penalized_budget':
                X = self.T01
                Z = np.reshape(np.repeat(self.b_0, self.m), [self.m, 1])
                Y = np.block([Z, self.MPB])
                if ylabel is None:
                    ylabel = 'penalized budget (averaged over repetitions, times survival rate)'
                if title is None:
                    title="Penalized Budget"
            elif Y=='survival':
                X = self.T01
                Z = np.reshape(np.ones(self.m, dtype='float'), [self.m, 1])
                Y = np.block([Z, self.MS])
                if ylabel is None:
                    ylabel = 'survival rate'
                if title is None:
                    title="Survival Rate"
            elif Y=='avg_reward':
                X = self.T1
                Y = self.MMR
                if ylabel is None:
                    ylabel = 'mean reward per step (averaged over repetitions, without ruin)'
                if title is None:
                    title="Mean Reward (without ruin)"
            elif Y=='avg_mortal_reward':
                X = self.T1
                Y = self.MMRR
                if ylabel is None:
                    ylabel = 'mean reward per step (averaged over repetitions, considering ruin)'
                if title is None:
                    title="Mean Reward (considering ruin)"
            elif Y=='sum_regret':
                X = self.T1
                Y = self.MSL
                if ylabel is None:
                    ylabel = 'cumulated regret (averaged over repetitions)'
                if title is None:
                    title="Cumulated Regret (without ruin)"
            elif Y=='avg_regret':
                X = self.T1
                Y = self.MML
                if ylabel is None:
                    ylabel = 'mean regret per step (averaged over repetitions)'
                if title is None:
                    title="Mean Regret (without ruin)"
            else:
                print('Unknown plot.')
                return

        #if showing partial horizon
        if h is not None and h < self.h:
            Y = Y[:,:h]

        #if showing specific algorithms
        if algs_idx is not None:
            if isinstance(algs_idx, int):
                algs_idx = [algs_idx]
            Y = Y[algs_idx]
        else:
            algs_idx = range(len(Y))
        
        if line_properties is None:
            line_properties = [{} for _ in algs_idx]
            
        #line_properties = np.pad(np.array(line_properties[:m]), (0, max(0, m-len(line_properties))), mode='empty')
        for i, j in enumerate(algs_idx):
            line_properties[i].update(self.algs[j].style)            
            if 'label' not in line_properties[i]:
                line_properties[i]['label'] = str(self.algs[j])
            if 'antialiased' not in line_properties[i]:
                line_properties[i]['antialiased'] = True

                    
        line_properties = np.array(line_properties)
        
        #ordering
        if reorder is not None:
            idx=np.argsort(Y[:,-1])
            if reorder == 'desc':
                idx = idx[::-1]
            Y=Y[idx]
            line_properties = line_properties[idx]

        line_properties = line_properties.tolist()
        
        if X is None:
            X = range(len(Y[0]))

        if smooth_window is not None and smooth_window > 1:
            Y = np.array([uniform_filter1d(Y_g, size=smooth_window, mode='nearest') for Y_g in Y])

        for i, Y_g in enumerate(Y):
            if (showlast == 'legend') or (showlast == 'both') or (showlast == True):
                line_properties[i]['label'] = line_properties[i]['label'] + " [" + str(round(Y_g[-1],2)) + "]"
            line, = ax.plot(X, Y_g, **line_properties[i])
            if (showlast == 'axis') or (showlast == 'both') or (showlast == True):
                ax.annotate('%0.2f'%Y_g[-1], xy=(1, Y_g[-1]), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')                    
            
        ax.legend()

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        if filename is not None:
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()


    """ 
    Plot the graph
    """
    def _call_plot(self, xlabel=None, ylabel=None, title=None, names=None, filename=None, show=True):

        if names is not None:
            plt.legend(names)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        if filename is not None:
            plt.tight_layout()
            plt.savefig(filename)

        if show:
            plt.show()


    """ 
    Plot a line graph
    """
    def _plot_progression(self, Y, X=None, names=None, linestyles=None, linecolors=None, xlabel="$t$", ylabel="Value", reorder='desc', showlast=False, title=None, filename=None, show=True):

        if Y.ndim > 1:

            #ordering
            if reorder is not None:
                idx=np.argsort(Y[:,-1])
                if reorder == 'desc':
                    idx = idx[::-1]
                Y=Y[idx]
                if names is not None:
                    names=np.array(names)[idx]
                if linestyles is not None:
                    linestyles=np.array(linestyles)[idx]
                if linecolors is not None:
                    linecolors=np.array(linecolors)[idx]

            if X is None:
                X = range(len(Y[0]))

            for i, Y_i in enumerate(Y):
                line, = plt.plot(X, Y_i)
                if linestyles is not None:
                    line.set_linestyle(linestyles[i % len(linestyles)])
                if linecolors is not None:
                    line.set_color(linecolors[i % len(linecolors)])
                if showlast:
                    plt.annotate('%0.2f'%Y_i[-1], xy=(1, Y_i[-1]), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')                    

        else:
            if X is None:
                X = range(len(Y))
            plt.plot(X, Y)

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, names=names, filename=filename, show=show)


    """ 
    Plot a bar graph (series correspond to arms)
    """
    def _plot_comp_arms(self, Y, E=None, names=None, xlabel="Arms", ylabel=None, title="Arms Comparison", filename=None, show=True):

        if Y.ndim > 1:
            w = 0.8 / len(Y)
            for i, Y_i in enumerate(Y):
                yerr = None if E==None else E[i]
                plt.bar(self.K1 + w*i, Y_i, yerr=yerr, width=w)   #color=plt.cm.get_cmap(i)
        else:
            plt.bar(self.K1, Y, yerr=E)

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, names=names, filename=filename, show=show)


    """ 
    Plot the history (temporal map) of actions taken (series correspond to arms)
     i : repetition
     j : algorithm 
    """
    def plot_history(self, i=0, j=0, xlabel='$t$', ylabel='Arm', title='History of pulled arms', alpha=0.5, markersize=None, filename=None, show=True):
    
        if (i is None) or (j is None):
            print('History of pulls cannot be shwon for average.')
            return None

        plt.plot(self.T1, self.H1[i,j], 'o', markersize=markersize, alpha=alpha)

        plt.yticks(self.K1)
        plt.ylim([0.5, self.k+0.5])
        plt.gca().invert_yaxis()    

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)


    """ 
    Plot the progression of the actions counter (series correspond to arms)
     i : repetition (None for an average over repetitions)
     j : algorithm 
    """
    def plot_action_count_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Number of pulls", title="Arm pull counter", linestyles=None, linecolors=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #prepare labels
        if names is None:
            names = [f"$N_{a}$" for a in self.K1]

        #prepare data
        if i is None:
            Y = self.MN_a[j]
        else:
            Y = self.N_a[i,j]

        #add zeros at time zero
        Z = np.reshape(np.zeros(self.k, dtype='int'), [self.k, 1])
        Y = np.block([Z, Y])

        #call plot progression
        self._plot_progression(Y, X=self.T01, names=names, ylabel=ylabel, xlabel=xlabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    """ 
    Plot the progression of the actions frequency (series correspond to arms)
     i : repetition (None for an average over repetitions)
     j : algorithm 
    """
    def plot_action_freq_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Pull Frequency", title="Arm Selection Frequency", linestyles=None, linecolors=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #prepare labels
        if names is None:
            names = [f"$F_{a}$" for a in self.K1]

        #prepare data
        if i is None:
            Y = self.MF_a[j]  #averaged over repetitions
        else:
            Y = self.F_a[i,j]  

        #call plot progression
        self._plot_progression(Y, X=self.T1, names=names, ylabel=ylabel, xlabel=xlabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    """ 
    Plot the progression of the precision (frequency of the best arm)
     i : repetition (None for an average over repetitions)
     j : algorithm (None for all)
    """
    def plot_precision_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Best Arm Pull Frequency", title="Precision", linestyles=None, linecolors=None, show=True):

        #all algorithms
        if j is None:
            if names is None:
                names=[str(g) for g in self.algs]		
            if i is None:  #average over repetitions
                Y = self.MF_a[:,self.i_star]
            else:  #specific repetition
                Y = self.F_a[i,:,self.i_star]
        #specific algorithm
        else:
            if names is None:
                names=['best', 'others']		
            #prepare best and others frequencies
            if i is None:
                Y = np.array([self.MF_a[j, self.i_star], 1-self.MF_a[j, self.i_star]])
            else:
                Y = np.array([self.F_a[i, j, self.i_star], 1-self.F_a[i, j, self.i_star]])

        #call plot progression
        self._plot_progression(Y, X=self.T1, names=names, ylabel=ylabel, xlabel=xlabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    """ 
    Plot the progression of the cumulated reward (sum of rewards over time)
     i : repetition (None for an average over repetitions)
     j : algorithm (None for all)
    """
    def plot_cumulated_reward_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Cumulated Reward", title="Cumulated Reward", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = self.MSR
            else:		   #in a specific repetition
                Y = self.SR[i]
            Z = np.reshape(np.zeros(self.m, dtype='float'), [self.m, 1])
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = self.MSR[j]
            else: 		   #in a specific repetition
                Y = self.SR[i,j]
            Z = np.array([0.0], dtype='float')

        #add zeros at time zero
        Y = np.block([Z, Y])

        #call plot		
        self._plot_progression(Y, X=self.T01, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_budget_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Budget", title="Budget", linestyles=None, linecolors=None, show=True):		

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = self.MB
            else:		   #in a specific repetition
                Y = self.B[i]
            Z = np.reshape(np.repeat(self.b_0, self.m), [self.m, 1])
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = self.MB[j]
            else: 		   #in a specific repetition
                Y = self.B[i,j]
            Z = np.array([self.b_0], dtype='float')

        #add zeros at time zero
        Y = np.block([Z, Y])

        #call plot
        self._plot_progression(Y, X=self.T01, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)

        
    def plot_survival_progression(self, j=None, names=None, xlabel="$t$", ylabel="Survival", title="Survival", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            Z = np.reshape(np.ones(self.m, dtype='float'), [self.m, 1])
            Y = self.SC
            if names is None:
                names=[str(g) for g in self.algs]
        else:	#specific algorithm		   
            Z = np.array([1.0], dtype='float')
            Y = self.A[j]
            if names is None:
                names=[str(self.algs[j])]

        #add zeros at time zero
        Y = np.block([Z, Y])

        #call plot		
        self._plot_progression(Y, X=self.T01, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)

        
    def plot_negative_budget_time_map(self, i=None, j=None, names=None, inibudget=0.0, xlabel="$t$", ylabel="Time on Negative Budget", title="Time on Negative Budget", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = self.TNMB #MSR
            else:		   #in a specific repetition
                Y = self.TNB[i]
            ##for i, Y_i in enumerate(Y):
            ##	Y[i] = np.array([1 if(v<inibudget) else 0 for v in Y_i])
            #Y = np.array([[1 if(v<-inibudget) else 0 for v in Y_i] for Y_i in Y])
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = self.TNMB[j]
            else: 		   #in a specific repetition
                Y = self.TNB[i,j]
            #Y = np.array([1 if(v<-inibudget) else 0 for v in Y])
        #call plot		
        self._plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_negative_budget_time_progression(self, i=None, j=None, names=None, inibudget=0.0, xlabel="$t$", ylabel="Cumulated Time on Negative Budget", title="Cumulated Time on Negative Budget", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = self.STNMB #MSR
            else:		   #in a specific repetition
                Y = self.STNB[i]
            #Y = np.array([[-1 if(v<-inibudget) else 0 for v in Y_i] for Y_i in Y])
            #Y = np.cumsum(Y, axis=1, dtype='float')
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = self.STNMB[j]
            else: 		   #in a specific repetition
                Y = self.STNB[i,j]
            #Y = np.array([-1 if(v<-inibudget) else 0 for v in Y])
            #Y = np.cumsum(Y, axis=0, dtype='float')

        #call plot		
        self._plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_negative_budget_progression(self, i=None, j=None, names=None, inibudget=0.0, xlabel="$t$", ylabel="Negative Budget", title="Negative Budget", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = self.NMB
            else:		   #in a specific repetition
                Y = self.NB[i]
            #Y = np.array([[v+inibudget if(v<-inibudget) else 0 for v in Y_i] for Y_i in Y])
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = self.NMB[j]
            else: 		   #in a specific repetition
                Y = self.NB[i,j]
            #Y = np.array([v+inibudget if(v<-inibudget) else 0 for v in Y])

        #call plot		
        self._plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)

    def plot_cumulated_negative_budget_progression(self, i=None, j=None, names=None, inibudget=0.0, xlabel="$t$", ylabel="Cumulated Negative Budget", title="Cumulated Negative Budget", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = self.SNMB
            else:		   #in a specific repetition
                Y = self.SNB[i]
            #Y = np.array([[v+inibudget if(v<-inibudget) else 0 for v in Y_i] for Y_i in Y])
            #Y = np.cumsum(Y, axis=1, dtype='float')
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = self.SNMB[j]
            else: 		   #in a specific repetition
                Y = self.SNB[i,j]
            #Y = np.array([v+inibudget if(v<-inibudget) else 0 for v in Y])
            #Y = np.cumsum(Y, axis=0, dtype='float')

        #call plot		
        self._plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_average_reward_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Average Reward", title="Average Reward", linestyles=None, linecolors=None, show=True):	

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = self.MMR
            else:		   #in a specific repetition
                Y = self.MR[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = self.MMR[j]
            else: 		   #in a specific repetition
                Y = self.MR[i,j]

        #call plot		
        self._plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_cumulated_reward_regret_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Cumulated Reward and Regret", title="Cumulated Reward and Regret", show=True):	

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y1 = self.MSR
                Y2 = self.MSL
            else:		   #in a specific repetition
                Y1 = self.SR[i]
                Y2 = self.SL[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y1 = self.MSR[j]
                Y2 = self.MSL[j]
            else: 		   #in a specific repetition
                Y1 = self.SR[i,j]
                Y2 = self.SL[i,j]

        #call plot		
        self._plot_progression(Y1, X=self.T1, names=None, xlabel=None, ylabel=None, title=None, show=False)
        self._plot_progression(Y2, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show, linestyles=['--']*self.m, linecolors=['C' + str(j) for j in range(self.m)])


    def plot_average_reward_regret_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Average Reward and Regret", title="Average Reward and Regret", show=True):	

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y1 = self.MMR
                Y2 = self.MML
            else:		   #in a specific repetition
                Y1 = self.MR[i]
                Y2 = self.ML[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y1 = self.MMR[j]
                Y2 = self.MML[j]
            else: 		   #in a specific repetition
                Y1 = self.MR[i,j]
                Y2 = self.ML[i,j]

        #call plot		
        self._plot_progression(Y1, X=self.T1, names=None, xlabel=None, ylabel=None, title=None, show=False)
        self._plot_progression(Y2, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show, linestyles=['--']*self.m, linecolors=['C' + str(j) for j in range(self.m)])


    def plot_cumulated_regret_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Cumulated Regret", title="Cumulated Regret", linestyles=None, linecolors=None, show=True, inverse=False):

        w = -1 if inverse else 1
        
        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = w*self.MSL
            else:		   #in a specific repetition
                Y = w*self.SL[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = w*self.MSL[j]
            else: 		   #in a specific repetition
                Y = w*self.SL[i,j]

        #call plot		
        self._plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_average_regret_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Average Regret", title="Average Regret", linestyles=None, linecolors=None, show=True, inverse=False):	

        w = -1 if inverse else 1

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.algs]
            if i is None:  #averaged over repetitions      
                Y = w*self.MML
            else:		   #in a specific repetition
                Y = w*self.ML[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.algs[j])]
            if i is None:  #averaged over repetitions      
                Y = w*self.MML[j]
            else: 		   #in a specific repetition
                Y = w*self.ML[i,j]

        #call plot
        self._plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_freq_spectrum(self, F, bar_size=2, interpolation='hanning', cmap='gray_r', xlabel="$t$", ylabel="Actions", title="Frequency Spectrum", filename=None, show=True):

        bs = max(bar_size, 2)   #bar size (>1)
        h = self.k*bs+1     	#fig height (rows depends on the number of arms and bar size)
        w = len(F[0])			#fig width  (columns depends on time)

        img_map = np.zeros([h,w])   #image map

        for c in range(h):
            if (c % bs != 0):
                img_map[c] = F[c//bs]

        plt.imshow(img_map, aspect="auto", interpolation=interpolation, cmap=cmap)
        plt.yticks(np.arange(1, h, step=bs), self.K1)

        ##ax = plt.gca()
        #plt.xticks(np.arange(-.5, 10, 1))
        #plt.xticklabels(np.arange(1, 12, 1))

        plt.colorbar()

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)



    def plot_action_window_freq_spectrum(self, i=None, j=None, bar_size=2, interpolation='none', cmap='gray_r', xlabel="$t$", ylabel="Arms", title="Arm Pull Local Frequency Spectrum", show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #add zeros at time zero
        Z = np.reshape(np.zeros(self.k, dtype='float'), [self.k, 1])
        if i is None:
            Y = np.block([Z, self.MFW_a[j]])
        else:
            Y = np.block([Z, self.FW_a[i,j]])

        #call plot
        self.plot_freq_spectrum(Y, bar_size=bar_size, interpolation=interpolation, cmap=cmap, xlabel=xlabel, ylabel=ylabel, title=title)		


    def plot_comp_arm_count(self, i=None, j=None, xlabel="Arm (Selected Action)", ylabel="Number of Actions Taken", title="Selected Actions", show=True):

        if j is None:  #comparing algorithms
            names=[str(g) for g in self.algs]
            if i is None:   #averaging on repetitions
                Y = self.mn_a
            else:           #specific repetition
                Y = self.n_a[i]
        else:          #specific algorithm
            names=None
            if i is None:   #averaging on repetitions
                Y = self.mn_a[j]
            else:           #specific repetition
                Y = self.n_a[i,j]

        #call plot
        self._plot_comp_arms(Y, names=names, xlabel=xlabel, ylabel=ylabel, title=title)


    def plot_comp_arm_rewards(self, i=None, j=None, xlabel="Arms", ylabel="Total Rewards", title="Total Rewards per Arm", show=True):

        if j is None:  #comparing algorithms
            names=[str(g) for g in self.algs]
            if i is None:
                Y = self.msr_a
            else:
                Y = self.sr_a[i]
        else:          #specific algorithm
            names=None
            if i is None:
                Y = self.msr_a[j]
            else:
                Y = self.sr_a[i,j]

        #call plot
        self._plot_comp_arms(Y, names=names, xlabel=xlabel, ylabel=ylabel, title=title)


    def _call_plot_comp_algs(self, Y, E=None, xlabel="Algorithms", ylabel="Value", title="Comparison", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        x = np.arange(self.m, dtype='int')
        
        if names is None:
            names = [str(g) for g in self.algs]

        if compact_view:
            low = min(Y)
            high = max(Y)
            plt.ylim([low, high])

        #sort
        if sort:
            idx = np.argsort(Y)[::-1]  #desc order
            Y = Y[idx]
            names = [names[i] for i in idx]

        plt.xticks(x, names, rotation=names_rotation)
        plt.bar(x, Y, yerr=E, align='center', alpha=0.5)

        #bar labels
        if bar_labels:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
        
        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)
        
        
    def plot_comp_algs_total_rewards(self, i=None, xlabel="Algorithm", ylabel="Total Reward", title="Comparison (Total Reward)", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        Y = self.msr if i is None else self.sr[i]
        self._call_plot_comp_algs(Y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)


    def plot_comp_algs_survival_time(self, i=None, xlabel="Algorithm", ylabel="Average Survival Time", title="Comparison (Average Survival Time)", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        Y = self.MTTNB if i is None else self.TTNB[i]
        E = self.DTTNB if i is None else None
        self._call_plot_comp_algs(Y, E=E, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)


    def plot_comp_algs_ruined_episodes(self, xlabel="Algorithm", ylabel="Survival Episodes", title="Comparison (Survival Episodes)", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        Y = self.n - self.senb
        self._call_plot_comp_algs(Y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)


    def plot_comp_algs_cumulated_negative_budget(self, i=None, xlabel="Algorithm", ylabel="Cumulated Negative Budget", title="Comparison", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        #Y = self.snmb if i is None else self.snb[i]
        #E = self.dnmb if i is None else self.dnb[i]
        Y = self.msnb if i is None else self.snb[i]
        E = self.dsnb if i is None else None
        self._call_plot_comp_algs(Y, E=E, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)

    def plot_comp_algs_cumulated_regret(self, i=None, xlabel="Algorithm", ylabel="Cumulated Regret", title="Comparison", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        Y = self.msl if i is None else self.sl[i]
        E = self.dsl if i is None else None
        self._call_plot_comp_algs(Y, E=E, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)



    def plot_comp_freq_prop(self, i=None, j=None, names=['Cumulated Reward Proportion', 'Pull Frequency'], xlabel="$t$", ylabel="Cumulated Reward Proportion and Pull Frequency", title="Cumulated Reward and Number of Pulls", filename=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        if i is None:
            plt.bar(self.K1, self.mfr_a[j], width=0.8)
            plt.bar(self.K1, self.mf_a[j], width=0.4, alpha=0.5)
        else:
            plt.bar(self.K1, self.fr_a[i,j], width=0.8)
            plt.bar(self.K1, self.f_a[i,j], width=0.4, alpha=0.5)

        if names is not None:
            plt.legend(names)

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)


    def plot_reward_regret(self, i=None, j=None, names=['Cumulated Reward', 'Cumulated Regret', 'Best Strategy', 'Worst Possible Regret'], xlabel="$t$", ylabel="Reward and Regret", title="Best Strategy vs Cumulated Reward vs Regret ", filename=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #best policy
        SS = np.linspace(self.mu_star, self.mu_star*self.h, self.h)
        #worst regret
        l = -(self.mu_star-self.mu_worst)
        #worst cumulated regret
        SW = np.linspace(l, l*self.h, self.h)

        #best, reward, regret
        if i is None:
            Y = np.array([self.MSR[j], -self.MSL[j], SS, SW])
        else:
            Y = np.array([self.SR[i,j], -self.SL[i,j], SS, SW])

        self._plot_progression(Y, X=self.T1, show=False, names=names, title=title, ylabel=ylabel)

        if i is None:
            plt.fill_between(self.T1, 0, self.MSR[j], alpha=0.5)
            plt.fill_between(self.T1, 0, -self.MSL[j], alpha=0.5)
        else:
            plt.fill_between(self.T1, 0, self.SR[i,j], alpha=0.5)
            plt.fill_between(self.T1, 0, -self.SL[i,j], alpha=0.5)

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)


    def plot_survival_histogram(self, j=None, xlabel="$t$", ylabel="Time before ruin", title="Survival Time Histogram", filename=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #prepare histogram bins in log scale
        logbins=np.geomspace(10, self.h+1, 50, endpoint=True, dtype='int')

        #prepare data
        Y=self.TTNB.transpose()[j]

        plt.hist(Y, bins=logbins) #log=True
        plt.xscale('log')

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)
