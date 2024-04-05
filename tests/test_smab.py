# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:51:40 2024

@author: fperotto
"""

def main():

    #Dependencies
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ioff()
    
    plt.rcParams['figure.figsize'] = (10, 8)
    
    
    from pyrl import Sim, Agent, EnvWrapper, PyGameRenderer, PyGameGUI, ensure_tuple
    
    from pyrl.mab.arms import RandomArm, BernoulliArm
    from pyrl.mab.policies import  BasePolicy, RandomPolicy, FixedPolicy, GreedyPolicy, EmpiricalSumPolicy, EpsilonGreedyPolicy, SoftMaxPolicy, UCBPolicy, BernKLUCBPolicy, ThompsonPolicy, BayesUCBPolicy #, MaRaBPolicy
    #from pyrl.mab.policies import BanditGamblerPolicy, BanditGamblerUCBPolicy, AlarmedUCBPolicy, AlarmedBernKLUCBPolicy, AlarmedEpsilonGreedyPolicy, PositiveGamblerUCB
    from pyrl.mab.simulator import SMAB

    from pyrl.mab.env import EnvMAB
    
    
    #BERNOULLI ARMS
    means = np.array([0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.45, 0.55, 0.6])
    #means = np.concatenate((np.repeat(0.1, 15), np.repeat(0.7, 5), [0.9]))
    k = len(means)
    maxr = +1.0
    minr = -1.0
    ampl = maxr - minr
    #arms objects
    arms = [BernoulliArm(m, maxr=maxr, minr=minr) for m in means]
    #arms = [BernoulliArm(m) for m in means]
    
    #SHOW ARMS
    for i, arm in enumerate(arms):
        print(i, arm.mean)
        plt.bar(i+1, arm.mean)
    plt.xlabel('$i$ (arm)')
    plt.ylabel('$P(X_i = 1)$ (probability of success)')
    plt.show()
    
    
    #initial budget
    b_0 = k
    
    
    #each arm must be tried at least w times at beginning
    w=1
    
    #algorithm
    algs = [UCBPolicy(k, alpha=1.0*ampl),  #alpha is related to the amplitude of rewards
            EpsilonGreedyPolicy(k, epsilon=0.1)]
    
    #time-horizon
    h = 1000 #5000
    
    #repetitions
    n = 15
    
    #window average parameter
    #win = tau//10


    env = EnvMAB(arms, h=h, b_0=b_0)
    env.reset()
    alg = algs[0]
    alg.reset()
    while not (env.terminated or env.truncated):
        action = alg.choose()
        _, reward, terminated, truncated, _ = env.step(action)
        alg.observe(reward)

    print(env.b)    
    
    
    M = SMAB(arms=arms, algs=algs, h=h, b_0=10, n=n, w=1, run=False, prev_draw=True, use_multiprocess=True, save_only_means=False)
    M.run(tqdm_leave=True)
    
    #M.plot_action_window_freq_spectrum()
    
    M.plot_precision_progression()
    
    M.plot_cumulated_reward_progression()
    M.plot_average_reward_progression()
    M.plot_cumulated_regret_progression()
    M.plot_average_regret_progression()
    
    #M.plot_comp_arm_count()
    #M.plot_comp_arm_rewards()
    
    M.plot_cumulated_reward_regret_progression()
    M.plot_average_reward_regret_progression()
    
    M.plot_budget_progression()
    #M.plot_negative_budget_progression()
    #M.plot_negative_budget_time_progression()
    #M.plot_cumulated_negative_budget_progression()
    
    M.plot('mortal_budget')
    M.plot('immortal_budget')
    M.plot('survival')
    M.plot('sum_reward')
    M.plot('mortal_precision')
    M.plot('immortal_precision')
    M.plot('penalized_budget')
    M.plot('avg_reward')
    M.plot('avg_mortal_reward')
    M.plot('sum_regret')
    M.plot('avg_regret')
    

if __name__ == "__main__":
    main()