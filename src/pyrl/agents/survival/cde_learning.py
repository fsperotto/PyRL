import time
from math import exp

import numpy as np
from gymnasium.spaces import Space

from pyrl import Agent

class CDELearning(Agent):
    def __init__(self, observation_space, action_space, survival_threshold, exploration_threshold, initial_observation=None, budget=None):
        super().__init__(observation_space, action_space, initial_observation, budget)
        self.survival_threshold = survival_threshold
        self.exploration_threshold = exploration_threshold

    def reset(self, initial_observation, reset_knowledge=True):
        self.c = np.zeros((self.observation_space.n, self.action_space.n)) # confidence
        self.d = np.full((self.observation_space.n, self.action_space.n), -1) # distance
        self.e = np.zeros((self.observation_space.n, self.action_space.n)) # efficiency

        self.h = History()
        self.history_should_reset = True
        self.recharge = False
        
        # TODO: create a max distance variable which contains the higher distance
        # that the agent have walked through and update the exploration threshold with this value

        return super().reset(initial_observation, reset_knowledge)

    def act(self):
        command = "n"
        while command not in ["n", "0", "1", "2", "3"]:
            command = input("Passer à l'étape suivante (n) : ")
        # print("====================================")
        # print(f"== STEP : {self.t}")
        # print("====================================")
        
        # print("--------> ACTION")
        if command != "n":
            # print(f"Human action !")
            self.a = int(command)
        else:
            self.a = np.random.randint(0, self.action_space.n)
            
            # print(f"Default random action : {self.a}")
            # print(f"Recharge : {self.recharge}")
            # if survival mode
            if self.recharge:
                # print('Mode : Recharge')
                for a in np.argsort(self.e[self.s, :])[::-1]: # Higher value index is first element of sorted_indices
                    # Positive efficiency
                    if self.e[self.s, a] > 0 and self.budget - self.d[self.s, a] > 0 and self.d[self.s, a] != -1:
                        # print("Choose action with best efficiency")
                        self.a = a
                        break
                    
                    # zero efficiency but not confident
                    if self.e[self.s, a] == 0 and self.budget - self.d[self.s, a] > 0 and self.d[self.s, a] != -1 and self.c[self.s, a] < 10:
                        # print("Choose null efficiency with low confidence")
                        self.a = a
                        break
                    
                    # only distance is known
                    if self.e[self.s, a] == 0 and self.d[self.s, a] != -1 and self.budget - self.d[self.s, a] > 0:
                        # print("Choose only distance known")
                        self.a = a
                        break
                    
                    if self.e[self.s, a] < 0 and self.budget - self.d[self.s, a] > 0 and self.c[self.s, a] < 10:
                        # print("Choose negative efficiency with low confidence")
                        self.a = a
                        break
            # else exploration mode
            else:
                # full curiosity
                # print(f"Curiosity balance : {self.curiousity_balance()}")
                random = np.random.random()
                # print(f"Curiosity random is : {random}")
                # print("Full curious if random is lower than curiosity balance")
                if random < self.curiousity_balance():
                    # print("Mode : Full curious")
                    # TODO: Utiliser l'indice de confiance pour se détacher de la récompense positive
                    random = np.random.random()
                    # print(f"Should detach : {random} > 0.8 and {self.r} > 0")
                    if random > 0.8 and self.r > 0: # On se détach de la récompense positive pour en trouver d'autre.
                        # print("Detach from origin !")
                        # print("Reset history !")
                        self.h.reset()
                        self.history_should_reset = True
                        
                    actions = np.where(self.c[self.s, :] == self.c[self.s, :].min())[0]
                
                    # print(f"{len(actions)} actions trouvées")
                    
                    if len(actions) > 0:
                        self.a = np.random.choice(actions)
                # careful curiosity
                else:
                    # print("Mode : Careful curiosity")
                    for a in np.argsort(self.e[self.s, :])[::-1]:
                        if self.e[self.s, a] > 0 and self.d[self.s, a] != -1 and self.c[self.s, a] < 10:
                            # print("Choose positive efficiency with low confidence")
                            self.a = a
                            break
                        
                        # zero efficiency but not confident
                        if self.e[self.s, a] == 0 and self.budget - self.d[self.s, a] > 0 and self.d[self.s, a] != -1 and self.c[self.s, a] < 10:
                            # print("Choose null efficiency with low confidence")
                            self.a = a
                            break
                        
                        if self.e[self.s, a] < 0 and self.budget - self.d[self.s, a] > 0 and self.c[self.s, a] < 10:
                            # print("Choose negative efficiency with low confidence")
                            self.a = a
                            break
                        
                        # only distance is known
                        if self.e[self.s, a] == 0 and self.d[self.s, a] != -1 and self.budget - self.d[self.s, a] > 0:
                            # print("Choose only distance known")
                            self.a = a
                            break
                
        # print(f"Choosed action : {self.a}")
        
        self.c[self.s, self.a] = self.c[self.s, self.a] + 1 # TODO: Use confidence for exploration

        return self.a

    def observe(self, s, r, terminated, truncated):
        self.h.append(self.s, self.a, r)
        
        # print("--------> OBSERVATIONS")
        # print(f"Last state : {self.s}")
        # print(f"New state: {s}")
        # print(f"Reward : {r}")
        
        super().observe(s, r, terminated, truncated)
        
        # print(f"Budget : {self.budget}")
        
        if self.budget < self.survival_threshold:
            self.recharge = True
        elif self.budget > self.exploration_threshold:
            self.recharge = False
            
        self.exploration_threshold = self.survival_threshold + self.d.max() if self.survival_threshold + self.d.max() > self.exploration_threshold else self.exploration_threshold

    def learn(self):
        # print("--------> LEARNING")
        if self.r > 0 or self.d[self.s, :].max() > -1:
            # print("Learn is True")
            # distance
            d = - self.r if self.r > 0 else np.extract(self.d[self.s, :] > -1, self.d[self.s, :]).min()
            # print(f"Distance : {d}")
            # efficiency
            os, _, _ = self.h.get_at(0) # origin state
            # print(f"Is loop : {self.s == os}")
            
            self.h.spread(self, d, self.e[self.s, :].max(), self.s == os)

            # print(f"Is state same as origin state ? current state : {self.s}, origin state : {os})")
            if self.s == os or self.history_should_reset:
                # print("History before reset")
                # print(self.h)
                # print("Reset history !")
                self.h.reset()
                self.history_should_reset = False

        # print("History before clean")
        # print(self.h)
        self.h.clean(self.s)
        # print("History after clean")
        # print(self.h)
        # print("Distance :")
        # print(self.d)
        # print("Efficiency :")
        # print(self.e)
        # print("Confidence :")
        # print(self.c)

    def curiousity_balance(self):
        return (1/(self.exploration_threshold - self.survival_threshold)) * (self.budget - self.survival_threshold)

CHRONOLOGICAL = True
ANTICHRONOLOGICAL = False

class History:
    def __init__(self, history: list = list()) -> None:
        self.h = history
        self.sh = list()

    def reset(self):
        self.h = list()
        self.sh = list() # sub history

    def append(self, s, a, r):
        self.h.append((s, a, r))
        
    def get_at(self, i):
        return self.h[i]
    
    def size(self):
        return len(self.h)
        
    def get_all(self, order=CHRONOLOGICAL):
        if order == CHRONOLOGICAL:
            return self.h
        else:
            return self.h[::-1]
        
    def get_last(self):
        return self.h[-1]
        
    def clean(self, s):
        for i in range(len(self.h)):
            state, _, _ = self.h[i]

            if state == s:
                sh = History(self.h[i:]) # new sub history

                # Add sub sub history to sub history 
                for sub_state, _, _ in sh.get_all():
                    # print("Maybe create sub sub history")
                    for index, sub_history in enumerate(self.sh):
                        sub_s, _, _ = sub_history.get_at(0)
                        
                        if sub_s == sub_state:
                            # print("Create sub history")
                            sh.append_sub_history(self.sh[index])
                            del self.sh[index]
                
                self.sh.append(sh)
                self.h = self.h[:i]
                break
            
    def append_sub_history(self, history):
        self.sh.append(history)
            
    def delete(self, i):
        del self.h[i]
            
    def spread(self, agent, d, e, loop_done=False):
        # efficiency
        if loop_done:
            tr = 0 # total reward
            for _, _, r in self.get_all(ANTICHRONOLOGICAL): # TODO: Improve this double following loop
                tr = tr + r
                
            c = self.size() # count
            e = tr / c
                
        if e != 0:
            for s, a, r in self.get_all(ANTICHRONOLOGICAL):
                if e > agent.e[s, a]:
                    agent.e[s, a] = e
                
        # distance
        for s, a, r in self.get_all(ANTICHRONOLOGICAL):
            d = d + np.abs(r)
            if agent.d[s, a] > d or agent.d[s, a] == -1:
                agent.d[s, a] = d
                for h in self.sh:
                    ss, sa, _ = h.get_at(0) 
                    if ss == s: # Si l'historique démarre avec le même état
                        # print(f"Spread for ({ss}, {sa})")
                        h.spread(agent, d, e)

    def __str__(self, detph=0):
        value = ""
        
        for _ in range(detph):
            value = value + "  "
            
        value = value + "| " + self.h.__str__()
        
        for h in self.sh:
            value = value + "\n" + h.__str__(detph + 1)
            
        return value
