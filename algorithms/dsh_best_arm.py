import numpy as np
import random
import math


class DSH:
    def __init__(self, bandit, m):
        self.bandit = bandit
        self.m = m
        self.reward_perarm = [[] for i in range(len(self.bandit.arms))]
        self.mean_estimates = np.full(len(self.bandit.arms), float(0))
        self.S = np.arange(0,len(self.bandit.arms),1)
        self.Jt = np.random.choice(len(self.bandit.arms))
        self.J = []  # history of top m
        self.J.append(self.Jt)
        self.s = 1
        self.count = int(len(self.bandit.arms)/(len(self.S)*np.ceil(math.log2(len(self.bandit.arms)))))
        self.i = 0
        self.r = 0

    def initial_stage(self):
        # self.mean_estimates = np.full(len(self.bandit.arms), float(0))
        # self.reward_perarm = [[] for i in range(len(self.bandit.arms))]
        self.S = np.arange(0, len(self.bandit.arms), 1)
        self.Jt = self.J[self.s]
        self.s = self.s + 1
        budget = 2 ** (self.s - 1) * len(self.bandit.arms)
        self.count = int(budget/(len(self.S)*np.ceil(math.log2(len(self.bandit.arms)))))
        self.r = 0

    def initial_rank(self):
        self.i = 0
        self.r = self.r + 1
        self.S = self.elimiteS()
        budget = 2 ** (self.s - 1) * len(self.bandit.arms)
        self.count = int(budget / (len(self.S) * np.ceil(math.log2(len(self.bandit.arms)))))

    def add_reward(self, arm_i, reward):
        self.reward_perarm[arm_i].append(reward)
        self.mean_estimates[arm_i] = np.mean(self.reward_perarm[arm_i])

    def elimiteS(self):
        n = math.ceil(len(self.S)/2)
        mean_S = self.mean_estimates[self.S]
        nonzero = np.nonzero(mean_S)
        if len(nonzero[0]) == 0:
             S = np.random.choice(self.S, n, replace = False)

        elif len(nonzero[0]) < n:
            zero_index = np.where(mean_S == 0)[0]
            zero_index_choice = np.random.choice(self.S[zero_index], n-len(nonzero[0]),replace=False)
            nonzero_index = self.S[nonzero[0]]
            S = np.hstack((nonzero_index, zero_index_choice))
            print(len(S))
        else:
            sorted = np.argsort(-mean_S)
            S = self.S[sorted[:n]]
        return S

    def step(self, t):
        if self.count == 0:
            while self.count == 0:
                self.initial_rank()
        else:
            reward = self.bandit.play(self.S[self.i])
            self.add_reward(self.S[self.i], reward)
            index = np.where(self.mean_estimates == np.max(self.mean_estimates[self.S]))[0]
            self.Jt = np.random.choice(index)
            if self.i == len(self.S) - 1:
                self.i = 0
                self.count = self.count - 1
                if self.count == 0:
                    if self.r < math.ceil(math.log2(len(self.bandit.arms))) - 1:
                        self.initial_rank()
                    else:
                        self.S = self.elimiteS()
                        self.J.append(self.S[0])
                        self.initial_stage()
            else:
                self.i = self.i + 1
        return self.Jt