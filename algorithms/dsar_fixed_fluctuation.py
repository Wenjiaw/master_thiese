import numpy as np
import bottleneck
import math


class DSAR_fixed_fluctuation:
    def __init__(self, bandit, m):
        self.bandit = bandit
        self.m = m
        self.reward_perarm = [[] for i in range(len(self.bandit.arms))]
        self.mean_estimates = np.full(len(self.bandit.arms), float(0))
        self.active = np.arange(1 ,len(self.bandit.arms)+1 , 1)
        self.Jt = np.random.choice(len(self.bandit.arms),self.m)
        self.J = []  # history of top m
        self.J.append(self.Jt)
        self.s = 1
        self.k = 1
        self.m_lack = self.m
        self.n = [0 if k == 0 else 1 for k in range(len(self.bandit.arms))]
        self.count = self.n[self.k] - self.n[self.k - 1]
        self.i = 0

    def initial_stage(self):
        self.m_lack = self.m
        self.s = self.s + 1
        budget = 2 ** (self.s - 1) * len(self.bandit.arms)
        self.n = [math.ceil(self.n_k(k, budget)) for k in range(len(self.bandit.arms))]
        self.active = np.arange(1, len(self.bandit.arms) + 1, 1)
        self.k = 1
        self.count = self.n[self.k] - self.n[self.k-1]

    def n_k(self, k, budget):
        if k == 0:
            return 0
        else:
            K = len(self.bandit.arms)
            n = budget
            return 1/self.log(K) * (n-k)/(K+1-k)

    def log(self, n):
        a = 1 / 2
        for i in range(2, n + 1):
            a = a + 1 / i
        return a

    def add_reward(self, arm_i, reward):
        self.reward_perarm[arm_i].append(reward)
        self.mean_estimates[arm_i] = np.mean(self.reward_perarm[arm_i])

    def empirical_gaps(self):
        if self.m_lack == 0:
            pass
        else:
            mean_active = self.mean_estimates[self.active-1]
            index = np.argsort(-mean_active)
            if mean_active[index[0]] - mean_active[index[self.m_lack]] >= (
                mean_active[index[self.m_lack - 1]] - mean_active[index[-1]]):
                if self.active[index[0]] - 1 in self.Jt:
                    same_index = np.argwhere(self.Jt == self.active[index[0]] - 1)
                    self.Jt[same_index] = self.Jt[self.m - self.m_lack]
                self.Jt[self.m - self.m_lack] = self.active[index[0]] - 1
                self.m_lack = self.m_lack - 1
                self.active = np.delete(self.active, index[0])
            else:
                self.active = np.delete(self.active, index[-1])
        return self.active

    def step(self, t):
        reward = self.bandit.play(self.active[self.i] - 1)
        self.add_reward(self.active[self.i] - 1, reward)
        if self.i == len(self.active) - 1:
            self.count = self.count - 1
            self.i = 0
            if self.count == 0:
                while self.count == 0:
                    self.active = self.empirical_gaps()
                    self.i = 0
                    if self.k < len(self.bandit.arms) - 1:
                        self.k = self.k + 1
                        self.count = self.n[self.k] - self.n[self.k - 1]
                    else:
                        J_laststage = np.copy(self.Jt)
                        self.J.append(J_laststage)
                        self.initial_stage()
        else:
            self.i = self.i + 1
        return self.Jt
