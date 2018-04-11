import numpy as np
import bottleneck
import random
class AT_LUCB:
    def __init__(self, sigma, alpha, epsilon ,bandit, m):
        self.sigma = sigma
        self.alpha = alpha
        self.bandit = bandit
        self.epsilon = epsilon
        self.m = m
        self.reward_perarm = [[] for i in range(len(bandit.arms))]
        self.mean_estimates = np.full(len(bandit.arms), float(0))
        #self.Jt = np.full(self.m , int(0))
        self.Jt = np.random.randint(len(bandit.arms), size=self.m)
        self.J = [] # history of top m
        self.J.append(self.Jt)
        self.S = [1] #list of stages, initially 1

    def beta(self, u, t, sigma1):
        k1 = 1.25
        beta = (np.log((len(self.bandit.arms))*k1*((t)**4)/sigma1)/(2*u))**0.5
        return beta

    def term(self, t, epsilon): #satifies the terminal candition or not
        L = self.low_arm(t)[1]
        U = self.high_arm(t)[1]
        if True == L or True == U:
            return False

        else:
            if (L - U < epsilon):
                return True
            else:
                return False

    def high_arm(self, t): #return the mini  mean -beta  of top m and its index
        mini_list = [True if self.mean_estimates[i] == 0 else (self.mean_estimates[i] - self.beta(np.sum(self.reward_perarm[i])/self.mean_estimates[i], t-1, self.sigma)) for i in self.J[t-1]]
        # mini_list store the mean_estimates - Beta of the top m arms.
        # store the ones nerve been pulled as True (beta is infinite)
        if True in mini_list:
            mini = True
        else:
            mini = np.min(mini_list)
        index = self.J[t-1][mini_list.index(mini)]
        return index, mini


    def low_arm(self,t):#return the max  mean + beta  of top m and its index
        maxm_list = [True if self.mean_estimates[i]==0 else (self.mean_estimates[i] + self.beta(np.sum(self.reward_perarm[i]) / self.mean_estimates[i], t-1, self.sigma)) for i in range(len(self.mean_estimates))]
        # maxm_list store the mean_estimates + Beta of the all the arms.
        # store the ones nerve been pulled as True (beta is infinite)
        for i in range(self.m):
            maxm_list[(self.J[t-1])[i]] = 0
        #keep the mean_estimates + beta of top m as 0 to make sure it will not been pulled
        if True in maxm_list:
            maxm = True
        else:
            maxm = np.max(maxm_list)
        index = maxm_list.index(maxm)
        return index, maxm

    def get_top_m(self):
        return bottleneck.argpartition(-self.mean_estimates, self.m)[:self.m]

    def add_reward(self, arm_i, reward):
        self.reward_perarm[arm_i].append(reward)
        self.mean_estimates[arm_i] = np.mean(self.reward_perarm[arm_i])

    def step(self, t):
        if self.term(t, self.epsilon):
            s = self.S[t - 1]
            while self.term(t, self.epsilon):
                s = s + 1
                self.sigma = self.sigma*(self.alpha**(self.s-1))
            self.S.append(s)
            self.Jt = self.get_top_m()
        else:
            self.S.append(self.S[t-1])
            if self.S[t] == 1:
                self.Jt = self.get_top_m()
        high_index = self.high_arm(t)[0]
        low_index = self.low_arm(t)[0]
        highreward = self.bandit.play(high_index)
        self.add_reward(high_index, highreward)
        #pull H
        lowreward = self.bandit.play(low_index)
        self.add_reward(low_index, lowreward)
        #pull L
        self.J.append(self.Jt)
        return self.Jt,self.S[t]




















