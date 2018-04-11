import matplotlib.pyplot as plt
import numpy as np

from algorithms import AT_LUCB
from algorithms import Uniform
from algorithms import NAIVE_DSAR
from algorithms import DSAR
from environments.gaussian_jun import linear_bandit, linear_means
from environments.gaussian_jun import polynomial_bandit, polynomial_means
class Linear():
    def __init__(self, m, n, steps, arms_n, variance, means, bandits):
        self.m = m
        self.n = n
        self.steps =steps
        self.arms_n = arms_n
        self.variance = variance
        self.means = means
        self.bandits = bandits

    def uniform_experment(self):
        J_t_result = np.zeros((2,self.steps))
        for i in range(self.n):
            J_t_n = np.zeros((2,self.steps))
            algo = Uniform(self.bandits, self.m)
            for t in range(self.steps):
                J_t = algo.step(t)
                top_m = [self.means[int(i)] for i in J_t]
                J_t_n[0][t] = np.min(top_m)
                J_t_n[1][t] = np.sum(top_m)
            J_t_result = J_t_result+J_t_n
        J_t_result = J_t_result/self.n
        return J_t_result

    def atlucb_experment(self):
        J_t_result = np.zeros((2,self.steps))
        sigma = 0.5
        alpha = 0.99
        epsilon = 0
        for i in range(self.n):
            J_t_n = np.zeros((2,self.steps))
            algo = AT_LUCB(sigma, alpha, epsilon, self.bandits, m)
            for t in range(self.steps):
                J_t = algo.step(t)
                top_m = [self.means[int(i)] for i in J_t]
                J_t_n[0][t] = np.min(top_m)
                J_t_n[1][t] = np.sum(top_m)
            J_t_result = J_t_result+J_t_n
        J_t_result = J_t_result/self.n
        return J_t_result

    def dsar_experment(self):
        J_t_result = np.zeros((2,self.steps))
        for i in range(self.n):
            J_t_n = np.zeros((2,self.steps))
            algo = DSAR(self.bandits, self.m)
            for t in range(self.steps):
                J_t = algo.step(t)
                top_m = [self.means[int(i)] for i in J_t]
                J_t_n[0][t] = np.min(top_m)
                J_t_n[1][t] = np.sum(top_m)
            J_t_result = J_t_result+J_t_n
        J_t_result = J_t_result/self.n
        return J_t_result


    def dsarnaive_experment(self):
        J_t_result = np.zeros((2,self.steps))
        for i in range(self.n):
            J_t_n = np.zeros((2,self.steps))
            algo = NAIVE_DSAR(self.bandits, self.m)
            for t in range(self.steps):
                J_t = algo.step(t)
                top_m = [self.means[int(i)] for i in J_t]
                J_t_n[0][t] = np.min(top_m)
                J_t_n[1][t] = np.sum(top_m)
            J_t_result = J_t_result+J_t_n
        J_t_result = J_t_result/self.n
        return J_t_result

    def plot(self):
        Linear_unform = self.uniform_experment()
        Linear_ATLUCB = Linear.atlucb_experment()
        Linear_DSAR = self.dsar_experment()
        Linear_naiveDSAR = self.dsarnaive_experment()

        plt.figure(1)
        plt.title('Linear m = %i' %self.m, fontweight='bold')
        plt.xlabel('t')
        plt.ylabel('min')
        plt.plot(Linear_unform[0], 'b-', color='r', label='uniform')
        plt.plot(Linear_ATLUCB[0],'b-',color='y',label="ATLUCB")
        plt.plot(Linear_DSAR[0], 'b-', color='g', label='DSAR')
        plt.plot(Linear_naiveDSAR[0], 'b-', color='b', label='NAIVE_DSAR')
        plt.legend()
        plt.xlim(0, steps)
        plt.ylim(0.5, 0.9)
        plt.savefig('linear_m=%i" _min.png'% self.m)

        plt.figure(2)
        plt.title('Linear m = %i' % self.m, fontweight='bold')
        plt.xlabel('t')
        plt.ylabel('sum')
        plt.plot(Linear_unform[1], 'b-', color='r', label='uniform')
        plt.plot(Linear_ATLUCB[1],'b-',color='y',label="ATLUCB")
        plt.plot(Linear_DSAR[1], 'b-', color='g', label='DSAR')
        plt.plot(Linear_naiveDSAR[1], 'b-', color='b', label='NAIVE_DSAR')
        plt.legend()
        plt.xlim(0, steps)
        plt.ylim(7.8, 9.0)
        plt.savefig('linear_m=%i" sum.png' % self.m)


n = 200
steps = 150000
m = 10
arms_n = 1000
variance = 0.25
linearmeans = linear_means(arms_n)
lin_bandits = linear_bandit(arms_n, variance)
Linear = Linear(m, n, steps, arms_n, variance, linearmeans,lin_bandits)
Linear.plot()





