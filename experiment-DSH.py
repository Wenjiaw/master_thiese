import matplotlib.pyplot as plt
import numpy as np

from algorithms.dsh_best_arm import DSH
from environments.gaussian_jun import linear_bandit, linear_means

#
# def atlucb_experment(n, steps, m, arms_n, variance, sigma, alpha, epsilon):
#     J_t_result = np.zeros((2,steps))
#     linearmeans = linear_means(arms_n)
#     for i in range(n):
#         lin_bandits = linear_bandit(arms_n, variance)
#         J_t_n = np.zeros((2,steps))
#         algo = AT_LUCB(sigma, alpha, epsilon, lin_bandits, m)
#         for t in range(steps):
#             J_t = algo.step(t)[0]
#             top_m = [linearmeans[int(i)] for i in J_t]
#             J_t_n[0][t] = np.min(top_m)
#             J_t_n[1][t] = np.sum(top_m)
#         J_t_result = J_t_result+J_t_n
#     J_t_result = J_t_result/n
#     return J_t_result

n=200
steps = 150000
m = 1
arms_n = 1000
variance = 0.25


# ATLUCB_experiment=atlucb_experment(n, steps, m, arms_n, variance, sigma, alpha, epsilon)
#
# linearmeans = linear_means(arms_n)
# lin_bandits = linear_bandit(arms_n, variance)
# J_t_n = np.zeros((2,steps))
# algo = DSAR(lin_bandits, m)
# for t in range(1,steps):
#
#     J_t = algo.step(t)
#
#     top_m = [linearmeans[int(i)] for i in J_t]
#     J_t_n[0][t] = np.min(top_m)
#     J_t_n[1][t] = np.sum(top_m)


linearmeans = linear_means(arms_n)
J_t_result = np.zeros(steps)

for i in range(n):
    #lin_bandits = linear_bandit(arms_n, variance, steps)
    lin_bandits = linear_bandit(arms_n, variance)
    J_t_n = np.zeros(steps)
    algo = DSH(lin_bandits, m)
    for t in range(steps):
        J_t = algo.step(t)
        J_t_n[t] = linearmeans[int(J_t)]


    J_t_result = J_t_result+J_t_n
    print(J_t_result)
result = J_t_result/n




plt.figure(1)
plt.title('DS m = 10', fontweight='bold')
plt.xlabel('t')
plt.ylabel('min')
plt.plot(result,'b-',color='r',label='DSAR')
plt.legend()
plt.xlim(0,steps)
plt.ylim(0.75,0.9)
plt.savefig('lin_DSH.png')


