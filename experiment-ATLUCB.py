import matplotlib.pyplot as plt
import numpy as np

from algorithms import AT_LUCB
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

n = 200
steps = 15000
m = 10
sigma =0.5
alpha = 0.99
arms_n = 1000
variance = 0.25
epsilon = 0

# ATLUCB_experiment=atlucb_experment(n, steps, m, arms_n, variance, sigma, alpha, epsilon)

linearmeans = linear_means(arms_n)
lin_bandits = linear_bandit(arms_n, variance)
J_t_n = np.zeros((2,steps))
algo = AT_LUCB(sigma, alpha, epsilon, lin_bandits, m)
for t in range(1,steps):
    J_t = algo.step(t)

    top_m = [linearmeans[int(i)] for i in J_t[0]]
    J_t_n[0][t] = np.min(top_m)

    J_t_n[1][t] = np.sum(top_m)




plt.figure(1)
plt.title('AT-LUCB m = 10', fontweight='bold')
plt.xlabel('t')
plt.ylabel('min')
plt.plot(J_t_n[0],'b-',color='r',label='AT-LUCB')
plt.legend()
plt.xlim(0,steps)
plt.ylim(0.5,0.9)
plt.savefig('lin_AT-LUCB_min.png')
plt.figure(2)
plt.title('AT-LUCB m = 10', fontweight='bold')
plt.xlabel('t')
plt.ylabel('sum')
plt.plot(J_t_n[1],'b-',color='r',label='AT-LUCB')
plt.legend()
plt.xlim(0,steps)
plt.ylim(7.8,9.0)
plt.savefig('lin_AT-LUCB_sum.png')
plt.show()


