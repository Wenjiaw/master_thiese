import matplotlib.pyplot as plt
import numpy as np

from algorithms.uniform import Uniform
from environments.gaussian_jun import linear_bandit, linear_means

n = 200
steps = 150000
m = 10
arms_n = 1000
variance = 0.25
# lin_bandits = [linear_bandit(arms_n, variance) for i in range(steps)]
linearmeans = linear_means(arms_n)
J_t_result = np.zeros((2,steps))

for i in range(n):
    #lin_bandits = linear_bandit(arms_n, variance, steps)
    lin_bandits = linear_bandit(arms_n, variance)
    J_t_n = np.zeros((2,steps))
    algo = Uniform(lin_bandits, m)
    for t in range(steps):
        J_t = algo.step(t)
        top_m = [linearmeans[int(i)] for i in J_t]
        J_t_n[0][t] = np.min(top_m)
        J_t_n[1][t] = np.sum(top_m)

    J_t_result = J_t_result+J_t_n
result = J_t_result/n

plt.figure(1)
plt.title('Linear m = 10', fontweight='bold')
plt.xlabel('t')
plt.ylabel('min')
plt.plot(result[0],'b-',color='r',label='uniform')
plt.legend()
plt.xlim(0,steps)
plt.ylim(0.5,0.9)
plt.savefig('lin_uniform_min.png')
plt.figure(2)
plt.title('Linear m = 10', fontweight='bold')
plt.xlabel('t')
plt.ylabel('sum')
plt.plot(result[1],'b-',color='r',label='uniform')
plt.legend()
plt.xlim(0,steps)
plt.ylim(7.8,9.0)
plt.savefig('lin_uniform_sum.png')
plt.show()


