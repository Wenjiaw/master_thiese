import numpy as np
import bottleneck
class Uniform:
    def __init__(self, bandit, m):
        self.bandit = bandit
        self.m = m
        self.reward_perarm = [[] for i in range(len(self.bandit.arms))]
        self.mean_estimates = np.full(len(bandit.arms), float(0))

    def add_reward(self, arm_i, reward):
        self.reward_perarm[arm_i].append(reward)
        self.mean_estimates[arm_i] = np.mean(self.reward_perarm[arm_i])

    def leastsample_index(self):
        if 0 in self.mean_estimates:
            leastsampled = np.where(self.mean_estimates == 0)[0]
        else:
            counts_per_arm = [sum(self.reward_perarm[i])/self.mean_estimates[i] for i in range(len(self.bandit.arms))]
            leastsampled = np.where(counts_per_arm == np.min(counts_per_arm))[0]
        return leastsampled

    def step(self, t):
        leastsampled = self.leastsample_index()
        arm_i = np.random.choice(leastsampled)
        reward = self.bandit.play(arm_i)
        self.add_reward(arm_i, reward)
        J_t = bottleneck.argpartition(-self.mean_estimates, self.m)[:self.m]
        return J_t













