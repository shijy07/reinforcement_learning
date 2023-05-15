import numpy as np
from solver import Solver


class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.k)

    def run_one_step(self):
        if np.rando.rand() < self.epsilon:
            i = np.random.randint(0, self.bandit.k)
        else:
            i = np.argmax(self.estimates)
        r = self.bandit.step(i)
        self.estimates[i] += 1.0 / (self.counts[i] + 1) * (r - self.estimates[i])
        return i


