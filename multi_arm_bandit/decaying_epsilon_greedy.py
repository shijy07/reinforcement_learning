import numpy as np
from solver import Solver


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.k)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.rando.rand() < 1.0 / self.total_count:
            i = np.random.randint(0, self.bandit.k)
        else:
            i = np.argmax(self.estimates)
        r = self.bandit.step(i)
        self.estimates[i] += 1.0 / (self.counts[i] + 1) * (r - self.estimates[i])
        return i

