import numpy as np
from solver import Solver


class ThompsonSampling(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.k)
        self._b = np.ones(self.bandit.k)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        i = np.argmax(samples)
        r = self.bandit.step(k)
        # update parameter for Beta distribution
        self._a[i] += r
        self._b[i] += (1 - r)
        return i
