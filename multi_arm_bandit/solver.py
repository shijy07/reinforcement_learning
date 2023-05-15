import numpy as np


class Solver:
    """
    Base class for multi-arm bandit
    """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.k)
        self.regret = 0.
        self.actions = []
        self.regrets =[]

    def update_regret(self, i):
        self.regret += self.bandit.best_prob - self.bandit.probs[i]
        self.regrets.append(self.regret)

    def run_one_step(self):
        return NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            i = self.run_one_step()
            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)
    
