import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """
    Bernoulli Bandit with k arms
    """
    def __init__(self, k):
        self.probs = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.k = k

    def step(self, i):
        """
        When arm i is pulled, return binary reward based on probability
        :param i: index of arm
        :return: reward
        """
        if np.random.rand() < self.probs[i]:
            return 1
        else:
            return 0

