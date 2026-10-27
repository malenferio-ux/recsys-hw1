"""EASE: closed-form ridge item-to-item regression with zero diagonal."""
import numpy as np
from scipy.linalg import solve


class EASE:
    def __init__(self, reg=100.):
        self.reg = reg

    def fit(self, interactions):
        self.history = interactions.astype(float).copy()
        self.history.data[:] = 1
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        precision = solve(gram, np.eye(len(gram)), assume_a='pos', check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
