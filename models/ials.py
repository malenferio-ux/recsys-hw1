"""Weighted squared-error ALS. Missing pairs have preference 0, confidence 1."""
import numpy as np
from scipy.linalg import solve
from .linalg import matmul


class IALS:
    def __init__(self, factors=32, reg=1.0, alpha=20., iterations=12, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def targets(self, counts):
        p = counts.astype(float).copy()
        p.data[:] = 1.
        return p

    def fit(self, counts):
        counts = counts.astype(float).tocsr()
        counts.sort_indices()
        p = self.targets(counts)
        c = counts.copy()
        c.data = 1 + self.alpha * np.log1p(c.data)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .05, (counts.shape[0], self.factors))
        self.item_factors = rng.normal(0, .05, (counts.shape[1], self.factors))
        ct, pt = c.T.tocsr(), p.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._step(self.item_factors, c, p)
            self.item_factors = self._step(self.user_factors, ct, pt)
        return self

    def _step(self, other, confidence, preference):
        base = matmul(other.T, other) + self.reg * np.eye(self.factors)
        result = np.zeros((confidence.shape[0], self.factors))
        for row in range(confidence.shape[0]):
            start, end = confidence.indptr[row:row + 2]
            idx = confidence.indices[start:end]
            y = other[idx]
            c = confidence.data[start:end]
            p = preference.data[start:end]
            if not len(idx):
                continue
            a = base + matmul(y.T * (c - 1), y)
            b = np.einsum('ij,i->j', y, c * p, optimize=False)
            result[row] = solve(a, b, assume_a='pos', check_finite=False)
        return result

    def score(self, users):
        return matmul(self.user_factors[users], self.item_factors.T)
