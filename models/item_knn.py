"""Item-based cosine collaborative filtering with co-occurrence shrinkage."""
import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=100, shrink=10):
        self.neighbors, self.shrink = neighbors, shrink

    def fit(self, interactions):
        self.history = interactions.astype(float).copy()
        self.history.data[:] = 1
        co = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / np.maximum(norms[:, None] * norms[None, :], 1e-12)
        sim *= co / (co + self.shrink + 1e-12)
        np.fill_diagonal(sim, 0)
        # Each source item votes for its strongest neighbors; stable item-ID ties.
        indices = np.argsort(-sim, axis=1, kind='stable')[:, :self.neighbors]
        rows = np.repeat(np.arange(len(sim)), indices.shape[1])
        self.similarity = sparse.csr_matrix(
            (sim[rows, indices.ravel()], (rows, indices.ravel())), shape=sim.shape)
        return self

    def score(self, users):
        return (self.history[users] @ self.similarity).toarray()
