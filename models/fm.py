"""Second-order FM, sparse NumPy/SciPy gradients and handwritten Adam."""
import numpy as np
from scipy import sparse
from scipy.special import expit
from .linalg import matmul


class FM:
    def __init__(self, factors=16, reg=1e-5, lr=.01, epochs=10,
                 negatives=3, batch_size=2048, seed=42):
        self.factors, self.reg, self.lr = factors, reg, lr
        self.epochs, self.negatives = epochs, negatives
        self.batch_size, self.seed = batch_size, seed

    @staticmethod
    def forward(x, w, v, bias):
        xv = x @ v
        scores = bias + x @ w + .5 * ((xv * xv).sum(1) - (x.multiply(x) @ (v * v)).sum(1))
        return scores, xv

    def fit(self, counts, user_features, item_features):
        self.uf, self.itf = user_features.tocsr(), item_features.tocsr()
        dim = self.uf.shape[1] + self.itf.shape[1]
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(dim)
        self.v = rng.normal(0, .03, (dim, self.factors))
        self.bias = np.zeros(1)
        params = [self.w, self.v, self.bias]
        moments = [np.zeros_like(a) for a in params]
        squares = [np.zeros_like(a) for a in params]
        pos_u, pos_i = counts.nonzero()
        assert np.max(np.diff(counts.indptr)) < counts.shape[1], 'No available negatives'
        self.loss_history = []
        step = 0
        for _ in range(self.epochs):
            nu = np.repeat(pos_u, self.negatives)
            ni = rng.integers(counts.shape[1], size=len(nu))
            # Rejection sampling excludes ALL observed training items, no future labels.
            bad = np.asarray(counts[nu, ni]).ravel() > 0
            while bad.any():
                ni[bad] = rng.integers(counts.shape[1], size=bad.sum())
                bad = np.asarray(counts[nu, ni]).ravel() > 0
            users = np.concatenate([pos_u, nu])
            items = np.concatenate([pos_i, ni])
            labels = np.concatenate([np.ones(len(pos_u)), np.zeros(len(nu))])
            order = rng.permutation(len(users))
            epoch_loss = 0.
            for begin in range(0, len(order), self.batch_size):
                ids = order[begin:begin + self.batch_size]
                x = sparse.hstack([self.uf[users[ids]], self.itf[items[ids]]], format='csr')
                logits, xv = self.forward(x, self.w, self.v, self.bias)
                error = (expit(logits) - labels[ids]) / len(ids)
                dw = x.T @ error + self.reg * self.w
                dv = x.T @ (error[:, None] * xv) - (x.multiply(x).T @ error)[:, None] * self.v
                dv += self.reg * self.v
                grads = [dw, dv, np.array([error.sum()])]
                step += 1
                for param, grad, m, s in zip(params, grads, moments, squares):
                    m *= .9
                    m += .1 * grad
                    s *= .999
                    s += .001 * grad * grad
                    param -= self.lr * (m / (1 - .9**step)) / (np.sqrt(s / (1 - .999**step)) + 1e-8)
                epoch_loss += np.sum(np.logaddexp(0, logits) - labels[ids] * logits)
            self.loss_history.append(epoch_loss / len(users))
        # Exact FM decomposition: within-user + within-item + cross interactions.
        cut = self.uf.shape[1]
        self.u_offset, self.u_latent = self.forward(self.uf, self.w[:cut], self.v[:cut], 0.)
        self.i_offset, self.i_latent = self.forward(self.itf, self.w[cut:], self.v[cut:], 0.)
        return self

    def score(self, users):
        return (self.bias + self.u_offset[users, None] + self.i_offset[None, :]
                + matmul(self.u_latent[users], self.i_latent.T))
