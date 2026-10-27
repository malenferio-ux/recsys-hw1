"""IALS with a repeat-purchase preference target; confidence is unchanged."""
import numpy as np
from .ials import IALS


class RepeatTargetIALS(IALS):
    def __init__(self, beta=.5, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def targets(self, counts):
        p = counts.astype(float).copy()
        for u in range(p.shape[0]):
            start, end = p.indptr[u:u + 2]
            if end > start:
                values = np.log1p(p.data[start:end])
                p.data[start:end] = (1 - self.beta) + self.beta * values / values.max()
        return p
