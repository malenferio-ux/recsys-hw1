"""Independent checks on tiny matrices; run: python -m unittest discover -s tests."""
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from models.ials import IALS
from models.custom_ials import RepeatTargetIALS
from models.ease import EASE
from models.fm import FM
from evaluation import evaluate, recommend
from models.linalg import matmul
from models.item_knn import ItemKNN
from data_utils import make_protocol


class Correctness(unittest.TestCase):
    def test_item_cosine_and_zero_diagonal(self):
        r = sparse.csr_matrix([[1., 1., 0.], [1., 0., 1.], [1., 0., 0.]])
        model = ItemKNN(neighbors=2, shrink=0).fit(r)
        expected = np.array([[0, 1/np.sqrt(3), 1/np.sqrt(3)],
                             [1/np.sqrt(3), 0, 0], [1/np.sqrt(3), 0, 0]])
        np.testing.assert_allclose(model.similarity.toarray(), expected, atol=1e-12)
        np.testing.assert_allclose(model.score([0]), r.toarray()[[0]] @ expected, atol=1e-12)

    def test_temporal_protocol_uses_only_past_catalog_and_seen_items(self):
        rows = [(1, str(i), '2011-08-01', 'a') for i in range(13)]
        rows += [(2, '0', '2011-08-02', 'b'), (2, '1', '2011-09-01', 'c'),
                 (2, '1', '2011-11-01', 'd'), (2, '2', '2011-11-01', 'd'),
                 (2, 'NEW', '2011-11-01', 'd'), (999, '0', '2011-11-01', 'e')]
        frame = pd.DataFrame(rows, columns=['CustomerID', 'StockCode', 'InvoiceDate', 'InvoiceNo'])
        frame['InvoiceDate'] = pd.to_datetime(frame.InvoiceDate)
        p = make_protocol(frame)
        self.assertNotIn('NEW', p['items'])
        self.assertNotIn(999, p['users'])
        u = int(np.where(p['users'] == 2)[0][0])
        np.testing.assert_array_equal(p['items'][p['validation'][u]], ['1'])
        np.testing.assert_array_equal(p['items'][p['test'][u]], ['2'])

    def test_blas_matches_independent_contraction(self):
        rng = np.random.default_rng(42)
        a, b = rng.normal(size=(128, 64)), rng.normal(size=(64, 200))
        np.testing.assert_allclose(matmul(a, b), np.einsum('ik,kj->ij', a, b, optimize=False), atol=1e-12)

    def test_als_step_matches_dense_normal_equations(self):
        r = sparse.csr_matrix([[1., 0., 3.], [0., 2., 1.]])
        model = RepeatTargetIALS(beta=.5, factors=2, reg=.7, alpha=5.)
        p = model.targets(r)
        c = r.copy()
        c.data = 1 + model.alpha * np.log1p(c.data)
        y = np.array([[.2, .3], [-.1, .7], [.5, -.4]])
        actual = model._step(y, c, p)
        dense_c = np.ones(r.shape)
        dense_c[r.toarray() > 0] = c.data
        for u in range(2):
            expected = np.linalg.solve(y.T @ np.diag(dense_c[u]) @ y + .7*np.eye(2),
                                       y.T @ np.diag(dense_c[u]) @ p.toarray()[u])
            np.testing.assert_allclose(actual[u], expected)

    def test_custom_target_is_not_confidence_only(self):
        r = sparse.csr_matrix([[1., 0., 3.]])
        np.testing.assert_allclose(RepeatTargetIALS(beta=0).targets(r).data, [1, 1])
        np.testing.assert_allclose(RepeatTargetIALS(beta=.5).targets(r).data, [.75, 1])
        base = IALS(factors=2, iterations=2).fit(r)
        equivalent = RepeatTargetIALS(beta=0, factors=2, iterations=2).fit(r)
        np.testing.assert_allclose(base.score([0]), equivalent.score([0]))

    def test_ease_matches_constrained_ridge(self):
        r = sparse.csr_matrix([[1., 1., 0.], [0., 1., 1.], [1., 0., 0.]])
        model = EASE(reg=2.).fit(r)
        x = r.toarray()
        for j in range(3):
            idx = np.arange(3) != j
            expected = np.linalg.solve(x[:, idx].T @ x[:, idx] + 2*np.eye(2), x[:, idx].T @ x[:, j])
            np.testing.assert_allclose(model.weights[idx, j], expected)
        np.testing.assert_array_equal(np.diag(model.weights), np.zeros(3))

    def test_fm_formula_gradient_and_decomposed_scoring(self):
        rng = np.random.default_rng(5)
        x = sparse.csr_matrix([[1., .3, 0., 1.], [0., 1., 1., .5]])
        w, v = rng.normal(size=4), rng.normal(size=(4, 3))
        scores, xv = FM.forward(x, w, v, .4)
        expected = []
        for row in x.toarray():
            expected.append(.4 + row @ w + sum(row[j]*row[k]*(v[j]@v[k]) for j in range(4) for k in range(j+1,4)))
        np.testing.assert_allclose(scores, expected)
        err = np.array([.2, -.3])
        grad = x.T @ (err[:, None]*xv) - (x.multiply(x).T @ err)[:, None]*v
        eps = 1e-6
        for j in range(4):
            for k in range(3):
                vp, vm = v.copy(), v.copy()
                vp[j, k] += eps
                vm[j, k] -= eps
                numerical = (FM.forward(x,w,vp,.4)[0] @ err - FM.forward(x,w,vm,.4)[0] @ err)/(2*eps)
                self.assertAlmostEqual(grad[j,k], numerical, places=6)
        r = sparse.csr_matrix([[1., 0., 1.], [0., 1., 0.]])
        uf = sparse.csr_matrix([[1., 0., .5], [0., 1., .7]])
        itf = sparse.csr_matrix([[1., 0., 0., .4], [0., 1., 0., .2], [0., 0., 1., .9]])
        model = FM(factors=3, epochs=2, negatives=1).fit(r, uf, itf)
        for u in range(2):
            features = sparse.hstack([uf[np.repeat(u, 3)], itf], format='csr')
            expected, _ = model.forward(features, model.w, model.v, model.bias)
            np.testing.assert_allclose(model.score([u])[0], expected)

    def test_metrics_masks_and_ties(self):
        class Constant:
            def score(self, users):
                return np.ones((len(users), 12))
        seen = sparse.csr_matrix(([1.], ([0], [0])), shape=(1,12))
        rec = recommend(Constant(), [0], seen)[0]
        np.testing.assert_array_equal(rec, np.arange(1,11))
        result, _ = evaluate(Constant(), {0: np.array([1,11])}, seen)
        self.assertAlmostEqual(result['Recall@10'], .5)
        self.assertAlmostEqual(result['Precision@10'], .1)
        self.assertAlmostEqual(result['NDCG@10'], 1 / (1 + 1/np.log2(3)))


if __name__ == '__main__':
    unittest.main()
