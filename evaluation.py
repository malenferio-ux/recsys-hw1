"""Full-catalog deterministic top-K and macro metrics, shared by every model."""
import numpy as np
import pandas as pd


def recommend(model, users, seen, k=10):
    scores = np.asarray(model.score(np.asarray(users)), dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite model scores')
    for row, u in enumerate(users):
        scores[row, seen[u].indices] = -np.inf
        if scores.shape[1] - seen[u].nnz < k:
            raise ValueError('Fewer than K eligible items')
    # Stable descending sort: equal scores resolved by ascending encoded StockCode.
    return np.argsort(-scores, axis=1, kind='stable')[:, :k]


def evaluate(model, truth, seen, k=10, batch_size=128):
    if not truth:
        raise ValueError('Empty evaluation cohort')
    users = np.array(sorted(truth))
    discounts = 1 / np.log2(np.arange(2, k + 2))
    rows, covered = [], set()
    for begin in range(0, len(users), batch_size):
        batch = users[begin:begin + batch_size]
        recs = recommend(model, batch, seen, k)
        for u, rec in zip(batch, recs):
            hit = np.isin(rec, truth[u])
            rows.append((u, float(hit @ discounts / discounts[:min(k, len(truth[u]))].sum()),
                         float(hit.sum() / len(truth[u])), float(hit.mean())))
            covered.update(rec.tolist())
    per_user = pd.DataFrame(rows, columns=['u', 'NDCG@10', 'Recall@10', 'Precision@10'])
    metrics = per_user.drop(columns='u').mean().to_dict()
    metrics['Coverage@10'] = len(covered) / seen.shape[1]
    metrics['users'] = len(users)
    return metrics, per_user


def paired_bootstrap(a, b, seed=42, repeats=2000):
    joined = a.set_index('u')[['NDCG@10']].join(b.set_index('u')[['NDCG@10']], lsuffix='_a', rsuffix='_b', how='inner')
    assert len(joined) == len(a) == len(b)
    delta = (joined.iloc[:, 0] - joined.iloc[:, 1]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return dict(delta=float(delta.mean()), low=float(np.quantile(means, .025)), high=float(np.quantile(means, .975)))
