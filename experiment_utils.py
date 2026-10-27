"""Small validation grid; test labels never enter model selection."""
import json
import time
import numpy as np
import pandas as pd
from models.item_knn import ItemKNN
from models.ials import IALS
from models.ease import EASE
from models.fm import FM
from models.custom_ials import RepeatTargetIALS
from evaluation import evaluate

SEARCH = {
    'ItemKNN': [dict(neighbors=k, shrink=s) for k in [50, 150] for s in [0, 20]],
    'IALS': [dict(factors=k, reg=r, alpha=20., iterations=12, seed=42)
             for k in [32, 64] for r in [1., 10.]],
    'EASE': [dict(reg=r) for r in [30., 100., 300.]],
    'FM': [dict(factors=k, reg=r, epochs=10, lr=.01, negatives=3, seed=42)
           for k in [16, 32] for r in [1e-5, 1e-4]],
    # Other IALS settings are fixed to the selected baseline for an isolated comparison.
    'RepeatTargetIALS': [dict(beta=b) for b in [.25, .5, .75]],
}
CLASSES = dict(ItemKNN=ItemKNN, IALS=IALS, EASE=EASE, FM=FM, RepeatTargetIALS=RepeatTargetIALS)


def tune(name, protocol, features=None, base_params=None):
    rows, best_model, best_params, best_value = [], None, None, -np.inf
    for options in SEARCH[name]:
        params = {**(base_params or {}), **options}
        start = time.perf_counter()
        model = CLASSES[name](**params)
        if name == 'FM':
            model.fit(protocol['counts'], *features)
        else:
            model.fit(protocol['counts'])
        train_seconds = time.perf_counter() - start
        metrics, _ = evaluate(model, protocol['validation'], protocol['seen_validation'])
        rows.append(dict(model=name, params=json.dumps(params, sort_keys=True),
                         train_seconds=train_seconds, **metrics))
        print(f'{name}: {params} → NDCG@10={metrics["NDCG@10"]:.5f}, {train_seconds:.1f}s', flush=True)
        # Ties choose the first predetermined configuration, never test results.
        if metrics['NDCG@10'] > best_value:
            best_model, best_params, best_value = model, params, metrics['NDCG@10']
    return best_model, best_params, pd.DataFrame(rows)
