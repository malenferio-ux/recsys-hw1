"""Reproducible UCI download, cleaning, temporal protocol and train-only features."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import re
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
TRAIN_END = pd.Timestamp('2011-09-01')
VALID_END = pd.Timestamp('2011-11-01')


def load_data(cache=None):
    cache = Path(cache or os.environ.get('RECSYS_CACHE', '.cache'))
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'online-retail.zip'
    if not archive.exists():
        tmp = archive.with_suffix('.partial')
        urllib.request.urlretrieve(SOURCE, tmp)
        tmp.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected_file = Path(__file__).with_name('dataset_manifest.json')
    if expected_file.exists():
        expected = json.loads(expected_file.read_text())['sha256']
        if digest != expected:
            raise ValueError('Dataset checksum changed; inspect the source before rerunning.')
    parsed = cache / ('raw-' + digest[:12] + '.pkl')
    if parsed.exists():
        raw = pd.read_pickle(parsed)  # Only a locally generated cache; never download pickle.
    else:
        with zipfile.ZipFile(archive) as z:
            with z.open('Online Retail.xlsx') as f:
                raw = pd.read_excel(f, engine='openpyxl', dtype={'StockCode': str, 'InvoiceNo': str})
        raw.to_pickle(parsed)
    return raw, digest


def clean_data(raw):
    audit = [('raw', len(raw))]
    d = raw.dropna(subset=['CustomerID', 'StockCode', 'InvoiceDate']).copy()
    audit.append(('required IDs/timestamp', len(d)))
    d = d.drop_duplicates()
    audit.append(('exact duplicate removal', len(d)))
    d = d[(d.Quantity > 0) & (d.UnitPrice > 0) & ~d.InvoiceNo.str.upper().str.startswith('C')].copy()
    audit.append(('positive purchases', len(d)))
    # Goods have five digits and optionally an alphabetic suffix; remove postage/fees.
    d = d[d.StockCode.str.fullmatch(r'\d{5}[A-Za-z]*')].copy()
    audit.append(('product stock codes', len(d)))
    d['CustomerID'] = d.CustomerID.astype('int64')
    d['Description'] = d.Description.fillna('').str.strip().str.upper()
    d['Country'] = d.Country.fillna('Unknown')
    d = d.sort_values(['InvoiceDate', 'InvoiceNo', 'CustomerID', 'StockCode'], kind='stable').reset_index(drop=True)
    return d, pd.DataFrame(audit, columns=['step', 'rows'])


def make_protocol(d):
    train_raw = d[d.InvoiceDate < TRAIN_END].copy()
    users = np.sort(train_raw.CustomerID.unique())
    items = np.sort(train_raw.StockCode.unique())
    umap, imap = {v: k for k, v in enumerate(users)}, {v: k for k, v in enumerate(items)}

    def encode(frame):
        out = frame[frame.CustomerID.isin(umap) & frame.StockCode.isin(imap)].copy()
        out['u'] = out.CustomerID.map(umap)
        out['i'] = out.StockCode.map(imap)
        return out

    train = encode(train_raw)
    # A purchase event is one distinct invoice-user-item, regardless of split lines.
    events = train[['u', 'i', 'InvoiceNo']].drop_duplicates()
    counts = sparse.csr_matrix((np.ones(len(events)), (events.u, events.i)), shape=(len(users), len(items)))
    binary = counts.copy()
    binary.data[:] = 1
    validation_raw = d[(d.InvoiceDate >= TRAIN_END) & (d.InvoiceDate < VALID_END)]
    test_raw = d[d.InvoiceDate >= VALID_END]
    validation, test = encode(validation_raw), encode(test_raw)
    history_test = binary.copy()
    if len(validation):
        history_test += sparse.csr_matrix((np.ones(len(validation)), (validation.u, validation.i)), shape=counts.shape)
        history_test.data[:] = 1

    def labels(frame, seen):
        result = {}
        for u, group in frame.groupby('u', sort=True):
            candidates = np.setdiff1d(group.i.unique(), seen[u].indices)
            if len(candidates) and counts.shape[1] - seen[u].nnz >= 10:
                result[int(u)] = candidates
        return result

    vtruth, ttruth = labels(validation, binary), labels(test, history_test)
    audit = []
    for name, source, known, truth in [('validation', validation_raw, validation, vtruth), ('test', test_raw, test, ttruth)]:
        audit.append(dict(split=name, raw_rows=len(source), warm_rows=len(known),
                          cold_user_rows=int((~source.CustomerID.isin(umap)).sum()),
                          cold_item_rows=int((~source.StockCode.isin(imap)).sum()),
                          warm_users=known.u.nunique(), evaluated_users=len(truth),
                          relevant_pairs=sum(map(len, truth.values()))))
    return dict(train=train, counts=counts, users=users, items=items,
                validation=vtruth, test=ttruth, seen_validation=binary,
                seen_test=history_test, audit=pd.DataFrame(audit))


def make_features(train, n_users, n_items, metadata=True, max_tokens=128):
    uf, itf = sparse.eye(n_users, format='csr'), sparse.eye(n_items, format='csr')
    info = {}
    if not metadata:
        return uf, itf, info
    latest_country = train.groupby('u', sort=True).Country.last().reindex(range(n_users))
    countries = sorted(latest_country.unique())
    country_codes = latest_country.map({v: k for k, v in enumerate(countries)}).to_numpy()
    country = sparse.csr_matrix((np.ones(n_users), (np.arange(n_users), country_codes)), shape=(n_users, len(countries)))
    descriptions = train.groupby('i').Description.last().reindex(range(n_items)).fillna('')
    token_sets = [set(re.findall(r'[A-Z]{3,}', x)) for x in descriptions]
    frequencies = Counter(t for ts in token_sets for t in ts)
    vocab = sorted(frequencies, key=lambda t: (-frequencies[t], t))[:max_tokens]
    lookup = {t: k for k, t in enumerate(vocab)}
    rows, cols, values = [], [], []
    for i, tokens in enumerate(token_sets):
        selected = sorted(lookup[t] for t in tokens if t in lookup)
        for j in selected:
            rows.append(i)
            cols.append(j)
            values.append(1 / np.sqrt(len(selected)))
    title = sparse.csr_matrix((values, (rows, cols)), shape=(n_items, len(vocab)))
    price = train.groupby('i').UnitPrice.median().reindex(range(n_items)).to_numpy()
    edges = np.unique(np.quantile(np.log1p(price), [.2, .4, .6, .8]))
    bins = np.searchsorted(edges, np.log1p(price))
    price_features = sparse.csr_matrix((np.ones(n_items), (np.arange(n_items), bins)), shape=(n_items, len(edges) + 1))
    uf = sparse.hstack([uf, country], format='csr')
    itf = sparse.hstack([itf, title, price_features], format='csr')
    info = dict(countries=countries, vocabulary=vocab, log_price_edges=edges.tolist())
    return uf, itf, info
