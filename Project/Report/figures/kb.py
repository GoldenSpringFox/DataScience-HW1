"""Shared loader and graph helpers for the report figures — the pieces every figure script needs
so that all of them describe the *same* card graph the analysis chapters do, rather than each
re-deriving its own approximation.

`load()` returns the two matrices Question 1 settles on (the near-uniform-weighted,
colour-conditioned t-score and plain lift) plus the card index and its name/row lookups. Callers
form the synergy score themselves as `t-score * log(1 + lift)` — the product Question 1 argues for
— then build the graph with the two helpers below: `sparsify_top_k_union` (keep each card's
`TOP_K` strongest partners, symmetrise by union) and `jaccard_all` (per-edge neighbourhood
overlap, gated at `MIN_JACCARD`). `BASICS` and `STAPLES` are the node exclusions that go with it.

Kept identical to the two community notebooks' own §2 on purpose; see `docs/devlog/6.3d`.
"""
import numpy as np, pandas as pd, scipy.sparse as sparse
from pathlib import Path

# Resolved from this file, not the working directory, so the scripts run from anywhere:
# Project/Report/figures/ -> Project/EDHCut/data/kb/dev
KB = Path(__file__).resolve().parents[2] / "EDHCut" / "data" / "kb" / "dev"

BASICS = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
# the 12 detected format staples (docs/devlog/6.3d), max_lift < 3.5 and playrate >= 10%
STAPLES = ["Sol Ring", "Command Tower", "Arcane Signet", "Exotic Orchard",
           "Swords to Plowshares", "Path to Exile", "Swiftfoot Boots", "Lightning Greaves",
           "Reliquary Tower", "Rogue's Passage", "Path of Ancestry", "Fellwar Stone"]

TOP_K = 15
MIN_JACCARD = 0.03


def load():
    ci = pd.read_parquet(KB / "card_index.parquet")
    tscore = sparse.load_npz(KB / "tscore_color_global_nearuniform.npz").tocsr()
    lift = sparse.load_npz(KB / "lift_global.npz").tocsr()
    name2row = dict(zip(ci["name"], ci["row"]))
    row2name = dict(zip(ci["row"], ci["name"]))
    deck_count = ci.set_index("row")["deck_count"].to_dict()
    return ci, tscore, lift, name2row, row2name, deck_count


def sparsify_top_k_union(matrix, top_k, min_tscore, exclude):
    """Keep each row's top_k strongest entries above min_tscore; symmetrise by union."""
    m = matrix.tocsr(copy=True)
    m.data[m.data <= min_tscore] = 0
    m.eliminate_zeros()
    if exclude is not None and exclude.any():
        keep = ~exclude
        d = sparse.diags(keep.astype(m.dtype))
        m = (d @ m @ d).tocsr()
        m.eliminate_zeros()
    rows, cols, vals = [], [], []
    for i in range(m.shape[0]):
        s, e = m.indptr[i], m.indptr[i + 1]
        if e == s:
            continue
        idx = m.indices[s:e]
        dat = m.data[s:e]
        if len(dat) > top_k:
            sel = np.argpartition(-dat, top_k)[:top_k]
            idx, dat = idx[sel], dat[sel]
        rows.extend([i] * len(idx)); cols.extend(idx); vals.extend(dat)
    half = sparse.coo_matrix((vals, (rows, cols)), shape=m.shape).tocsr()
    return half.maximum(half.T).tocsr()


def jaccard_all(adjacency):
    """Per-edge Jaccard overlap of endpoint neighbourhoods. Returns (u, v, jaccard, weight)."""
    binary = (adjacency > 0).astype(np.int32).tocsr()
    degree = np.asarray(binary.sum(axis=1)).ravel()
    upper = sparse.triu(adjacency, k=1).tocoo()
    u, v, w = upper.row, upper.col, upper.data
    shared = np.asarray(binary[u].multiply(binary[v]).sum(axis=1)).ravel()
    jac = shared / (degree[u] + degree[v] - shared)
    return u, v, jac, w, shared, degree


def build(ci, tscore, name2row):
    excl = np.zeros(tscore.shape[0], dtype=bool)
    for n in BASICS + STAPLES:
        if n in name2row:
            excl[name2row[n]] = True
    adj = sparsify_top_k_union(tscore, TOP_K, 0.0, excl)
    return adj, excl
