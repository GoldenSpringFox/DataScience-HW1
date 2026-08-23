"""Seed-stability check for the symNMF communities (open item from devlog 6.3d).

Reproduces the notebook's graph exactly (top-15 union sparsification, Jaccard gate 0.03,
basics + 12 format staples excluded, edge weight t-score x log(1+lift)), fits symNMF at
several seeds, and measures how much the result moves. A degree-preserving rewired graph
gives the floor: whatever stability a structureless graph of the same shape produces.
"""
import sys, json, time
from pathlib import Path
import numpy as np, scipy.sparse as sparse
import networkx as nx
from sklearn.metrics import adjusted_rand_score

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb
from edhcut.analysis.symnmf_packages import fit_symnmf
from edhcut.analysis.nmf_packages import match_topic_stability

K = 200
SEEDS = [42, 7, 13, 99, 2024]


def build_graph():
    """The gated card graph, built exactly as `kb`/the notebooks build it."""
    ci, tscore, lift, n2r, r2n, dc = kb.load()
    adj, excl = kb.build(ci, tscore, n2r)              # top-15 union, basics+staples out
    binary = (adj > 0).astype(np.int32).tocsr()
    degree = np.asarray(binary.sum(axis=1)).ravel()
    up = sparse.triu(adj, k=1).tocoo()
    u, v, w = up.row, up.col, up.data
    shared = np.asarray(binary[u].multiply(binary[v]).sum(axis=1)).ravel()
    jac = shared / (degree[u] + degree[v] - shared)
    keep = jac >= kb.MIN_JACCARD
    u, v, w = u[keep], v[keep], w[keep]
    # notebook edge weight: t-score x log(1 + lift)
    L = lift.tocsr()
    combined = w * np.log1p(np.maximum(np.asarray(L[u, v]).ravel(), 0.0))
    n = adj.shape[0]
    half = sparse.coo_matrix((combined, (u, v)), shape=(n, n)).tocsr()
    S = (half + half.T).tocsr()
    alive = np.flatnonzero(np.diff(S.indptr) > 0)
    S = S[alive][:, alive]
    return S, alive


def rewired(S, seed=0):
    """Degree-preserving configuration model with the same weight multiset."""
    binary = (S > 0).tocsr()
    deg = np.asarray(binary.sum(axis=1)).ravel().astype(int)
    G = nx.configuration_model(list(deg), seed=seed)
    G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
    u, v = np.array(G.edges()).T
    rng = np.random.default_rng(seed)
    w_real = sparse.triu(S, k=1).tocoo().data
    w = rng.choice(w_real, size=len(u), replace=True)
    n = S.shape[0]
    half = sparse.coo_matrix((w, (u, v)), shape=(n, n)).tocsr()
    return (half + half.T).tocsr()


def run(S, label, seeds):
    """Cluster the graph under several random seeds and report pairwise agreement — the check that
    the packages are a property of the graph rather than of one lucky seed. Writes
    `seed_stability.json`, whose numbers are quoted in Question 2."""
    fits = {}
    for s in seeds:
        t0 = time.perf_counter()
        fits[s] = fit_symnmf(S, K, seed=s)
        print(f"  [{label}] seed {s:5d}: {fits[s].n_iter} iters, "
              f"residual {fits[s].relative_residual:.4f}, {time.perf_counter()-t0:.0f}s", flush=True)
    base = seeds[0]
    rows = []
    for s in seeds[1:]:
        soft = match_topic_stability(fits[base].H.T, fits[s].H.T)
        hard = adjusted_rand_score(fits[base].H.argmax(axis=1), fits[s].H.argmax(axis=1))
        rows.append((s, soft, hard))
        print(f"  [{label}] {base} vs {s}: topic-match {soft:.3f}, ARI {hard:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    S, alive = build_graph()
    print(f"graph: {S.shape[0]:,} nodes, {S.nnz // 2:,} edges", flush=True)

    print("\nREAL GRAPH")
    real = run(S, "real", SEEDS)

    print("\nDEGREE-PRESERVING REWIRED NULL")
    null = run(rewired(S), "null", SEEDS[:3])

    out = {
        "k": K, "seeds": SEEDS, "nodes": int(S.shape[0]), "edges": int(S.nnz // 2),
        "real": [{"seed": s, "topic_match": m, "ari": a} for s, m, a in real],
        "null": [{"seed": s, "topic_match": m, "ari": a} for s, m, a in null],
    }
    (SP / "seed_stability.json").write_text(json.dumps(out, indent=1))
    rm = np.mean([r[1] for r in real]); ra = np.mean([r[2] for r in real])
    nm = np.mean([r[1] for r in null]); na = np.mean([r[2] for r in null])
    print(f"\nSUMMARY  real: topic-match {rm:.3f}, ARI {ra:.3f}")
    print(f"         null: topic-match {nm:.3f}, ARI {na:.3f}")
