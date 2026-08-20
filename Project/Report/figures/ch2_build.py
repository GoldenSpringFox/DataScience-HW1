"""Rebuild the notebook's exact symNMF community assignment, and cache everything Chapter 2 needs.

Matches notebooks/symNMF_communities.ipynb: top-15 union sparsification on the near-uniform-weighted
colour-conditioned t-score, the 12 Basic-supertype lands and the 12 format staples excluded as
nodes, Jaccard gate 0.03, edge weight t-score x log(1+lift), symNMF k=200 seed 42, membership
share >= 0.10 capped at 5 topics, soft k-core floor k=4.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sparse
from itertools import combinations

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb
from edhcut.db import connect
from edhcut.config import CONFIG
from edhcut.analysis.communities import basic_land_mask, sparsify_top_k_union
from edhcut.analysis.symnmf_packages import fit_symnmf

K, SEED, MIN_SHARE, MAX_TOPICS, KCORE = 200, 42, 0.10, 5, 4
MIN_TAG_CARDS, THEME_SHARE, THEME_LIFT = 4, 0.10, 8.0
BROAD_SHARE, BROAD_LIFT = 0.20, 3.0   # rescue for labels too narrow to describe their community


def jaccard_gate(adjacency, threshold: float):
    """Second gate, verbatim from the notebook."""
    binary = (adjacency > 0).astype(np.int32).tocsr()
    degree = np.asarray(binary.sum(axis=1)).ravel()
    upper = sparse.triu(adjacency, k=1).tocoo()
    u, v, w = upper.row, upper.col, upper.data
    shared = np.asarray(binary[u].multiply(binary[v]).sum(axis=1)).ravel()
    jac = shared / (degree[u] + degree[v] - shared)
    survives = jac >= threshold
    n = adjacency.shape[0]
    half = sparse.coo_matrix((w[survives], (u[survives], v[survives])), shape=(n, n)).tocsr()
    full = (half + half.T).tocsr()
    connected = np.asarray((full != 0).sum(axis=1)).ravel() > 0
    return full[connected][:, connected], connected


def build_incidence(conn, ci):
    """Binary deck x card incidence, exactly what the notebook's edge_lift measures lift from.

    NOT the cached lift_global.npz, which is built from precon-down-weighted counts -- using that
    instead changes every edge weight and therefore the whole factorization.
    """
    row_of = {o: i for i, o in enumerate(ci["oracle_id"])}
    deck_ids, card_rows = [], []
    seen = {}
    for deck_id, oracle_id in conn.execute("SELECT deck_id, oracle_id FROM deck_cards"):
        r = row_of.get(oracle_id)
        if r is None:
            continue
        d = seen.setdefault(deck_id, len(seen))
        deck_ids.append(d); card_rows.append(r)
    data = np.ones(len(deck_ids), dtype=np.float64)
    inc = sparse.coo_matrix((data, (deck_ids, card_rows)),
                            shape=(len(seen), len(ci))).tocsc()
    inc.data[:] = 1.0
    return inc


def edge_lift(incidence, card_decks, rows, cols, orig_index):
    """Lift for each edge, verbatim from the notebook."""
    a, b = orig_index[rows], orig_index[cols]
    observed = np.asarray(incidence[:, a].multiply(incidence[:, b]).sum(axis=0)).ravel().astype(float)
    expected = card_decks[a] * card_decks[b] / incidence.shape[0]
    return np.divide(observed, expected, out=np.zeros_like(observed), where=expected > 0)


def build():
    ci, tscore, lift, n2r, r2n, dc = kb.load()
    with connect(CONFIG.paths.db_path) as conn:
        basics = basic_land_mask(conn, ci)
        colour = dict(conn.execute("SELECT oracle_id, color_identity FROM cards"))
        tags = pd.read_sql("SELECT oracle_id, tag FROM card_tags", conn)
        deck_total = conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0]
        incidence = build_incidence(conn, ci)
    card_decks = ci["deck_count"].to_numpy().astype(float)
    print(f"incidence: {incidence.shape[0]:,} decks x {incidence.shape[1]:,} cards, {incidence.nnz:,} entries")
    excl = basics.copy()
    for n in kb.STAPLES:
        if n in n2r:
            excl[n2r[n]] = True
    print(f"excluded as nodes: {int(basics.sum())} basics + {len(kb.STAPLES)} staples")

    # exactly the notebook's call order, using the library's own sparsification
    adjacency, kept = sparsify_top_k_union(tscore, top_k=kb.TOP_K, min_tscore=0.0, exclude=excl)
    adjacency, connected = jaccard_gate(adjacency, kb.MIN_JACCARD)
    kept[np.flatnonzero(kept)[~connected]] = False
    up = sparse.triu(adjacency, k=1).tocoo()
    combined = up.data * np.log1p(edge_lift(incidence, card_decks, up.row, up.col, np.flatnonzero(kept)))
    half = sparse.coo_matrix((combined, (up.row, up.col)), shape=adjacency.shape).tocsr()
    S = (half + half.T).tocsr()
    alive = np.flatnonzero(kept)
    print(f"graph: {S.shape[0]:,} nodes, {S.nnz // 2:,} edges")

    fit = fit_symnmf(S, K, seed=SEED)
    H = fit.H
    print(f"symNMF: {fit.n_iter} iters, residual {fit.relative_residual:.4f}, {fit.fit_seconds:.0f}s")

    share = H / np.maximum(H.sum(axis=1, keepdims=True), 1e-12)
    member = share >= MIN_SHARE
    # cap at MAX_TOPICS strongest
    for i in np.flatnonzero(member.sum(axis=1) > MAX_TOPICS):
        keep_t = np.argsort(-share[i])[:MAX_TOPICS]
        row = np.zeros(K, bool); row[keep_t] = True
        member[i] = row & (share[i] >= MIN_SHARE)

    # soft k-core: a card stays in a topic only while it has >= KCORE edges to other members
    binS = (S > 0).tocsr()
    floored = member.copy()
    for t in range(K):
        idx = np.flatnonzero(floored[:, t])
        while len(idx):
            sub = binS[idx][:, idx]
            d = np.asarray(sub.sum(axis=1)).ravel()
            bad = d < KCORE
            if not bad.any():
                break
            idx = idx[~bad]
        drop = np.setdiff1d(np.flatnonzero(floored[:, t]), idx)
        floored[drop, t] = False

    n_topics = floored.sum(axis=1)
    primary = np.where(n_topics > 0, np.where(floored, share, -1).argmax(axis=1), -1)
    print(f"unassigned: {(n_topics == 0).sum():,} ({(n_topics == 0).mean():.1%})   "
          f"mean topics/card {n_topics[n_topics > 0].mean():.2f}   "
          f">1 topic: {(n_topics > 1).sum() / max((n_topics > 0).sum(), 1):.0%}")

    cards = pd.DataFrame({
        "oracle_id": ci["oracle_id"].values[alive],
        "name": ci["name"].values[alive],
        "deck_count": ci["deck_count"].values[alive],
        "community": primary,
        "n_topics": n_topics,
    })
    cards["color_identity"] = cards["oracle_id"].map(colour).fillna("C")
    return cards, share, floored, S, tags, deck_total


def name_communities(cards, tags):
    tags = tags[tags["oracle_id"].isin(set(cards["oracle_id"]))].copy()
    assigned = cards[cards["community"] >= 0]
    tags["community"] = tags["oracle_id"].map(dict(zip(assigned["oracle_id"], assigned["community"])))
    tags = tags[tags["community"].notna()]
    tags["community"] = tags["community"].astype(int)
    sizes = assigned["community"].value_counts()
    tag_total = tags.groupby("tag")["oracle_id"].nunique()
    st = tags.groupby(["community", "tag"])["oracle_id"].nunique().rename("k").reset_index()
    st["share"] = st["k"] / st["community"].map(sizes)
    st["lift"] = st["share"] / (st["tag"].map(tag_total) / len(cards))
    st["score"] = st["share"] * np.log1p(st["lift"])
    ok = st[(st["k"] >= MIN_TAG_CARDS) & (st["share"] >= THEME_SHARE) & (st["lift"] >= THEME_LIFT)]
    names = {}
    for cid, grp in ok.groupby("community"):
        top = grp.nlargest(2, "score")["tag"].tolist()
        # A distinctive tag can still be too narrow to describe the community it names -- e.g. a
        # 49-card white lifegain pile whose only tag clearing lift>=8 covers 7 cards. When that
        # happens, lead with the tag that actually covers the community instead.
        if grp["share"].max() < BROAD_SHARE:
            broad = st[(st["community"] == cid) & (st["k"] >= MIN_TAG_CARDS)
                       & (st["lift"] >= BROAD_LIFT)].nlargest(1, "share")
            if len(broad) and broad.iloc[0]["tag"] not in top:
                top = [broad.iloc[0]["tag"]] + top[:1]
        names[cid] = "/".join(top)
    return names, sizes


if __name__ == "__main__":
    cards, share, floored, S, tags, deck_total = build()
    names, sizes = name_communities(cards, tags)
    cards["theme"] = cards["community"].map(names).fillna("(generic)")
    with open(SP / "ch2.pkl", "wb") as f:
        pickle.dump(dict(cards=cards, share=share, floored=floored, S=S, names=names), f)
    print(f"\nnamed {len(names)} of {cards['community'].nunique()} communities")
    print("\nlargest communities:")
    for cid, sz in sizes.head(12).items():
        print(f"  {cid:4d}  {sz:4d}  {names.get(cid, '(generic)')}")
    print("\ndragon communities:")
    for cid, nm in names.items():
        if "dragon" in nm:
            mem = cards[cards["community"] == cid]
            print(f"  {cid:4d}  {len(mem):4d}  {nm}")
            print("       ", ", ".join(mem.nlargest(8, "deck_count")["name"]))
    ls = cards[cards["name"] == "Lion Sash"]
    if len(ls):
        i = ls.index[0]
        tops = np.argsort(-share[i])[:4]
        print("\nLion Sash memberships:")
        for t in tops:
            print(f"   topic {t:3d}  {share[i, t]:.0%}  in_floor={floored[i, t]}  {names.get(t, '(generic)')}")
