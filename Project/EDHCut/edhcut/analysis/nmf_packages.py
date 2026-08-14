"""Soft/overlapping synergy packages via NMF over the deck x card matrix (plan
`EDHCut_PLAN.md` task 6.3's follow-up, decided once the hard-partition Leiden rebuild in
`communities.py` was validated): a card can genuinely belong to more than one package (Ashnod's
Altar is a sac outlet *and* a combo piece), which `communities.py`'s hard `cluster_id` can't
represent. NMF additionally yields deck -> package proportions directly (the `W` factor), which
is what task 7.2 wants for "dominant synergy clusters (% of deck per cluster)" -- a hard-cluster
scheme would need a separate aggregation step to approximate the same thing.

**Deck subsampling, not post-hoc weighting.** The same oversampling problem
`communities.py` fixed with `compute_near_uniform_weights` applies here -- Krenko/Kyler/Yenna's
thousands of decks vs. ~9 typical would dominate the factorization exactly as they dominated the
graph -- but NMF's input is a literal deck x card presence matrix, not a derived association
matrix, so the natural fix is different. Every commander slot is capped at
`MAX_DECKS_PER_SLOT` (25, matching the metagame harvest's own per-commander ceiling), subsampled
once with a fixed seed before any k-sweep or stability check runs, rather than reweighting rows
post hoc -- sklearn's `NMF` has no native per-sample weight, and scaling a binary row by
`sqrt(weight)` would corrupt the 0/1 presence semantics the factorization relies on. A real
consequence, not a bug: a card whose only >=3-deck-count-qualifying appearances all happen to
sit inside an oversampled slot's *excess* volume can be dropped from the training matrix
entirely by the subsample (tracked in `NMFBuildStats.n_cards_dropped_no_signal`).

**All lands excluded, not just basics -- a deliberate deviation from `communities.py`'s land
handling, found live, not assumed.** The first real build (basics-only excluded, mirroring
`communities.py`) produced topics that read as color-pair mana bases, not synergy packages:
Cultivate split across five different topics, each topic's top members almost entirely *that
color pair's own dual/shock/check lands*, Cultivate itself buried at 11-26% share in each.
Root cause is specific to NMF's raw-presence factorization, not a repeat of any land-handling
mistake already ruled out for `communities.py`: a nonbasic land is present in the large majority
of *every* deck sharing its color pair regardless of theme, which is an extremely strong,
easy-to-fit block of correlated presence for Frobenius-loss NMF to spend components on --
`communities.py`'s pairwise t-score doesn't have this failure mode (a land's association with
any *one* specific other card stays weak/undifferentiated, so its edges don't concentrate on a
single archetype), which is why excluding only basics was sufficient there but not here.
Excluding every land (`card_index["is_land"]`, not just the basic-supertype subset
`communities.py`'s `basic_land_mask` targets) fixed it: re-verified live -- Ashnod's Altar now
lands 79% in a genuine aristocrats/sac-outlet topic (Skullclamp, Zulaport Cutthroat, Blood
Artist, Viscera Seer) and 10% in a reanimator/graveyard topic (Reanimate, Animate Dead, Entomb,
Living Death) -- exactly the two themes the plan's own motivating example named it for. Cultivate
lands cleanly in a ramp topic containing Kodama's Reach, the project's own standing regression
check.

**Binary presence, not quantity.** Commander is singleton for nonlands, so quantity is already
~always 1; lands (the exception, and the only cards that commonly run at qty > 1) are excluded
anyway. No precon-novelty per-card weighting (unlike `cooccurrence.py`'s pair counts) -- a
documented simplification, not yet revisited.

**k selection**: sweep `K_GRID`, for each k fit at `PRIMARY_SEED` (relative Frobenius
reconstruction error) and at `STABILITY_SEEDS` (topic-matching stability -- Hungarian-match each
extra run's topics against the primary run's by cosine similarity over card weights, mean
matched similarity is the stability score, the NMF analogue of `communities.py`'s seed-stability
ARI). Chosen k maximizes stability; ties broken toward the larger (finer-grained) k.

**Generalizes to any deck, not just the training subsample** (mirrors `communities.py`'s own
`cluster_of` design goal): `topic_proportions_for_deck` projects an arbitrary card set into
topic space via `sklearn.decomposition.non_negative_factorization` with `H` fixed
(`update_H=False`), not just a lookup into the training-only `nmf_deck_proportions.parquet`.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.metrics.pairwise import cosine_similarity

from edhcut.analysis.cooccurrence import KB_DEV_DIR, _resolve_card_name, load_card_index
from edhcut.config import CONFIG
from edhcut.db import connect

MAX_DECKS_PER_SLOT = 25
SUBSAMPLE_SEED = 42
K_GRID = [15, 20, 25, 30, 40]
PRIMARY_SEED = 42
STABILITY_SEEDS = [1, 2, 3]
NMF_MAX_ITER = 300
# A topic must explain at least this share of a card's total H-mass to count as a real
# membership -- keeps the long-format membership table from drowning in near-zero noise while
# still allowing genuine overlap (a card split ~50/50 between two topics keeps both).
MIN_MEMBERSHIP_SHARE = 0.10
MAX_TOPICS_PER_CARD = 5


def _sample_deck_ids(
    conn: sqlite3.Connection, *, max_decks_per_slot: int = MAX_DECKS_PER_SLOT, seed: int = SUBSAMPLE_SEED
) -> list[int]:
    """Every deck for slots with `<= max_decks_per_slot` decks; a fixed-seed random sample of
    exactly `max_decks_per_slot` for slots over that (roster commanders, mainly). One sample
    shared by the whole k-sweep and stability check, so every candidate k is compared on
    identical training data."""
    by_slot: dict[str, list[int]] = {}
    for deck_id, slot_key in conn.execute("SELECT deck_id, slot_key FROM decks ORDER BY deck_id"):
        by_slot.setdefault(slot_key, []).append(deck_id)

    rng = random.Random(seed)
    sampled: list[int] = []
    for deck_ids in by_slot.values():
        if len(deck_ids) <= max_decks_per_slot:
            sampled.extend(deck_ids)
        else:
            sampled.extend(rng.sample(deck_ids, max_decks_per_slot))
    return sorted(sampled)


def build_deck_card_matrix(
    conn: sqlite3.Connection, card_index: pd.DataFrame, deck_ids: list[int]
) -> sparse.csr_matrix:
    """Binary deck x card presence matrix (`len(deck_ids)` x `len(card_index)`), columns aligned
    to `card_index`'s own `row`. Cards outside `card_index` (below the pool's min-deck-count
    threshold) are silently skipped, same convention `cooccurrence.build_cooccurrence` uses."""
    oracle_to_col = dict(zip(card_index["oracle_id"], card_index["row"]))
    deck_to_row = {deck_id: i for i, deck_id in enumerate(deck_ids)}

    placeholders = ",".join("?" * len(deck_ids))
    rows_idx: list[int] = []
    cols_idx: list[int] = []
    for deck_id, oracle_id in conn.execute(
        f"SELECT deck_id, oracle_id FROM deck_cards WHERE deck_id IN ({placeholders})", deck_ids
    ):
        col = oracle_to_col.get(oracle_id)
        if col is None:
            continue
        rows_idx.append(deck_to_row[deck_id])
        cols_idx.append(col)

    data = np.ones(len(rows_idx), dtype=np.float64)
    matrix = sparse.csr_matrix((data, (rows_idx, cols_idx)), shape=(len(deck_ids), len(card_index)))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0  # a duplicate (deck, card) row would sum to >1; presence is binary
    return matrix


@dataclass
class KKSweepResult:
    k: int
    reconstruction_error: float
    stability: float


def _fit_nmf(X: sparse.csr_matrix, k: int, seed: int) -> NMF:
    model = NMF(
        n_components=k, init="nndsvda", solver="cd", random_state=seed, max_iter=NMF_MAX_ITER,
    )
    model.fit(X)
    return model


def match_topic_stability(h_a: np.ndarray, h_b: np.ndarray) -> float:
    """Hungarian-match `h_a`'s topics (rows) against `h_b`'s by cosine similarity over their card
    weight vectors, return the mean matched similarity -- 1.0 if every topic has a
    near-identical counterpart in the other run, low if the factorization is unstable across
    seeds. The NMF analogue of `communities.py`'s seed-stability ARI.

    Public (not `_`-prefixed) because `edhcut.analysis.symnmf_packages` reuses it verbatim for
    its own seed-stability check -- both `h_a`/`h_b` args are shape `(k, n_cards)` regardless of
    which module built them; `symnmf_packages`' own `H` is cards x k, so it passes `H.T`."""
    sim = cosine_similarity(h_a, h_b)
    row_idx, col_idx = linear_sum_assignment(-sim)
    return float(sim[row_idx, col_idx].mean())


def sweep_k(X: sparse.csr_matrix, *, k_grid: list[int] = K_GRID) -> list[KKSweepResult]:
    """Fit NMF at every k in `k_grid` (primary seed for reconstruction error, `STABILITY_SEEDS`
    for topic-matching stability). Runtime is `len(k_grid) * (1 + len(STABILITY_SEEDS))` NMF
    fits over the (subsampled) training matrix."""
    x_norm = sparse.linalg.norm(X)
    results = []
    for k in k_grid:
        primary = _fit_nmf(X, k, PRIMARY_SEED)
        reconstruction = np.linalg.norm(X - primary.transform(X) @ primary.components_, ord="fro")
        relative_error = float(reconstruction / x_norm)

        stabilities = []
        for seed in STABILITY_SEEDS:
            other = _fit_nmf(X, k, seed)
            stabilities.append(match_topic_stability(primary.components_, other.components_))

        results.append(KKSweepResult(k=k, reconstruction_error=relative_error, stability=float(np.mean(stabilities))))
    return results


def best_k(sweep: list[KKSweepResult]) -> KKSweepResult:
    """Maximize stability; ties broken toward the larger (finer-grained) k."""
    return max(sweep, key=lambda r: (round(r.stability, 4), r.k))


@dataclass
class NMFBuildStats:
    n_decks_total: int
    n_decks_sampled: int
    n_cards_total: int
    n_lands_excluded: int
    n_cards_in_pool: int
    n_cards_dropped_no_signal: int
    k_grid: list[int]
    sweep: list[KKSweepResult]
    chosen_k: int
    chosen_stability: float
    chosen_reconstruction_error: float


def build_and_save(
    conn: sqlite3.Connection, *, out_dir: Path = KB_DEV_DIR, k_grid: list[int] | None = None
) -> NMFBuildStats:
    """Build NMF synergy packages, sweep k, pick the most stable, and write
    `nmf_card_memberships.parquet` (`oracle_id`, `name`, `topic_id`, `weight`, `share` -- one row
    per card per topic it meaningfully belongs to, so a card can appear multiple times) and
    `nmf_deck_proportions.parquet` (`deck_id`, `topic_id`, `proportion` -- training-subsample
    decks only; use `topic_proportions_for_deck` for an arbitrary deck)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    card_index = load_card_index(out_dir)
    exclude = card_index["is_land"].to_numpy().astype(bool)
    pool = card_index.loc[~exclude].reset_index(drop=True)

    n_decks_total = conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0]
    deck_ids = _sample_deck_ids(conn)

    matrix = build_deck_card_matrix(conn, card_index, deck_ids)
    # card_index["row"] is positional (0..n-1 in card_index's own row order, see
    # cooccurrence.build_card_index), so a boolean mask over card_index aligns directly with a
    # boolean mask over the matrix's columns.
    matrix = matrix[:, ~exclude]  # drop basic-land columns

    has_signal = np.asarray((matrix.sum(axis=0) > 0)).ravel()
    n_dropped = int((~has_signal).sum())
    matrix = matrix[:, has_signal]
    pool = pool.loc[has_signal].reset_index(drop=True)

    resolved_k_grid = k_grid if k_grid is not None else K_GRID
    sweep = sweep_k(matrix, k_grid=resolved_k_grid)
    chosen = best_k(sweep)

    model = _fit_nmf(matrix, chosen.k, PRIMARY_SEED)
    w = model.transform(matrix)  # decks x k
    h = model.components_  # k x cards (pool-scoped)

    memberships = memberships_table(pool, h)
    memberships.to_parquet(out_dir / "nmf_card_memberships.parquet", index=False)

    deck_topic_id, deck_proportion, deck_id_out = [], [], []
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    w_norm = w / row_sums
    for row, deck_id in enumerate(deck_ids):
        for topic_id in range(chosen.k):
            proportion = float(w_norm[row, topic_id])
            if proportion >= MIN_MEMBERSHIP_SHARE:
                deck_id_out.append(deck_id)
                deck_topic_id.append(topic_id)
                deck_proportion.append(proportion)
    deck_proportions = pd.DataFrame(
        {"deck_id": deck_id_out, "topic_id": deck_topic_id, "proportion": deck_proportion}
    )
    deck_proportions.to_parquet(out_dir / "nmf_deck_proportions.parquet", index=False)

    np.save(out_dir / "nmf_components.npy", h)
    pool[["oracle_id", "row"]].rename(columns={"row": "original_row"}).to_parquet(
        out_dir / "nmf_pool_index.parquet", index=False
    )

    stats = NMFBuildStats(
        n_decks_total=n_decks_total,
        n_decks_sampled=len(deck_ids),
        n_cards_total=len(card_index),
        n_lands_excluded=int(exclude.sum()),
        n_cards_in_pool=len(pool),
        n_cards_dropped_no_signal=n_dropped,
        k_grid=resolved_k_grid,
        sweep=sweep,
        chosen_k=chosen.k,
        chosen_stability=chosen.stability,
        chosen_reconstruction_error=chosen.reconstruction_error,
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "max_decks_per_slot": MAX_DECKS_PER_SLOT,
        "subsample_seed": SUBSAMPLE_SEED,
        "min_membership_share": MIN_MEMBERSHIP_SHARE,
        "n_decks_total": stats.n_decks_total,
        "n_decks_sampled": stats.n_decks_sampled,
        "n_cards_total": stats.n_cards_total,
        "n_lands_excluded": stats.n_lands_excluded,
        "n_cards_in_pool": stats.n_cards_in_pool,
        "n_cards_dropped_no_signal": stats.n_cards_dropped_no_signal,
        "k_grid": stats.k_grid,
        "chosen_k": stats.chosen_k,
        "chosen_stability": stats.chosen_stability,
        "chosen_reconstruction_error": stats.chosen_reconstruction_error,
        "sweep": [vars(r) for r in sweep],
    }
    (out_dir / "nmf_manifest.json").write_text(json.dumps(manifest, indent=2))

    return stats


def memberships_table(pool: pd.DataFrame, h: np.ndarray) -> pd.DataFrame:
    """Long-format `oracle_id`/`name`/`topic_id`/`weight`/`share`: for each card (a column of
    `h`, shape `(k, n_cards)` -- topics x cards, matching sklearn's own `NMF.components_`
    convention), keep topics whose share of that card's total H-mass is `>=
    MIN_MEMBERSHIP_SHARE`, capped at the top `MAX_TOPICS_PER_CARD`. A card with zero total
    weight across every topic (shouldn't happen post-`has_signal` filtering, but guarded) is
    skipped entirely.

    Public (not `_`-prefixed) because `edhcut.analysis.symnmf_packages` reuses it verbatim --
    it's a pure function of a card pool and a nonnegative topic/card weight matrix, nothing
    NMF-specific about it. `symnmf_packages`' own `H` is cards x k (the natural shape for
    `S ≈ HHᵀ`), the transpose of this function's convention -- pass `H.T`."""
    oracle_ids, names, topic_ids, weights, shares = [], [], [], [], []
    col_totals = h.sum(axis=0)
    for col in range(h.shape[1]):
        total = col_totals[col]
        if total <= 0:
            continue
        card_weights = h[:, col]
        order = np.argsort(-card_weights)
        kept = 0
        for topic_id in order:
            if kept >= MAX_TOPICS_PER_CARD:
                break
            share = float(card_weights[topic_id] / total)
            if share < MIN_MEMBERSHIP_SHARE:
                break
            oracle_ids.append(pool.at[col, "oracle_id"])
            names.append(pool.at[col, "name"])
            topic_ids.append(int(topic_id))
            weights.append(float(card_weights[topic_id]))
            shares.append(share)
            kept += 1
    return pd.DataFrame(
        {"oracle_id": oracle_ids, "name": names, "topic_id": topic_ids, "weight": weights, "share": shares}
    )


def load_card_memberships(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "nmf_card_memberships.parquet")


def load_deck_proportions(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "nmf_deck_proportions.parquet")


def load_components(out_dir: Path = KB_DEV_DIR) -> np.ndarray:
    return np.load(out_dir / "nmf_components.npy")


def load_pool_index(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "nmf_pool_index.parquet")


def topic_members(
    memberships: pd.DataFrame, topic_id: int, *, k: int | None = None, min_share: float = 0.0
) -> pd.DataFrame:
    """Members of `topic_id`, ranked by weight (descending). `min_share` filters out cards for
    which this topic is only a minor component of their overall membership."""
    members = memberships[(memberships["topic_id"] == topic_id) & (memberships["share"] >= min_share)]
    members = members.sort_values("weight", ascending=False)
    return members.head(k) if k is not None else members


def card_topics(memberships: pd.DataFrame, oracle_id: str) -> pd.DataFrame:
    """This card's own topic memberships, sorted by share (descending) -- empty if the card was
    excluded (basic land) or dropped for lack of signal in the subsampled training matrix."""
    return memberships[memberships["oracle_id"] == oracle_id].sort_values("share", ascending=False)


def topic_proportions_for_deck(
    h: np.ndarray, pool: pd.DataFrame, oracle_ids: set[str]
) -> dict[int, float]:
    """Project an *arbitrary* deck's card set into topic space (not limited to the training
    subsample -- generalizes the same way `communities.cluster_of` generalizes to unseen
    commanders): a one-row NMF transform with `H` fixed (`update_H=False`), then row-normalized
    to proportions. Cards outside the NMF pool (excluded basics, or never in `card_index` at
    all) are silently ignored, same "unresolved -> no contribution" convention used elsewhere."""
    row_to_col = {oracle_id: col for col, oracle_id in enumerate(pool["oracle_id"])}
    cols = [row_to_col[oid] for oid in oracle_ids if oid in row_to_col]
    x = np.zeros((1, h.shape[1]), dtype=np.float64)
    if cols:
        x[0, cols] = 1.0

    w, _, _ = non_negative_factorization(
        x, H=h, n_components=h.shape[0], init="custom", update_H=False, solver="cd", random_state=PRIMARY_SEED,
    )
    total = w.sum()
    if total <= 0:
        return {}
    return {topic_id: float(w[0, topic_id] / total) for topic_id in range(h.shape[0]) if w[0, topic_id] > 0}


def _cmd_build(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        stats = build_and_save(conn)
    print(
        f"NMF packages: chosen k={stats.chosen_k} (stability={stats.chosen_stability:.3f}, "
        f"reconstruction_error={stats.chosen_reconstruction_error:.3f})"
    )
    print(f"  decks: {stats.n_decks_sampled:,}/{stats.n_decks_total:,} sampled")
    print(
        f"  cards: {stats.n_cards_total:,} total, {stats.n_lands_excluded} lands excluded, "
        f"{stats.n_cards_dropped_no_signal} dropped (no signal in sample), "
        f"{stats.n_cards_in_pool:,} in pool"
    )
    print("  k sweep:")
    for r in stats.sweep:
        marker = " <-- chosen" if r.k == stats.chosen_k else ""
        print(f"    k={r.k:3d}  stability={r.stability:.3f}  reconstruction_error={r.reconstruction_error:.3f}{marker}")


def _cmd_top(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        oracle_id = _resolve_card_name(conn, args.card)
    memberships = load_card_memberships()
    topics = card_topics(memberships, oracle_id)
    if topics.empty:
        print(f"{args.card!r} has no NMF topic membership (excluded land, or dropped for lack of signal).")
        return
    print(f"{args.card!r}'s topic membership(s):")
    for _, row in topics.iterrows():
        print(f"  topic {int(row['topic_id']):3d}  share={row['share']:.1%}  weight={row['weight']:.3f}")
    for _, row in topics.iterrows():
        topic_id = int(row["topic_id"])
        print(f"\nTopic {topic_id} top members:")
        for _, member in topic_members(memberships, topic_id, k=args.k).iterrows():
            print(f"  {member['share']:6.1%}  {member['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build & save NMF synergy packages to data/kb/dev/")
    build_p.set_defaults(func=_cmd_build)

    top_p = sub.add_parser("top", help="Show a card's topic membership(s) and each topic's top members")
    top_p.add_argument("card", help="Card name")
    top_p.add_argument("-k", type=int, default=10)
    top_p.set_defaults(func=_cmd_top)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
