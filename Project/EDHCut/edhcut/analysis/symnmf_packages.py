"""Color-corrected soft synergy packages via symmetric NMF over the color-conditioned t-score
matrix (plan `docs/plans/6.3b-communities-next-iteration.md` step 2, task 6.3-B) -- the answer to
that plan's Q1 ("can NMF use the color-identity fix"): not by patching raw-presence NMF's input
(a card's color-identity legality is *structural* in the deck x card matrix -- a mono-green card
has literal zero support outside green-legal decks at any column scaling, and non-negativity
rules out subtracting an expectation -- see the plan's own "rejected patches" section), but by
factorizing a *different, already-corrected* matrix instead.

`edhcut.analysis.cooccurrence.compute_color_conditioned_tscore` already solves this exact
problem for `communities.py`'s side of the project (validated against 20 real card pairs,
`docs/devlog/6.1-cooccurrence.md`'s own addendum). Its output is a symmetric card x card matrix,
non-negative once thresholded at 0 -- exactly the shape symmetric NMF wants (`S ~ H H^T`, `H >=
0`). Every column of `H` is a soft package, every row a card's own soft membership vector -- the
same object `nmf_packages.py`'s factor `H` gave downstream consumers, so `card_topics`,
`topic_members`, and the membership-table schema (`nmf_packages.memberships_table`, made public
specifically so this module can reuse it rather than duplicate it) carry over unchanged.

**Two costs, relative to `nmf_packages.py`'s raw-presence NMF, both deliberate, not oversights**:
- **No `W` (deck -> topic) factor.** SymNMF factorizes a card x card matrix, which never sees
  individual decks, and (unlike `nmf_packages.py`) there's no training-deck subsample to save
  proportions for in the first place -- `S` is built from the *entire* weighted corpus via
  `build_cooccurrence`, not a per-deck sample. `topic_proportions_for_deck` here is a plain
  projection (row-normalized sum of a deck's own `H` rows), computed on demand for any deck; this
  module does not save a `symnmf_deck_proportions.parquet`.
- **Land exclusion reverts to basics-only** (`communities.basic_land_mask`), not all lands.
  `nmf_packages.py`'s all-lands exclusion fixed a raw-presence-NMF-specific failure mode (a
  nonbasic land's near-universal within-color-pair presence became an easy, high-reward signal
  for that factorization to spend components on -- see that module's own docstring). t-score
  doesn't share this failure mode (`communities.py`'s own docstring: a land's association with
  any *one* other card stays weak/undifferentiated in t-score, so it never concentrates on a
  single archetype the way raw co-presence does) -- re-verified live for *this* build, not
  assumed by analogy; see this module's own regression checks / the accompanying devlog.

**Solver: a custom multiplicative-update loop, not `sklearn`** (`sklearn.decomposition.NMF` has
no symmetric mode). Never forms `H @ H.T` (`n x n` dense, ~2.7 GiB at `n`~=18,000, float64) --
every quantity, including the Frobenius residual used for convergence and reporting, goes
through `k x k` intermediates:

    G   = H^T @ H                                  # k x k
    num = S @ H                                    # n x k -- sparse @ dense
    den = H @ G                                     # n x k -- dense @ dense, never H @ H^T
    H  *= (1 - BETA) + BETA * num / (den + EPS)      # damped multiplicative update

    ||S - HH^T||_F^2 = ||S||_F^2 - 2*sum(num * H) + ||G||_F^2

(`<S, HH^T> = trace(H^T S H) = sum((S@H)*H)` since `S` is symmetric; `||HH^T||_F^2 =
||H^T H||_F^2 = ||G||_F^2` is a standard identity for any real `H`.) `BETA=0.5` damps the naive
Lee-Seung ratio update (`BETA=1`) -- checked live across 18 (k, seed) combinations on the real
color-conditioned `S` (k in {15, 40, 80}, 3 seeds each) before picking a default rather than
assuming damping was needed: `BETA=1` was monotone in most runs but genuinely oscillated in two
of them (k=40 seed=1: 65/149 iterations with an *increasing* residual, up to +0.00042 in one
step; k=80 seed=7: 54/149 increasing), while `BETA=0.5` showed zero residual increases across
every single run tested. Not a universal problem with `BETA=1` -- most runs were fine either way
-- but real and seed-dependent, which is exactly the kind of failure a k-sweep across many seeds
(this module's own `sweep_k`) would otherwise mask until it silently picked an unstable run's
output. See the devlog for the full number set."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from edhcut.analysis.communities import basic_land_mask, sparsify_top_k_union
from edhcut.analysis.cooccurrence import (
    KB_DEV_DIR,
    _resolve_card_name,
    build_card_color_identities,
    build_cooccurrence,
    compute_color_conditioned_tscore,
    load_card_index,
)
from edhcut.analysis.deck_weights import compute_near_uniform_weights
from edhcut.analysis.nmf_packages import card_topics, match_topic_stability, memberships_table, topic_members
from edhcut.analysis.playrate import build_color_identity_deck_counts
from edhcut.config import CONFIG
from edhcut.db import connect

__all__ = ["card_topics", "topic_members"]  # re-exported: same schema, no SymNMF-specific logic

TOP_K = 50
MIN_TSCORE = 0.0
# Wider than nmf_packages.py's own K_GRID (which tops out at 40) by design, not oversight --
# measured live (see the devlog) at ~3-6s/fit for k<=40 and ~33-143s/fit at k=80-300, orders of
# magnitude cheaper than raw-presence NMF's ~k^2.25 cost on the much bigger deck x card matrix
# (that module's own k-sweep never got to see whether stability truly peaked at its k=40 grid
# ceiling or was still rising -- this one can afford to actually look).
K_GRID = [15, 20, 25, 30, 40, 60, 80]
PRIMARY_SEED = 42
STABILITY_SEEDS = [1, 2, 3]

BETA = 0.5
MAX_ITER = 300
TOL = 1e-5
EPS = 1e-9

MIN_MEMBERSHIP_SHARE = 0.10
MAX_TOPICS_PER_CARD = 5


def build_color_conditioned_S(
    conn: sqlite3.Connection, card_index: pd.DataFrame
) -> tuple[sparse.csr_matrix, dict]:
    """The near-uniform-weighted, color-conditioned, non-negative card x card matrix this
    module factorizes -- `max(t_color, 0)`, `t_color` from
    `compute_color_conditioned_tscore` over a `compute_near_uniform_weights`-weighted
    `build_cooccurrence` result (mirrors `communities.py`'s own weighting choice: diversity of
    discovered packages over metagame-accurate representation -- see that module's docstring).
    `color_identity_deck_counts` uses the *same* weights (the `deck_slot_weights` fix from plan
    step 0 -- without it the null model's units mismatch the weighted `joint` counts and the
    color correction is systematically under-applied). Returns `(S, stats)`, `stats` holding the
    raw deck/weight totals for the manifest."""
    weights = compute_near_uniform_weights(conn)
    result = build_cooccurrence(conn, card_index, slot_key=None, deck_slot_weights=weights)
    card_identities = build_card_color_identities(conn, card_index)
    color_identity_deck_counts = build_color_identity_deck_counts(conn, deck_slot_weights=weights)
    tscore_color = compute_color_conditioned_tscore(result, card_identities, color_identity_deck_counts)

    mat = tscore_color.tocsr(copy=True)
    mat.data[mat.data < 0] = 0.0
    mat.eliminate_zeros()

    stats = {
        "n_decks_total": result.deck_count,
        "total_weight": result.total_weight,
        "n_weighted_slots": len(weights),
    }
    return mat, stats


@dataclass
class SymNMFFitResult:
    H: np.ndarray
    n_iter: int
    relative_residual: float
    residual_history: list[float]
    fit_seconds: float


def _init_H(n: int, k: int, s_mean: float, seed: int) -> np.ndarray:
    """`H`'s initial scale is set so `E[(HH^T)_ii'] ~= s_mean` (matching `S`'s own average
    entry) -- the same scale-consistency reasoning behind `sklearn`'s own `init="random"`."""
    rng = np.random.default_rng(seed)
    scale = 2.0 * math.sqrt(max(s_mean, 1e-12) / k)
    return scale * rng.random((n, k))


def _relative_residual_from(G: np.ndarray, num: np.ndarray, H: np.ndarray, s_frob_sq: float) -> float:
    if s_frob_sq <= 0:
        return 0.0
    residual_sq = max(s_frob_sq - 2.0 * float((num * H).sum()) + float((G * G).sum()), 0.0)
    return math.sqrt(residual_sq) / math.sqrt(s_frob_sq)


def _relative_residual(S: sparse.csr_matrix, H: np.ndarray, s_frob_sq: float) -> float:
    """Convenience one-off version of `_relative_residual_from` for callers (tests, notebook
    diagnostics) that don't already have `G`/`num` lying around -- `fit_symnmf`'s own loop
    computes them anyway for the update itself, so it calls `_relative_residual_from` directly
    rather than recomputing `S @ H` a second time per iteration."""
    G = H.T @ H
    num = S @ H
    return _relative_residual_from(G, num, H, s_frob_sq)


def fit_symnmf(
    S: sparse.csr_matrix,
    k: int,
    *,
    seed: int = PRIMARY_SEED,
    beta: float = BETA,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
    eps: float = EPS,
) -> SymNMFFitResult:
    """Fit `H` (`n x k`, `n = S.shape[0]`) minimizing `||S - HH^T||_F^2` via the damped
    multiplicative update described in the module docstring. Every iteration computes `G =
    H^T@H` and `num = S@H` exactly once, reused for both the update and that iteration's
    residual -- so `relative_residual`/`residual_history`'s last entry describes `H` as it stood
    *before* that final iteration's update. At a normal convergence break (consecutive residuals
    within `tol`) no update was applied in that final pass, so this is exact, not stale; it's
    only a very slight (one-update-stale) overstatement if `max_iter` is hit without converging
    -- acceptable for a k-sweep comparison, not exact optimization bookkeeping. Stops when
    consecutive iterations' relative residual changes by less than `tol`, or at `max_iter`."""
    start = time.perf_counter()
    n = S.shape[0]
    s_frob_sq = float((S.data ** 2).sum())
    s_mean = float(S.sum()) / (n * n) if n > 0 else 0.0
    H = _init_H(n, k, s_mean, seed)

    history: list[float] = []
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        G = H.T @ H
        num = S @ H
        rel = _relative_residual_from(G, num, H, s_frob_sq)
        history.append(rel)
        if len(history) >= 2 and abs(history[-2] - history[-1]) < tol:
            break
        den = H @ G
        H *= (1 - beta) + beta * num / (den + eps)

    return SymNMFFitResult(
        H=H,
        n_iter=n_iter,
        relative_residual=history[-1],
        residual_history=history,
        fit_seconds=time.perf_counter() - start,
    )


@dataclass
class KSweepResult:
    k: int
    relative_residual: float
    stability: float
    fit_seconds: float


def sweep_k(
    S: sparse.csr_matrix, *, k_grid: list[int] = K_GRID, seed: int = PRIMARY_SEED,
    stability_seeds: list[int] = STABILITY_SEEDS,
) -> list[KSweepResult]:
    """Fit at every `k` in `k_grid` (primary seed for reconstruction error, `stability_seeds`
    for `match_topic_stability`, reused from `nmf_packages.py` -- see that function's own
    docstring for why it accepts `(k, n_cards)`-shaped inputs regardless of which module built
    them, hence the `.T` below)."""
    results = []
    for k in k_grid:
        primary = fit_symnmf(S, k, seed=seed)
        stabilities = []
        for other_seed in stability_seeds:
            other = fit_symnmf(S, k, seed=other_seed)
            stabilities.append(match_topic_stability(primary.H.T, other.H.T))
        results.append(
            KSweepResult(
                k=k,
                relative_residual=primary.relative_residual,
                stability=float(np.mean(stabilities)),
                fit_seconds=primary.fit_seconds,
            )
        )
    return results


def best_k(sweep: list[KSweepResult]) -> KSweepResult:
    """Maximize stability; ties broken toward the larger (finer-grained) k -- same convention as
    `nmf_packages.best_k`."""
    return max(sweep, key=lambda r: (round(r.stability, 4), r.k))


@dataclass
class SymNMFBuildStats:
    n_cards_total: int
    n_basic_lands_excluded: int
    n_cards_in_pool: int
    n_edges: int
    k_grid: list[int]
    sweep: list[KSweepResult]
    chosen_k: int
    chosen_stability: float
    chosen_relative_residual: float
    chosen_fit_seconds: float


def build_and_save(
    conn: sqlite3.Connection, *, out_dir: Path = KB_DEV_DIR, k_grid: list[int] | None = None
) -> SymNMFBuildStats:
    """Build the color-corrected SymNMF card pool, sweep `k`, pick the most stable, and write
    `symnmf_card_memberships.parquet` (same schema as `nmf_packages.py`'s: `oracle_id`, `name`,
    `topic_id`, `weight`, `share`), `symnmf_components.npy` (`H`, `n_pool x k`),
    `symnmf_pool_index.parquet` (`oracle_id`, `original_row` -- row `i` of `H` <-> this pool's
    row `i`, which maps back to `card_index`'s own `row` via this table), and `symnmf_S.npz`
    (the sparsified factorization input, cached so later steps -- the hierarchical decomposition
    in particular -- don't have to rebuild the color-conditioned t-score from scratch)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    card_index = load_card_index(out_dir)
    S_full, cooc_stats = build_color_conditioned_S(conn, card_index)
    exclude = basic_land_mask(conn, card_index)

    S, has_edge = sparsify_top_k_union(S_full, top_k=TOP_K, min_tscore=MIN_TSCORE, exclude=exclude)
    pool = card_index.loc[has_edge].reset_index(drop=True)

    resolved_k_grid = k_grid if k_grid is not None else K_GRID
    sweep = sweep_k(S, k_grid=resolved_k_grid)
    chosen = best_k(sweep)

    fit = fit_symnmf(S, chosen.k, seed=PRIMARY_SEED)
    H = fit.H

    memberships = memberships_table(pool, H.T)
    memberships.to_parquet(out_dir / "symnmf_card_memberships.parquet", index=False)

    np.save(out_dir / "symnmf_components.npy", H)
    pool[["oracle_id", "row"]].rename(columns={"row": "original_row"}).to_parquet(
        out_dir / "symnmf_pool_index.parquet", index=False
    )
    sparse.save_npz(out_dir / "symnmf_S.npz", S)

    stats = SymNMFBuildStats(
        n_cards_total=len(card_index),
        n_basic_lands_excluded=int(exclude.sum()),
        n_cards_in_pool=len(pool),
        n_edges=S.nnz // 2,
        k_grid=resolved_k_grid,
        sweep=sweep,
        chosen_k=chosen.k,
        chosen_stability=chosen.stability,
        chosen_relative_residual=chosen.relative_residual,
        chosen_fit_seconds=fit.fit_seconds,
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "deck_weighting": "near_uniform",
        "n_decks_total": cooc_stats["n_decks_total"],
        "total_weight": cooc_stats["total_weight"],
        "n_weighted_slots": cooc_stats["n_weighted_slots"],
        "top_k": TOP_K,
        "min_tscore": MIN_TSCORE,
        "beta": BETA,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "min_membership_share": MIN_MEMBERSHIP_SHARE,
        "n_cards_total": stats.n_cards_total,
        "n_basic_lands_excluded": stats.n_basic_lands_excluded,
        "n_cards_in_pool": stats.n_cards_in_pool,
        "n_edges": stats.n_edges,
        "k_grid": stats.k_grid,
        "chosen_k": stats.chosen_k,
        "chosen_stability": stats.chosen_stability,
        "chosen_relative_residual": stats.chosen_relative_residual,
        "chosen_fit_seconds": stats.chosen_fit_seconds,
        "sweep": [vars(r) for r in sweep],
    }
    (out_dir / "symnmf_manifest.json").write_text(json.dumps(manifest, indent=2))

    return stats


def load_card_memberships(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "symnmf_card_memberships.parquet")


def load_components(out_dir: Path = KB_DEV_DIR) -> np.ndarray:
    return np.load(out_dir / "symnmf_components.npy")


def load_pool_index(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "symnmf_pool_index.parquet")


def load_S(out_dir: Path = KB_DEV_DIR) -> sparse.csr_matrix:
    return sparse.load_npz(out_dir / "symnmf_S.npz")


def topic_proportions_for_deck(H: np.ndarray, pool: pd.DataFrame, oracle_ids: set[str]) -> dict[int, float]:
    """A deck's topic proportions: row-normalized `sum(H[c, :] for c in deck cards in pool)` --
    a plain projection, not a fitted quantity (see module docstring for why SymNMF has no `W`
    factor the way `nmf_packages.py` does). Cards outside the pool are silently ignored, same
    "unresolved -> no contribution" convention used elsewhere in this project."""
    row_by_oracle_id = {oracle_id: i for i, oracle_id in enumerate(pool["oracle_id"])}
    rows = [row_by_oracle_id[oid] for oid in oracle_ids if oid in row_by_oracle_id]
    if not rows:
        return {}
    total = H[rows, :].sum(axis=0)
    denom = total.sum()
    if denom <= 0:
        return {}
    return {topic_id: float(total[topic_id] / denom) for topic_id in range(H.shape[1]) if total[topic_id] > 0}


def _cmd_build(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        stats = build_and_save(conn)
    print(
        f"SymNMF packages: chosen k={stats.chosen_k} (stability={stats.chosen_stability:.3f}, "
        f"relative_residual={stats.chosen_relative_residual:.3f}, fit={stats.chosen_fit_seconds:.1f}s)"
    )
    print(
        f"  cards: {stats.n_cards_total:,} total, {stats.n_basic_lands_excluded} basic lands excluded, "
        f"{stats.n_cards_in_pool:,} in pool (top-{TOP_K} union-sparsified, zero-degree dropped)"
    )
    print("  k sweep:")
    for r in stats.sweep:
        marker = " <-- chosen" if r.k == stats.chosen_k else ""
        print(
            f"    k={r.k:3d}  stability={r.stability:.3f}  relative_residual={r.relative_residual:.3f}  "
            f"fit={r.fit_seconds:6.1f}s{marker}"
        )


def _cmd_top(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        oracle_id = _resolve_card_name(conn, args.card)
    memberships = load_card_memberships()
    topics = card_topics(memberships, oracle_id)
    if topics.empty:
        print(f"{args.card!r} has no SymNMF topic membership (excluded basic land, or dropped for lack "
              f"of a strong-enough positive color-conditioned t-score edge).")
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

    build_p = sub.add_parser("build", help="Build & save SymNMF synergy packages to data/kb/dev/")
    build_p.set_defaults(func=_cmd_build)

    top_p = sub.add_parser("top", help="Show a card's topic membership(s) and each topic's top members")
    top_p.add_argument("card", help="Card name")
    top_p.add_argument("-k", type=int, default=10)
    top_p.set_defaults(func=_cmd_top)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
