"""Synergy communities via Leiden graph clustering over the global t-score matrix (plan
`EDHCut_PLAN.md` §7, task 6.3).

**Global-only, not per-slot** -- a deliberate deviation from the plan's literal "globally and
per-slot" wording, decided explicitly with the user: `cluster_of()` needs to work for *any*
deck, including a commander outside the 5-slot roster the rest of the tool is otherwise scoped
to (v1 rejects those decks entirely -- per-slot clusters wouldn't exist for them at all). Checked
live before deciding: per-slot t-score matrices are also mostly empty rows for the same
thin-corpus reasons task 6.6 already flags for archetypes (64-96% zero-degree cards depending on
slot, worst for Orysa at 95.9%), so the marginal value over a global fallback would have been
weak for the two thinnest slots anyway.

**t-score, not PMI** -- despite the plan's literal wording, per the deferred decision flagged in
devlog `6.1b`: PMI ties rare coincidental pairs at a ceiling value (noise, not synergy), which
would manufacture spurious tight little clusters around that noise instead of the "recognizable
package" the sanity check calls for. t-score's frequency-aware low-count damping avoids this
natively (see `cooccurrence.py`'s own module docstring for the underlying math).

**Only the 5 basic land types (+ Wastes, + snow variants) are excluded as graph nodes -- every
other land is a normal candidate node.** Checked live before deciding anything, in two rounds:
(1) Skirk Prospector's real t-score row showed Command Tower scoring strongly *negative*
against it (t-score correctly detects "anti-correlated with mono-red aggro" on its own, no
domain rule needed) but Mountain sitting right inside the top-15 goblin-package neighbors,
indistinguishable in score from genuine package members -- so "exclude all lands" would have
been an unjustified blanket rule given how mixed that picture was. (2) A real (un-excluded)
build confirmed the concrete failure mode: only 6-11 clusters total, the biggest containing
1,600+ cards -- Mountain/Forest/Plains had graph degree 1,000+ (vs. a median of 15 and a next-
highest of 660), because a basic's near-universal per-color inclusion rate means it co-occurs
at real, non-noise, above-chance volume with *every* popular archetype of that color, not just
one -- Mountain's own top-15 t-score partners were literally the goblin package, purely because
most red decks run enough Mountains, not because of any synergy. That's structurally different
from a genuine utility land (or a popular nonland staple like Sol Ring, whose own degree was
high but concentrated *within* one real package rather than bridging many) -- only the 5 basics
carry zero mechanical identity and are fungible within a color, which is what makes them a
narrowly-justified exclusion rather than a general "exclude popular cards" rule.

**Edge sparsification -- top-K, union-symmetrized**: a card's positive-only t-score row can have
1000+ nonzero entries, mostly weak tail noise (Skirk Prospector: 1,503 nonzero global t-score
edges). Every node keeps only its `TOP_K` strongest *positive* edges (negative/zero t-score
edges are dropped entirely -- Leiden's modularity optimization expects non-negative weights); an
edge survives into the final graph if *either* endpoint kept it ("union", not "mutual"/
intersection) -- keeps a genuine one-sided strong signal, e.g. a rare synergy piece whose #1
partner is a much more generic staple that itself has many other strong partners and so
wouldn't necessarily return the favor in its own top-K.

**Node universe**: cards with zero positive-degree after the above (no positive t-score edge
strong enough to survive on either side) are dropped from the graph entirely, not kept as
disconnected nodes -- an isolated node would otherwise become its own meaningless 1-card
"community," padding the cluster count with noise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from edhcut.analysis.cooccurrence import (
    KB_DEV_DIR,
    _resolve_card_name,
    build_cooccurrence,
    compute_tscore,
    load_card_index,
)
from edhcut.analysis.deck_weights import compute_near_uniform_weights
from edhcut.analysis.playrate import load_card_play_rates
from edhcut.config import CONFIG
from edhcut.db import connect

TOP_K = 15
MIN_TSCORE = 0.0  # keep only better-than-chance (positive) edges
RESOLUTION_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
PRIMARY_SEED = 42
STABILITY_SEEDS = [1, 2, 3, 4, 5]


def basic_land_mask(conn, card_index: pd.DataFrame) -> np.ndarray:
    """True for the 5 basic land types, Wastes, and their snow-covered variants (Scryfall's
    `type_line` carries the `Basic` supertype for exactly these 12 cards, checked live against
    the DB -- no false positives against nonbasic lands). See module docstring for why these
    specifically are excluded as graph nodes and no other land is."""
    oracle_ids = tuple(card_index["oracle_id"])
    placeholders = ",".join("?" * len(oracle_ids))
    rows = conn.execute(
        f"SELECT oracle_id FROM cards WHERE oracle_id IN ({placeholders}) AND type_line LIKE '%Basic%'",
        oracle_ids,
    ).fetchall()
    basic_ids = {r[0] for r in rows}
    return card_index["oracle_id"].isin(basic_ids).to_numpy()


def build_graph(
    tscore: sparse.csr_matrix,
    *,
    top_k: int = TOP_K,
    min_tscore: float = MIN_TSCORE,
    exclude: np.ndarray | None = None,
) -> tuple[ig.Graph, np.ndarray]:
    """Weighted undirected graph over `tscore`'s row universe: each card's `top_k` strongest
    positive-t-score edges, union-symmetrized. `exclude` (boolean, length `tscore.shape[0]`) zeroes
    out those rows/columns entirely first -- used for the basic-land exclusion (see module
    docstring); they can't be a node themselves, nor count toward any other card's top-k. Returns
    the graph plus a boolean mask (length `tscore.shape[0]`) of which rows ended up with at least
    one edge -- see module docstring for why zero-degree rows (excluded, or just never strongly
    positively connected to anything) are dropped rather than kept as isolated nodes. The graph's
    vertex `i` corresponds to original row `np.flatnonzero(mask)[i]`, also stored as the
    `orig_row` vertex attribute for convenience."""
    n = tscore.shape[0]
    mat = tscore.tocsr(copy=True)
    mat.data[mat.data <= min_tscore] = 0.0
    mat.eliminate_zeros()

    if exclude is not None and exclude.any():
        mat = mat.tolil()
        mat[exclude, :] = 0.0
        mat[:, exclude] = 0.0
        mat = mat.tocsr()
        mat.eliminate_zeros()

    keep = sparse.lil_matrix((n, n), dtype=bool)
    for i in range(n):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        if start == end:
            continue
        cols = mat.indices[start:end]
        vals = mat.data[start:end]
        if len(cols) > top_k:
            top_idx = np.argpartition(-vals, top_k - 1)[:top_k]
            cols = cols[top_idx]
        keep[i, cols] = True
    keep_csr = keep.tocsr()
    keep_union = (keep_csr + keep_csr.T).astype(bool)  # union: either endpoint's top-k

    weighted = mat.multiply(keep_union).tocsr()
    weighted.eliminate_zeros()

    degree = np.asarray((weighted != 0).sum(axis=1)).ravel()
    has_edge = degree > 0

    sub = weighted[has_edge][:, has_edge].tocoo()
    upper = sub.row < sub.col
    edges = list(zip(sub.row[upper].tolist(), sub.col[upper].tolist()))
    weights = sub.data[upper].tolist()

    graph = ig.Graph(n=int(has_edge.sum()), edges=edges)
    graph.es["weight"] = weights
    graph.vs["orig_row"] = np.flatnonzero(has_edge).tolist()
    return graph, has_edge


@dataclass
class ResolutionResult:
    resolution: float
    n_clusters: int
    modularity: float
    singleton_count: int
    median_size: float
    max_size: int


def sweep_resolutions(
    graph: ig.Graph, resolutions: list[float] = RESOLUTION_GRID, *, seed: int = PRIMARY_SEED
) -> list[ResolutionResult]:
    """Run Leiden once per resolution value (fixed seed, so differences reflect the resolution
    parameter alone). Modularity is the plan's own "default to best" criterion."""
    results = []
    for r in resolutions:
        partition = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=r, seed=seed
        )
        sizes = [len(c) for c in partition]
        results.append(
            ResolutionResult(
                resolution=r,
                n_clusters=len(partition),
                modularity=partition.modularity,
                singleton_count=sum(1 for s in sizes if s == 1),
                median_size=float(np.median(sizes)),
                max_size=max(sizes),
            )
        )
    return results


def best_resolution(results: list[ResolutionResult]) -> ResolutionResult:
    return max(results, key=lambda r: r.modularity)


def seed_stability(
    graph: ig.Graph,
    resolution: float,
    *,
    primary_seed: int = PRIMARY_SEED,
    other_seeds: list[int] = STABILITY_SEEDS,
) -> tuple[np.ndarray, float, float]:
    """Runs Leiden at a fixed resolution across several seeds and reports agreement with the
    primary-seed partition via Adjusted Rand Index (1.0 = identical partitions, ~0.0 = no
    better than random) -- ARI rather than a raw label match, since cluster *labels* are
    arbitrary/unordered across independent runs; ARI compares partition structure, not label
    identity. Returns (primary partition's membership array, mean ARI, min ARI)."""
    primary = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution,
        seed=primary_seed,
    )
    primary_labels = np.array(primary.membership)

    scores = []
    for seed in other_seeds:
        part = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution,
            seed=seed,
        )
        scores.append(adjusted_rand_score(primary_labels, np.array(part.membership)))

    return primary_labels, float(np.mean(scores)), float(np.min(scores))


@dataclass
class CommunityBuildStats:
    n_cards_total: int
    n_basic_lands_excluded: int
    n_cards_in_graph: int
    n_edges: int
    resolution_sweep: list[ResolutionResult]
    chosen_resolution: float
    chosen_modularity: float
    chosen_n_clusters: int
    stability_mean_ari: float
    stability_min_ari: float


def build_and_save(conn, *, out_dir: Path = KB_DEV_DIR) -> CommunityBuildStats:
    """Build the global synergy-community graph, sweep resolutions, pick the best-modularity
    one, check its seed stability, and write `clusters.parquet` (`oracle_id`, `name`, `is_land`,
    `cluster_id` -- `-1` for cards dropped from the graph: excluded basic lands, or any other
    card with no strong-enough positive t-score edge to anything).

    Builds its own near-uniform-weighted t-score matrix (`compute_near_uniform_weights`) rather
    than loading `cooccurrence.py`'s saved `tscore_global.npz` -- that file is unweighted, which
    left a handful of oversampled roster commanders (Krenko, Kyler, Yenna -- thousands of decks
    each vs. ~9 typical) dominating a few clusters even after the 5.7 metagame-broadening
    harvest, confirmed by directly re-measuring cluster/slot purity post-rebuild. `top_associated`
    / the PMI "cards similar to X" lookup deliberately keeps the unweighted global scope --
    only communities opts into this."""
    out_dir.mkdir(parents=True, exist_ok=True)

    card_index = load_card_index(out_dir)
    weights = compute_near_uniform_weights(conn)
    result = build_cooccurrence(conn, card_index, slot_key=None, deck_slot_weights=weights)
    tscore = compute_tscore(result)
    exclude = basic_land_mask(conn, card_index)

    graph, has_edge = build_graph(tscore, exclude=exclude)

    sweep = sweep_resolutions(graph)
    chosen = best_resolution(sweep)
    labels, mean_ari, min_ari = seed_stability(graph, chosen.resolution)

    cluster_id = np.full(len(card_index), -1, dtype=np.int64)
    cluster_id[has_edge] = labels

    clusters = card_index[["oracle_id", "name", "is_land"]].copy()
    clusters["cluster_id"] = cluster_id
    clusters.to_parquet(out_dir / "clusters.parquet", index=False)

    stats = CommunityBuildStats(
        n_cards_total=len(card_index),
        n_basic_lands_excluded=int(exclude.sum()),
        n_cards_in_graph=int(has_edge.sum()),
        n_edges=graph.ecount(),
        resolution_sweep=sweep,
        chosen_resolution=chosen.resolution,
        chosen_modularity=chosen.modularity,
        chosen_n_clusters=chosen.n_clusters,
        stability_mean_ari=mean_ari,
        stability_min_ari=min_ari,
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "scope": "global",
        "deck_weighting": "near_uniform",
        "n_weighted_slots": len(weights),
        "top_k": TOP_K,
        "min_tscore": MIN_TSCORE,
        "resolution_grid": RESOLUTION_GRID,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": STABILITY_SEEDS,
        "n_cards_total": stats.n_cards_total,
        "n_basic_lands_excluded": stats.n_basic_lands_excluded,
        "n_cards_in_graph": stats.n_cards_in_graph,
        "n_edges": stats.n_edges,
        "chosen_resolution": stats.chosen_resolution,
        "chosen_modularity": stats.chosen_modularity,
        "chosen_n_clusters": stats.chosen_n_clusters,
        "stability_mean_ari": stats.stability_mean_ari,
        "stability_min_ari": stats.stability_min_ari,
        "resolution_sweep": [vars(r) for r in sweep],
    }
    (out_dir / "communities_manifest.json").write_text(json.dumps(manifest, indent=2))

    return stats


def load_clusters(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "clusters.parquet")


def cluster_of(clusters: pd.DataFrame, oracle_id: str) -> int | None:
    """The cluster id for `oracle_id`, or `None` if it's not in the card pool at all, or was
    dropped from the graph (no strong-enough positive t-score edge to anything)."""
    match = clusters.loc[clusters["oracle_id"] == oracle_id, "cluster_id"]
    if len(match) == 0:
        return None
    value = int(match.iloc[0])
    return value if value >= 0 else None


def cluster_members(
    clusters: pd.DataFrame, play_rates: pd.DataFrame, cluster_id: int, *, k: int | None = None
) -> pd.DataFrame:
    """Members of `cluster_id`, ranked by play rate (with raw `deck_count` alongside -- see
    project memory's standing "keep raw counts with weighted" preference), descending. Play
    rate stands in for the plan's literal "EDHREC inclusion" ranking here since EDHREC data is
    scoped to the 5-slot roster's own commander pages, while this cluster's cards span the full
    commander-legal pool by design (global scope, generalizes to any commander) -- play rate
    already is this project's own commander-agnostic version of "how core is this card." Cards
    with no play-rate row (below `MIN_ELIGIBLE_DECKS` or otherwise unranked) sort last, not
    dropped, so `k=None` still returns every member."""
    members = clusters[clusters["cluster_id"] == cluster_id]
    merged = members.merge(
        play_rates[["oracle_id", "deck_count", "eligible_deck_count", "play_rate"]], on="oracle_id", how="left"
    )
    merged = merged.sort_values("play_rate", ascending=False, na_position="last")
    return merged.head(k) if k is not None else merged


def _cmd_build(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        stats = build_and_save(conn)

    print(f"Global synergy-community graph: {stats.n_cards_in_graph:,} / {stats.n_cards_total:,} cards, "
          f"{stats.n_edges:,} edges (top-{TOP_K} union-sparsified, positive t-score only, "
          f"{stats.n_basic_lands_excluded} basic lands excluded).")
    print()
    print("Resolution sweep:")
    print(f"  {'resolution':>10s}  {'clusters':>8s}  {'modularity':>10s}  {'singletons':>10s}  "
          f"{'median sz':>9s}  {'max sz':>6s}")
    for r in stats.resolution_sweep:
        marker = " <-- chosen (best modularity)" if r.resolution == stats.chosen_resolution else ""
        print(f"  {r.resolution:>10.2f}  {r.n_clusters:>8d}  {r.modularity:>10.4f}  {r.singleton_count:>10d}  "
              f"{r.median_size:>9.1f}  {r.max_size:>6d}{marker}")
    print()
    print(f"Seed stability at resolution {stats.chosen_resolution}: mean ARI={stats.stability_mean_ari:.3f}, "
          f"min ARI={stats.stability_min_ari:.3f} (across seeds {STABILITY_SEEDS} vs. primary seed {PRIMARY_SEED})")


def _cmd_report(args: argparse.Namespace) -> None:
    clusters = load_clusters()
    play_rates = load_card_play_rates()
    sizes = clusters[clusters["cluster_id"] >= 0]["cluster_id"].value_counts()
    sizes = sizes[sizes >= args.min_size].sort_values(ascending=False)

    print(f"{len(sizes)} clusters with >= {args.min_size} members (of {sizes.index.max() + 1 if len(sizes) else 0} "
          f"total):")
    for cluster_id in sizes.index[: args.limit]:
        members = cluster_members(clusters, play_rates, int(cluster_id), k=10)
        print(f"\nCluster {cluster_id} ({sizes[cluster_id]} members), top {len(members)} by play rate:")
        for _, row in members.iterrows():
            land_tag = " [land]" if row["is_land"] else ""
            play_rate = f"{row['play_rate']:.1%}" if pd.notna(row["play_rate"]) else "n/a"
            print(f"  {play_rate:>6s}  {row['name']}{land_tag}")


def _cmd_top(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        oracle_id = _resolve_card_name(conn, args.card)
    clusters = load_clusters()
    cid = cluster_of(clusters, oracle_id)
    if cid is None:
        print(f"{args.card!r} has no community (not in the card pool, or dropped -- no strong positive t-score "
              f"edge to anything).")
        return
    play_rates = load_card_play_rates()
    size = int((clusters["cluster_id"] == cid).sum())
    members = cluster_members(clusters, play_rates, cid, k=args.k)
    print(f"{args.card!r} is in cluster {cid} ({size} members), top {len(members)} by play rate:")
    for _, row in members.iterrows():
        land_tag = " [land]" if row["is_land"] else ""
        play_rate = f"{row['play_rate']:.1%}" if pd.notna(row["play_rate"]) else "n/a"
        print(f"  {play_rate:>6s}  {row['name']}{land_tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build & save clusters.parquet to data/kb/dev/")
    build_p.set_defaults(func=_cmd_build)

    report_p = sub.add_parser("report", help="Print every cluster's top-10 members by play rate")
    report_p.add_argument("--min-size", type=int, default=4, help="Skip clusters smaller than this")
    report_p.add_argument("--limit", type=int, default=30, help="Max number of clusters to print")
    report_p.set_defaults(func=_cmd_report)

    top_p = sub.add_parser("top", help="Print a card's cluster and its top members by play rate")
    top_p.add_argument("card", help="Card name")
    top_p.add_argument("-k", type=int, default=10)
    top_p.set_defaults(func=_cmd_top)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
