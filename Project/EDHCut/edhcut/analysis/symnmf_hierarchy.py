"""Hierarchical decomposition of `symnmf_packages.py`'s color-corrected SymNMF packages (plan
`docs/plans/archive/6.3b-communities-next-iteration.md` step 3, task 6.3-B) -- the answer to that plan's
Q2 ("more communities, formalized"): don't chase granularity with a wider flat `k`, recurse
instead. Fit a small top-level split, then for each resulting topic induce the color-conditioned
`S` on just its own member cards and split again, recursively -- many small, cheap SymNMF fits
instead of one very large one.

**Why recursion, not a wider flat sweep**: a flat build's `k` topics all compete for the *same*
global signal at once, so beyond a certain point adding components mostly re-slices already-
explained structure rather than resolving genuinely narrower themes (the 6.3 devlog's own
finding: both Leiden's resolution sweep and flat NMF's k-sweep independently topped out around
40 on this corpus). Recursion sidesteps this: once a top-level topic already isolates (say)
"green ramp" from everything else, a *local* refit within just that reduced card set only has to
resolve ramp's own internal structure (Genesis Wave-style big-ramp vs. Cultivate-style
mana-fixing vs. ...), a genuinely easier, smaller problem -- and cheap, since `symnmf_packages`'s
solver runs in seconds even at full pool size (measured live, see that module's own docstring
and the devlog).

**A card can be dropped from a branch, same "unresolved -> no contribution" convention used
elsewhere**: a card whose local membership share falls below `MIN_MEMBERSHIP_SHARE` in *every*
one of its parent's child topics doesn't carry into any child node -- it stays represented at
the parent level (or wherever it last cleared threshold) but disappears from any cut that
expands past that point. Not a bug; the same silent-drop convention `nmf_packages.py`'s own
`MIN_MEMBERSHIP_SHARE` thresholding already uses within a single flat fit.

**A tree, not a strict partition**: because `memberships_table` (reused verbatim per node's own
local fit -- see `symnmf_packages.py`'s own docstring for why it's public) allows a card to clear
threshold in more than one child topic, a card can recurse down more than one branch. This is
deliberate, not an oversight -- the whole reason this project moved off hard clustering in the
first place (Ashnod's Altar: sac outlet *and* combo piece) applies exactly as much to the
hierarchy as to a flat cut.

**`cut_at` turns "more communities" into a dial, not a rebuild**: pick any node-count and get a
membership table in the *same* schema `symnmf_packages.py`'s flat build produces (`oracle_id`,
`name`, `topic_id`, `weight`, `share`), built by greedily expanding whichever frontier node
currently holds the most cards -- a simple, deterministic way to grow the cut from coarse (few
big nodes) to fine (many small ones) without re-fitting anything; the whole tree is fit once."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from edhcut.analysis.cooccurrence import KB_DEV_DIR, load_card_index
from edhcut.analysis.nmf_packages import match_topic_stability, memberships_table
from edhcut.analysis.symnmf_packages import (
    MAX_TOPICS_PER_CARD,
    MIN_MEMBERSHIP_SHARE,
    PRIMARY_SEED,
    fit_symnmf,
    load_pool_index,
    load_S,
)
from edhcut.config import CONFIG
from edhcut.db import connect

TOP_K_HIERARCHY = 10
CHILD_K = 6
MIN_NODE_CARDS = 60
MIN_CHILD_STABILITY = 0.7
MAX_DEPTH = 4
STABILITY_SEEDS = [1, 2]

# The plan's own originally-stated sweep ceiling (~400 leaves) turned out to be conservative --
# measured live against the real build, resolved-label count kept climbing past it (peaking
# ~1200) before *declining* at extreme fragmentation (2020 leaves, near-every-node-a-leaf), and
# median labels-per-topic kept falling monotonically throughout. Default grid covers both
# regimes so that peak is visible rather than assumed to be at the old ceiling -- see the devlog.
DEFAULT_LEAF_COUNTS = [20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]


@dataclass
class HierarchyNode:
    """One node of the recursive package tree. The root is `node_id` 0 with `parent_id` None; every
    other node is one topic of its parent's own symNMF fit, so `members` is local to that node
    (positions into the full pool, with the weight/share it earned *within* the parent split).
    `is_leaf` means the recursion stopped here — too few cards, or the split was not stable
    enough to keep."""

    node_id: int
    parent_id: int | None
    depth: int
    n_cards: int
    stability: float | None  # None for the root, and for any node created but never itself split
    is_leaf: bool
    members: pd.DataFrame = field(repr=False)  # oracle_id, name, weight, share -- local to this node


def _fit_children(
    S: sparse.csr_matrix,
    pool: pd.DataFrame,
    indices: np.ndarray,
    k: int,
    *,
    seed: int,
    stability_seeds: list[int],
) -> tuple[pd.DataFrame | None, float]:
    """Induce `S` on `indices` (positions into the full pool's row numbering), fit SymNMF at
    `k`, check seed stability, and return `(children_membership_df, stability)` --
    `children_membership_df` is `None` if the split isn't reliable enough to keep (see
    `MIN_CHILD_STABILITY`) or degenerates to no memberships at all."""
    sub_S = S[indices][:, indices]
    primary = fit_symnmf(sub_S, k, seed=seed)
    stabilities = []
    for other_seed in stability_seeds:
        other = fit_symnmf(sub_S, k, seed=other_seed)
        stabilities.append(match_topic_stability(primary.H.T, other.H.T))
    stability = float(np.mean(stabilities))

    sub_pool = pool.iloc[indices].reset_index(drop=True)
    children = memberships_table(sub_pool, primary.H.T)
    if children.empty:
        return None, stability
    return children, stability


def build_hierarchy(
    S: sparse.csr_matrix,
    pool: pd.DataFrame,
    *,
    top_k: int = TOP_K_HIERARCHY,
    child_k: int = CHILD_K,
    min_node_cards: int = MIN_NODE_CARDS,
    min_child_stability: float = MIN_CHILD_STABILITY,
    max_depth: int = MAX_DEPTH,
    seed: int = PRIMARY_SEED,
    stability_seeds: list[int] = STABILITY_SEEDS,
) -> dict[int, HierarchyNode]:
    """Recursively split `S`'s full pool (see module docstring). Returns every node built,
    keyed by `node_id` (root is the unique node with `parent_id is None`). A node stops
    recursing (becomes a leaf) when its own card count drops below `min_node_cards`, `max_depth`
    is reached, or its own children's seed stability falls below `min_child_stability` -- the
    unreliable split is discarded and the node stays a leaf rather than keeping a split nobody
    could reproduce across seeds."""
    nodes: dict[int, HierarchyNode] = {}
    next_id = [0]

    def new_id() -> int:
        next_id[0] += 1
        return next_id[0] - 1

    root_id = new_id()
    nodes[root_id] = HierarchyNode(
        node_id=root_id,
        parent_id=None,
        depth=0,
        n_cards=len(pool),
        stability=None,
        is_leaf=True,
        members=pd.DataFrame(
            {"oracle_id": pool["oracle_id"], "name": pool["name"], "weight": 1.0, "share": 1.0}
        ),
    )

    def recurse(node_id: int, indices: np.ndarray, k: int, depth: int) -> None:
        """Try to split one node into `k` children, then recurse into each. Three ways to stop, all
        of which leave the node a leaf: too few cards or too deep (the guard below), a `k` the node
        is too small to support (`this_k` scales with node size so a 30-card node is not asked for
        20 topics), or a split whose across-seed stability misses `min_child_stability` — the
        important one, since it is what keeps the tree from inventing structure that is not there.
        `indices` are positions in the *full* pool throughout, so a child's members stay
        addressable against the original matrix."""
        if len(indices) < min_node_cards or depth >= max_depth:
            return
        this_k = min(k, max(2, len(indices) // 10))
        if this_k < 2:
            return

        children, stability = _fit_children(S, pool, indices, this_k, seed=seed, stability_seeds=stability_seeds)
        if children is None or stability < min_child_stability:
            return  # keep node_id as a leaf -- no reliable split found

        nodes[node_id].is_leaf = False
        local_pos = {oid: i for i, oid in enumerate(pool.iloc[indices]["oracle_id"])}

        for local_topic_id, group in children.groupby("topic_id"):
            child_id = new_id()
            child_indices = indices[[local_pos[oid] for oid in group["oracle_id"]]]
            nodes[child_id] = HierarchyNode(
                node_id=child_id,
                parent_id=node_id,
                depth=depth + 1,
                n_cards=len(child_indices),
                stability=stability,
                is_leaf=True,
                members=group[["oracle_id", "name", "weight", "share"]].reset_index(drop=True),
            )
            recurse(child_id, child_indices, child_k, depth + 1)

    root_indices = np.arange(len(pool))
    recurse(root_id, root_indices, top_k, 0)
    return nodes


def _children_by_parent(nodes: dict[int, HierarchyNode]) -> dict[int, list[HierarchyNode]]:
    by_parent: dict[int, list[HierarchyNode]] = {}
    for node in nodes.values():
        if node.parent_id is not None:
            by_parent.setdefault(node.parent_id, []).append(node)
    return by_parent


def frontier_at(nodes: dict[int, HierarchyNode], n_leaves: int) -> list[HierarchyNode]:
    """The tree cut closest to `n_leaves` nodes: start at the root, repeatedly expand whichever
    *expandable* (non-leaf) frontier node currently holds the most cards, until the frontier
    reaches `n_leaves` nodes or nothing is left to expand (the tree's own leaves are fewer than
    `n_leaves`, and this returns all of them)."""
    root = next(n for n in nodes.values() if n.parent_id is None)
    frontier = [root]
    by_parent = _children_by_parent(nodes)

    while len(frontier) < n_leaves:
        expandable = [n for n in frontier if not n.is_leaf]
        if not expandable:
            break
        biggest = max(expandable, key=lambda n: n.n_cards)
        frontier.remove(biggest)
        frontier.extend(by_parent[biggest.node_id])
    return frontier


def cut_at(
    nodes: dict[int, HierarchyNode],
    n_leaves: int,
    *,
    min_membership_share: float = MIN_MEMBERSHIP_SHARE,
    max_topics_per_card: int = MAX_TOPICS_PER_CARD,
) -> pd.DataFrame:
    """A membership table (`oracle_id`, `name`, `topic_id`, `weight`, `share`) for the tree cut
    closest to `n_leaves` nodes -- same schema as `symnmf_packages.py`'s flat build, so every
    downstream consumer (`theme_labels.py`'s enrichment/purity metrics in particular) works on a
    cut unmodified. `topic_id` is the node's own `node_id` (globally unique across the whole
    tree, not renumbered 0..n_leaves-1).

    `share` is each node's *own* value, computed once at tree-build time (by `memberships_table`,
    within that node's own local fit) -- **not** renormalized across the nodes in this cut.
    Summing `weight` across nodes and dividing (an earlier version of this function did exactly
    that) is not a meaningful operation: two different nodes' `weight` values come from two
    different local SymNMF fits over two different induced submatrices, with no shared scale --
    a card's `weight=2.99` in one branch's factorization and `weight=0.11` in an unrelated
    branch's aren't comparable, so summing them and calling the ratio a "share" silently
    manufactures a number, not a measurement. Found live: Cultivate genuinely clears its own
    `MIN_MEMBERSHIP_SHARE` threshold in 44 of the 1,200 nodes in one real cut (it recurses down
    many separate ramp/midrange branches, all legitimately), and cross-node renormalization
    diluted every one of those 44 real memberships below threshold simultaneously, dropping the
    card from the cut's output entirely -- not a rare edge case, a structural consequence of
    allowing overlap at all (see the module docstring). The fix: filter by each node's own
    already-meaningful `share` directly, then cap fan-out to the `max_topics_per_card` *nodes*
    with the highest such share -- the same "keep this card's strongest few memberships"
    intent `nmf_packages.memberships_table` has, applied across nodes instead of within one."""
    frontier = frontier_at(nodes, n_leaves)
    parts = []
    for node in frontier:
        part = node.members[["oracle_id", "name", "weight", "share"]].copy()
        part["topic_id"] = node.node_id
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)

    raw = raw[raw["share"] >= min_membership_share]
    raw = raw.sort_values(["oracle_id", "share"], ascending=[True, False])
    raw["_rank"] = raw.groupby("oracle_id").cumcount()
    raw = raw[raw["_rank"] < max_topics_per_card].drop(columns="_rank")
    return raw[["oracle_id", "name", "weight", "topic_id", "share"]].reset_index(drop=True)


def save_hierarchy(nodes: dict[int, HierarchyNode], out_dir: Path = KB_DEV_DIR) -> None:
    """`symnmf_hierarchy_members.parquet` (long: `node_id`, `oracle_id`, `name`, `weight`,
    `share` -- every node's own local membership, the whole tree, not just one cut) +
    `symnmf_hierarchy_tree.json` (`node_id -> {parent_id, depth, n_cards, stability, is_leaf}`,
    structure only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    tree = {}
    for node in nodes.values():
        part = node.members.copy()
        part["node_id"] = node.node_id
        parts.append(part)
        tree[str(node.node_id)] = {
            "parent_id": node.parent_id,
            "depth": node.depth,
            "n_cards": node.n_cards,
            "stability": node.stability,
            "is_leaf": node.is_leaf,
        }
    members = pd.concat(parts, ignore_index=True)[["node_id", "oracle_id", "name", "weight", "share"]]
    members.to_parquet(out_dir / "symnmf_hierarchy_members.parquet", index=False)
    (out_dir / "symnmf_hierarchy_tree.json").write_text(json.dumps(tree, indent=2))


def load_hierarchy(out_dir: Path = KB_DEV_DIR) -> dict[int, HierarchyNode]:
    """Rebuild `{node_id: HierarchyNode}` from the saved tree JSON plus the long members table."""
    members = pd.read_parquet(out_dir / "symnmf_hierarchy_members.parquet")
    tree = json.loads((out_dir / "symnmf_hierarchy_tree.json").read_text())

    nodes: dict[int, HierarchyNode] = {}
    for node_id_str, meta in tree.items():
        node_id = int(node_id_str)
        node_members = members.loc[members["node_id"] == node_id, ["oracle_id", "name", "weight", "share"]]
        nodes[node_id] = HierarchyNode(
            node_id=node_id,
            parent_id=meta["parent_id"],
            depth=meta["depth"],
            n_cards=meta["n_cards"],
            stability=meta["stability"],
            is_leaf=meta["is_leaf"],
            members=node_members.reset_index(drop=True),
        )
    return nodes


def build_and_save(conn: sqlite3.Connection, *, out_dir: Path = KB_DEV_DIR, **hierarchy_kwargs) -> dict:
    """Build the hierarchy over `symnmf_packages.py`'s already-saved, cached `S`/pool (run
    `python -m edhcut.analysis.symnmf_packages build` first) and save it. `**hierarchy_kwargs`
    forwards to `build_hierarchy` (e.g. `min_node_cards=`, `max_depth=`) -- everything else about
    the input (`S`, land exclusion, color-conditioning) was already decided when that flat build
    ran; this step only controls how far/how the tree recurses on top of it."""
    S = load_S(out_dir)
    pool_index = load_pool_index(out_dir)
    card_index = load_card_index(out_dir)
    name_by_oracle_id = dict(zip(card_index["oracle_id"], card_index["name"]))
    pool = pool_index.assign(name=pool_index["oracle_id"].map(name_by_oracle_id))[["oracle_id", "name"]]

    nodes = build_hierarchy(S, pool, **hierarchy_kwargs)
    save_hierarchy(nodes, out_dir=out_dir)

    n_leaves = sum(1 for n in nodes.values() if n.is_leaf)
    return {
        "n_nodes": len(nodes),
        "n_leaves": n_leaves,
        "n_internal": len(nodes) - n_leaves,
        "max_depth": max(n.depth for n in nodes.values()),
    }


def granularity_curve(
    nodes: dict[int, HierarchyNode],
    conn: sqlite3.Connection,
    *,
    leaf_counts: list[int] = DEFAULT_LEAF_COUNTS,
    n_null_draws: int = 500,
) -> pd.DataFrame:
    """The two curves the plan asks for, in one table: resolved-label count (tau=0.10) and
    median color purity, each vs. the *actual* leaf count realized at each requested cut (may be
    less than requested if the tree runs out of leaves). Reuses `theme_labels.py`'s
    `granularity_report`/`color_purity` at every cut -- the whole point of building the tree
    once is that no refitting happens here, just `cut_at` (cheap) followed by those two
    diagnostics (also cheap, seconds each even at the full pool size -- measured live)."""
    from edhcut.analysis.theme_labels import build_theme_labels, color_purity, granularity_report

    rows = []
    for n_leaves in leaf_counts:
        memberships = cut_at(nodes, n_leaves)
        actual_leaves = int(memberships["topic_id"].nunique())
        pool = set(memberships["oracle_id"])
        labels = build_theme_labels(conn, pool_oracle_ids=pool)
        report = granularity_report(memberships, labels)
        purity_df, _ = color_purity(memberships, conn, n_null_draws=n_null_draws)
        rows.append(
            {
                "requested_leaves": n_leaves,
                "actual_leaves": actual_leaves,
                "n_cards": len(pool),
                "vocab_size": report["vocab_size"],
                "n_labels_covered": report["n_labels_covered"],
                "n_labels_resolved_010": report["n_labels_resolved"]["0.10"],
                "median_labels_per_topic": report["median_labels_per_topic"],
                "median_purity": float(purity_df["purity"].median()),
            }
        )
    return pd.DataFrame(rows)


def _cmd_build(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        stats = build_and_save(conn)
    print(f"SymNMF hierarchy: {stats['n_nodes']} nodes ({stats['n_leaves']} leaves, "
          f"{stats['n_internal']} internal), max depth {stats['max_depth']}")


def _cmd_curve(args: argparse.Namespace) -> None:
    nodes = load_hierarchy()
    with connect(CONFIG.paths.db_path) as conn:
        curve = granularity_curve(nodes, conn)
    curve.to_parquet(KB_DEV_DIR / "symnmf_hierarchy_granularity_curve.parquet", index=False)
    print(curve.to_string(index=False))
    print(f"\nWrote {KB_DEV_DIR / 'symnmf_hierarchy_granularity_curve.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build & save the SymNMF hierarchy to data/kb/dev/")
    build_p.set_defaults(func=_cmd_build)

    curve_p = sub.add_parser("curve", help="Compute & save the resolved-label/purity vs. n_leaves curves")
    curve_p.set_defaults(func=_cmd_curve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
