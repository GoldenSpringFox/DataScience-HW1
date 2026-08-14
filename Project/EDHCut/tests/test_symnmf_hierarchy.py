import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from edhcut.analysis.symnmf_hierarchy import (
    HierarchyNode,
    build_hierarchy,
    cut_at,
    frontier_at,
    granularity_curve,
    load_hierarchy,
    save_hierarchy,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _hier_block_S(leaf_size: int = 10) -> sparse.csr_matrix:
    """4 leaf blocks (0,1,2,3) of `leaf_size` cards each -- 0&1 form top-level group A, 2&3 form
    top-level group B. Strong within-leaf-block signal (8.0), weaker but real within-top-group
    cross-leaf-block signal (3.0), zero signal across top-level groups -- a genuine 2-level
    hierarchy, not just 4 flat blocks, so a top-level split should find {A, B} before a second
    level of recursion finds the 4 individual leaf blocks."""
    n = leaf_size * 4
    dense = np.zeros((n, n))

    def block(i, j, value):
        dense[i * leaf_size : (i + 1) * leaf_size, j * leaf_size : (j + 1) * leaf_size] = value

    for i in range(4):
        block(i, i, 8.0)
    block(0, 1, 3.0)
    block(1, 0, 3.0)
    block(2, 3, 3.0)
    block(3, 2, 3.0)
    np.fill_diagonal(dense, 0.0)
    return sparse.csr_matrix(dense)


def _hier_pool(leaf_size: int = 10) -> pd.DataFrame:
    n = leaf_size * 4
    return pd.DataFrame({"oracle_id": [f"c{i}" for i in range(n)], "name": [f"Card {i}" for i in range(n)]})


# --- build_hierarchy --------------------------------------------------------------------------

def test_build_hierarchy_root_exists_and_recurses() -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )

    roots = [n for n in nodes.values() if n.parent_id is None]
    assert len(roots) == 1
    assert len(nodes) > 1  # some recursion happened
    assert all(n.depth <= 3 for n in nodes.values())


def test_build_hierarchy_stops_below_min_node_cards() -> None:
    S = _hier_block_S(leaf_size=3)  # only 12 cards total
    pool = _hier_pool(leaf_size=3)
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=100, min_child_stability=0.0, max_depth=5, seed=0
    )
    # min_node_cards way above the pool size -- root can never recurse
    assert len(nodes) == 1


def test_build_hierarchy_respects_max_depth() -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=2, min_child_stability=0.0, max_depth=1, seed=0
    )
    assert max(n.depth for n in nodes.values()) <= 1


# --- frontier_at -------------------------------------------------------------------------------

def test_frontier_at_one_returns_just_the_root() -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )
    frontier = frontier_at(nodes, 1)
    assert len(frontier) == 1
    assert frontier[0].parent_id is None


def test_frontier_at_large_n_returns_only_leaves() -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )
    frontier = frontier_at(nodes, 1000)
    assert all(n.is_leaf for n in frontier)
    total_leaves = sum(1 for n in nodes.values() if n.is_leaf)
    assert len(frontier) == total_leaves


def test_frontier_at_grows_monotonically() -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )
    sizes = [len(frontier_at(nodes, n)) for n in (1, 2, 4, 8, 100)]
    assert sizes == sorted(sizes)


# --- cut_at --------------------------------------------------------------------------------------

def test_cut_at_returns_expected_schema() -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )
    memberships = cut_at(nodes, 4, min_membership_share=0.0)
    assert list(memberships.columns) == ["oracle_id", "name", "weight", "topic_id", "share"]
    assert memberships["share"].between(0, 1 + 1e-9).all()
    assert set(memberships["topic_id"]) <= set(nodes.keys())


def test_cut_at_root_only_gives_every_card_full_share() -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )
    memberships = cut_at(nodes, 1)
    assert set(memberships["oracle_id"]) == set(pool["oracle_id"])
    assert np.allclose(memberships["share"].to_numpy(), 1.0)


def test_cut_at_max_topics_per_card_caps_row_count() -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )
    memberships = cut_at(nodes, 100, min_membership_share=0.0, max_topics_per_card=1)
    assert (memberships.groupby("oracle_id").size() <= 1).all()


def test_cut_at_does_not_dilute_share_across_unrelated_nodes() -> None:
    """Regression test for a real bug found live (see `cut_at`'s own docstring): a card that
    clears its own membership threshold in many different frontier nodes -- Cultivate did so in
    44 of 1,200 nodes in one real build -- must keep those real per-node shares, not have them
    diluted toward zero by summing `weight` across nodes from unrelated local SymNMF fits (which
    have no shared scale). Hand-built nodes here stand in for "many different branches," since
    the toy `build_hierarchy` fixtures elsewhere in this file are too small to produce that kind
    of fan-out on their own."""
    root = HierarchyNode(
        node_id=0, parent_id=None, depth=0, n_cards=1, stability=None, is_leaf=False,
        members=pd.DataFrame({"oracle_id": ["x"], "name": ["X"], "weight": [1.0], "share": [1.0]}),
    )
    # "x" clears MIN_MEMBERSHIP_SHARE (0.10) in each of 5 unrelated nodes -- own shares 0.90 down
    # to 0.50 -- but at wildly different raw `weight` scales (as different local SymNMF fits
    # would produce), so any cross-node sum-and-divide would land nowhere near those real values.
    other_nodes = [
        HierarchyNode(
            node_id=i,
            parent_id=0,
            depth=1,
            n_cards=1,
            stability=1.0,
            is_leaf=True,
            members=pd.DataFrame(
                {"oracle_id": ["x"], "name": ["X"], "weight": [weight], "share": [share]}
            ),
        )
        for i, (weight, share) in enumerate(
            [(3.0, 0.90), (0.05, 0.80), (10.0, 0.70), (0.5, 0.60), (1.5, 0.50)], start=1
        )
    ]
    nodes = {n.node_id: n for n in [root] + other_nodes}

    memberships = cut_at(nodes, 5, max_topics_per_card=5)

    x_rows = memberships[memberships["oracle_id"] == "x"].sort_values("share", ascending=False)
    assert len(x_rows) == 5  # every real membership survives, none diluted below 0.10
    np.testing.assert_allclose(sorted(x_rows["share"], reverse=True), [0.90, 0.80, 0.70, 0.60, 0.50])


def test_cut_at_fan_out_cap_keeps_the_strongest_shares() -> None:
    root = HierarchyNode(
        node_id=0, parent_id=None, depth=0, n_cards=1, stability=None, is_leaf=False,
        members=pd.DataFrame({"oracle_id": ["x"], "name": ["X"], "weight": [1.0], "share": [1.0]}),
    )
    other_nodes = [
        HierarchyNode(
            node_id=i, parent_id=0, depth=1, n_cards=1, stability=1.0, is_leaf=True,
            members=pd.DataFrame({"oracle_id": ["x"], "name": ["X"], "weight": [share], "share": [share]}),
        )
        for i, share in enumerate([0.90, 0.70, 0.50, 0.30, 0.20], start=1)
    ]
    nodes = {n.node_id: n for n in [root] + other_nodes}

    memberships = cut_at(nodes, 5, max_topics_per_card=2)

    x_rows = memberships[memberships["oracle_id"] == "x"]
    assert len(x_rows) == 2
    np.testing.assert_allclose(sorted(x_rows["share"], reverse=True), [0.90, 0.70])


# --- save_hierarchy / load_hierarchy round-trip ------------------------------------------------

def test_save_and_load_hierarchy_round_trips(tmp_path) -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )

    save_hierarchy(nodes, out_dir=tmp_path)
    reloaded = load_hierarchy(out_dir=tmp_path)

    assert set(reloaded.keys()) == set(nodes.keys())
    for node_id, node in nodes.items():
        reloaded_node = reloaded[node_id]
        assert reloaded_node.parent_id == node.parent_id
        assert reloaded_node.depth == node.depth
        assert reloaded_node.is_leaf == node.is_leaf
        assert set(reloaded_node.members["oracle_id"]) == set(node.members["oracle_id"])

    before = cut_at(nodes, 4)
    after = cut_at(reloaded, 4)
    pd.testing.assert_frame_equal(
        before.sort_values(["oracle_id", "topic_id"]).reset_index(drop=True),
        after.sort_values(["oracle_id", "topic_id"]).reset_index(drop=True),
    )


# --- granularity_curve -------------------------------------------------------------------------

def _insert_card_with_tag(conn, oracle_id, tag) -> None:
    conn.execute("INSERT OR REPLACE INTO cards (oracle_id, name, legal_commander) VALUES (?, ?, 1)", (oracle_id, oracle_id))
    conn.execute(
        "INSERT OR IGNORE INTO card_tags (oracle_id, tag, source) VALUES (?, ?, 'tagger_bulk')", (oracle_id, tag)
    )


def test_granularity_curve_returns_one_row_per_leaf_count_with_expected_columns(db) -> None:
    S = _hier_block_S()
    pool = _hier_pool()
    for oid in pool["oracle_id"]:
        _insert_card_with_tag(db, oid, "shared-tag")
    db.commit()

    nodes = build_hierarchy(
        S, pool, top_k=2, child_k=2, min_node_cards=5, min_child_stability=0.3, max_depth=3, seed=0
    )
    curve = granularity_curve(nodes, db, leaf_counts=[1, 2, 4], n_null_draws=20)

    assert list(curve["requested_leaves"]) == [1, 2, 4]
    assert set(curve.columns) == {
        "requested_leaves", "actual_leaves", "n_cards", "vocab_size",
        "n_labels_covered", "n_labels_resolved_010", "median_labels_per_topic", "median_purity",
    }
    assert (curve["actual_leaves"] <= curve["requested_leaves"]).all()
    assert (curve["n_cards"] == len(pool)).all()
