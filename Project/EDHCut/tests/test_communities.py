import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from edhcut.analysis.communities import (
    ResolutionResult,
    basic_land_mask,
    best_resolution,
    build_and_save,
    build_graph,
    cluster_members,
    cluster_of,
    load_clusters,
    seed_stability,
    sweep_resolutions,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, name, *, type_line=None) -> None:
    conn.execute(
        "INSERT INTO cards (oracle_id, name, type_line) VALUES (?, ?, ?)", (oracle_id, name, type_line)
    )


def _insert_deck(conn, deck_id, *, slot_key, cards) -> None:
    conn.execute(
        "INSERT INTO decks (deck_id, source, source_id, slot_key) VALUES (?, 'archidekt', ?, ?)",
        (deck_id, f"d{deck_id}", slot_key),
    )
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
        [(deck_id, oid) for oid in cards],
    )


# --- basic_land_mask ------------------------------------------------------------------------

def test_basic_land_mask_flags_basics_and_snow_variants_not_nonbasic_lands(db) -> None:
    _insert_card(db, "forest", "Forest", type_line="Basic Land — Forest")
    _insert_card(db, "snow-island", "Snow-Covered Island", type_line="Basic Snow Land — Island")
    _insert_card(db, "wastes", "Wastes", type_line="Basic Land")
    _insert_card(db, "command-tower", "Command Tower", type_line="Land")
    _insert_card(db, "goblin", "Skirk Prospector", type_line="Creature — Goblin")
    db.commit()
    card_index = pd.DataFrame({
        "oracle_id": ["forest", "snow-island", "wastes", "command-tower", "goblin"],
        "row": [0, 1, 2, 3, 4],
    })

    mask = basic_land_mask(db, card_index)

    assert mask.tolist() == [True, True, True, False, False]


# --- build_graph -----------------------------------------------------------------------------

def _dense_to_tscore(values: list[list[float]]) -> sparse.csr_matrix:
    return sparse.csr_matrix(np.array(values, dtype=np.float64))


def test_build_graph_drops_negative_and_zero_edges() -> None:
    # 0-1 positive, 0-2 negative, 1-2 zero -- only the 0-1 edge should survive.
    tscore = _dense_to_tscore([
        [0.0, 5.0, -3.0],
        [5.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0],
    ])
    graph, has_edge = build_graph(tscore, top_k=15)

    assert has_edge.tolist() == [True, True, False]
    assert graph.vcount() == 2
    assert graph.ecount() == 1


def test_build_graph_top_k_caps_degree_per_node() -> None:
    # Node 0 has 4 positive partners (1, 2, 3, 4) but top_k=2 -- only its 2 strongest (1, 2)
    # should survive *from node 0's own side*. Nodes 3 and 4 each also get 2 strong "distractor"
    # edges to their own dummy partners, stronger than their weak edge back to node 0, so their
    # own top-2 doesn't include node 0 either -- isolating top_k capping from union
    # symmetrization's "kept from the *other* side" behavior (covered by a separate test below).
    n = 9
    tscore = np.zeros((n, n))
    tscore[0, 1] = tscore[1, 0] = 10.0
    tscore[0, 2] = tscore[2, 0] = 9.0
    tscore[0, 3] = tscore[3, 0] = 1.0
    tscore[0, 4] = tscore[4, 0] = 0.5
    for node, distractors in [(3, [(5, 50.0), (6, 49.0)]), (4, [(7, 50.0), (8, 49.0)])]:
        for other, strength in distractors:
            tscore[node, other] = tscore[other, node] = strength
    graph, has_edge = build_graph(_dense_to_tscore(tscore.tolist()), top_k=2)

    orig_rows = graph.vs["orig_row"]
    v0 = orig_rows.index(0)
    neighbor_orig_rows = {orig_rows[n_idx] for n_idx in graph.neighbors(v0)}
    assert neighbor_orig_rows == {1, 2}  # the two strongest (10.0, 9.0), not 3/4


def test_build_graph_union_symmetrization_keeps_one_sided_strong_edge() -> None:
    # Node 0's top_k=1 partner is node 1 (strength 10). Node 1 has many *other* strong partners
    # (all stronger than its edge to 0), so 0 doesn't make node 1's own top-1 list. Union
    # symmetrization should still keep the 0-1 edge (node 0 kept it on its own side); mutual/
    # intersection would have dropped it.
    n = 5
    tscore = np.zeros((n, n))
    tscore[0, 1] = tscore[1, 0] = 10.0
    for j, v in [(2, 20.0), (3, 19.0), (4, 18.0)]:
        tscore[1, j] = tscore[j, 1] = v
    graph, has_edge = build_graph(_dense_to_tscore(tscore.tolist()), top_k=1)

    orig_rows = graph.vs["orig_row"]
    v0 = orig_rows.index(0)
    v1 = orig_rows.index(1)
    assert graph.are_adjacent(v0, v1)


def test_build_graph_exclude_removes_node_and_its_edges_from_everyone_elses_top_k() -> None:
    n = 3
    tscore = np.zeros((n, n))
    tscore[0, 1] = tscore[1, 0] = 10.0  # node 0's only positive edge is to the excluded node 1
    tscore[1, 2] = tscore[2, 1] = 5.0
    graph, has_edge = build_graph(
        _dense_to_tscore(tscore.tolist()), top_k=15, exclude=np.array([False, True, False])
    )

    # node 1 excluded entirely; node 0 loses its only edge and so is dropped too; node 2 loses
    # its only edge (to the excluded node) and is dropped too.
    assert has_edge.tolist() == [False, False, False]
    assert graph.vcount() == 0


def test_build_graph_orig_row_maps_back_correctly_after_dropping_zero_degree_nodes() -> None:
    tscore = _dense_to_tscore([
        [0.0, 5.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],  # row 2: fully isolated, dropped
    ])
    graph, has_edge = build_graph(tscore, top_k=15)

    assert has_edge.tolist() == [True, True, False]
    assert sorted(graph.vs["orig_row"]) == [0, 1]


# --- sweep_resolutions / best_resolution ------------------------------------------------------

def _two_clear_communities_graph():
    # Two 4-cliques (0-3, 4-7) joined by one weak bridge edge -- unambiguous community structure
    # for a sanity check that isn't fussy about exact modularity values.
    import igraph as ig

    edges = []
    weights = []
    for group in ([0, 1, 2, 3], [4, 5, 6, 7]):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.append((group[i], group[j]))
                weights.append(10.0)
    edges.append((3, 4))
    weights.append(0.1)
    graph = ig.Graph(n=8, edges=edges)
    graph.es["weight"] = weights
    return graph


def test_sweep_resolutions_finds_two_communities_in_obvious_structure() -> None:
    graph = _two_clear_communities_graph()
    results = sweep_resolutions(graph, resolutions=[1.0])
    assert results[0].n_clusters == 2
    assert results[0].modularity > 0.3


def test_sweep_resolutions_returns_one_result_per_resolution() -> None:
    graph = _two_clear_communities_graph()
    results = sweep_resolutions(graph, resolutions=[0.5, 1.0, 2.0])
    assert [r.resolution for r in results] == [0.5, 1.0, 2.0]
    assert all(isinstance(r, ResolutionResult) for r in results)


def test_best_resolution_picks_max_modularity() -> None:
    results = [
        ResolutionResult(0.5, 3, modularity=0.4, singleton_count=0, median_size=2.0, max_size=3),
        ResolutionResult(1.0, 2, modularity=0.7, singleton_count=0, median_size=4.0, max_size=4),
        ResolutionResult(2.0, 5, modularity=0.3, singleton_count=2, median_size=1.0, max_size=2),
    ]
    assert best_resolution(results).resolution == 1.0


# --- seed_stability ----------------------------------------------------------------------------

def test_seed_stability_identical_seed_is_perfect_agreement() -> None:
    graph = _two_clear_communities_graph()
    _, mean_ari, min_ari = seed_stability(graph, 1.0, primary_seed=42, other_seeds=[42, 42])
    assert mean_ari == pytest.approx(1.0)
    assert min_ari == pytest.approx(1.0)


def test_seed_stability_on_obvious_structure_is_high_across_different_seeds() -> None:
    graph = _two_clear_communities_graph()
    _, mean_ari, min_ari = seed_stability(graph, 1.0, primary_seed=42, other_seeds=[1, 2, 3])
    assert mean_ari > 0.9
    assert min_ari > 0.9


# --- cluster_of / cluster_members / load_clusters ----------------------------------------------

def _toy_clusters() -> pd.DataFrame:
    return pd.DataFrame({
        "oracle_id": ["a", "b", "c", "d"],
        "name": ["Card A", "Card B", "Card C", "Card D"],
        "is_land": [False, False, True, False],
        "cluster_id": [0, 0, 1, -1],
    })


def _toy_play_rates() -> pd.DataFrame:
    return pd.DataFrame({
        "oracle_id": ["a", "b", "c"],
        "deck_count": [100, 50, 10],
        "eligible_deck_count": [200, 200, 200],
        "play_rate": [0.5, 0.25, 0.05],
    })


def test_cluster_of_returns_id_for_clustered_card() -> None:
    assert cluster_of(_toy_clusters(), "a") == 0


def test_cluster_of_returns_none_for_dropped_card() -> None:
    assert cluster_of(_toy_clusters(), "d") is None


def test_cluster_of_returns_none_for_unknown_oracle_id() -> None:
    assert cluster_of(_toy_clusters(), "not-in-pool") is None


def test_cluster_members_ranks_by_play_rate_descending() -> None:
    members = cluster_members(_toy_clusters(), _toy_play_rates(), 0)
    assert list(members["oracle_id"]) == ["a", "b"]  # a (0.5) before b (0.25)


def test_cluster_members_unranked_card_sorts_last_not_dropped() -> None:
    # "e" is in the cluster but has no play_rate row at all.
    clusters = pd.concat([_toy_clusters(), pd.DataFrame(
        {"oracle_id": ["e"], "name": ["Card E"], "is_land": [False], "cluster_id": [0]}
    )], ignore_index=True)
    members = cluster_members(clusters, _toy_play_rates(), 0)
    assert list(members["oracle_id"]) == ["a", "b", "e"]


def test_cluster_members_respects_k() -> None:
    members = cluster_members(_toy_clusters(), _toy_play_rates(), 0, k=1)
    assert list(members["oracle_id"]) == ["a"]


# --- build_and_save end-to-end -------------------------------------------------------------------

def test_build_and_save_writes_clusters_and_manifest(db, tmp_path) -> None:
    out_dir = tmp_path / "kb"
    out_dir.mkdir()

    # 5 cards: 0-1-2 form an obvious triangle, 3 is a basic land (excluded), 4 has no positive
    # edge to anything (dropped for lack of connection, not exclusion).
    names = ["Goblin Warchief", "Skirk Prospector", "Impact Tremors", "Forest", "Lonely Card"]
    oracle_ids = [f"card-{i}" for i in range(5)]
    for oid, name in zip(oracle_ids, names):
        type_line = "Basic Land — Forest" if name == "Forest" else "Creature"
        _insert_card(db, oid, name, type_line=type_line)
    db.commit()

    card_index = pd.DataFrame({
        "oracle_id": oracle_ids, "name": names, "is_land": [False, False, False, True, False],
        "row": [0, 1, 2, 3, 4],
    })
    card_index.to_parquet(out_dir / "card_index.parquet", index=False)

    goblin, skirk, impact, forest, lonely = oracle_ids
    # 6 decks form the Goblin/Skirk/Impact triangle; 3 of those 6 also run Forest, giving it
    # real positive edges to all three (proving exclusion, not lack of signal, drops it from the
    # graph). Lonely Card sits alone in its own deck -- never co-occurs with anything.
    for i in range(1, 7):
        cards = [goblin, skirk, impact] + ([forest] if i <= 3 else [])
        _insert_deck(db, i, slot_key="s", cards=cards)
    _insert_deck(db, 7, slot_key="s", cards=[lonely])
    db.commit()

    stats = build_and_save(db, out_dir=out_dir)

    assert stats.n_cards_total == 5
    assert stats.n_basic_lands_excluded == 1
    assert stats.n_cards_in_graph == 3  # 0, 1, 2 -- Forest excluded, Lonely Card dropped (no edges)

    clusters = load_clusters(out_dir)
    assert set(clusters["oracle_id"]) == set(oracle_ids)
    by_id = clusters.set_index("oracle_id")
    assert by_id.loc["card-3", "cluster_id"] == -1  # excluded basic land
    assert by_id.loc["card-4", "cluster_id"] == -1  # dropped, no positive edges
    # the triangle (0, 1, 2) all land in the same cluster
    assert by_id.loc["card-0", "cluster_id"] == by_id.loc["card-1", "cluster_id"] == by_id.loc["card-2", "cluster_id"]
    assert by_id.loc["card-0", "cluster_id"] >= 0

    manifest = json.loads((out_dir / "communities_manifest.json").read_text())
    assert manifest["n_cards_total"] == 5
    assert manifest["n_basic_lands_excluded"] == 1
    assert manifest["n_cards_in_graph"] == 3
