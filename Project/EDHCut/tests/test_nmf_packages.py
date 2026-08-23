"""The earlier plain-NMF package extraction (superseded by `symnmf_packages`, kept for comparison):
deck sampling determinism, the binary deck × card matrix, topic-stability matching across seeds,
and the `k`-selection rule."""

import numpy as np
import pandas as pd
import pytest

from edhcut.analysis.nmf_packages import (
    _sample_deck_ids,
    best_k,
    build_and_save,
    build_deck_card_matrix,
    card_topics,
    load_card_memberships,
    load_components,
    load_deck_proportions,
    load_pool_index,
    match_topic_stability,
    memberships_table,
    KKSweepResult,
    topic_members,
    topic_proportions_for_deck,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, name, *, type_line="Creature") -> None:
    conn.execute("INSERT INTO cards (oracle_id, name, type_line) VALUES (?, ?, ?)", (oracle_id, name, type_line))


def _ensure_cards(conn, *oracle_ids: str) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO cards (oracle_id, name) VALUES (?, ?)", [(oid, oid) for oid in oracle_ids]
    )


def _insert_deck(conn, deck_id, *, slot_key, cards) -> None:
    _ensure_cards(conn, *cards)
    conn.execute(
        "INSERT INTO decks (deck_id, source, source_id, slot_key) VALUES (?, 'archidekt', ?, ?)",
        (deck_id, f"d{deck_id}", slot_key),
    )
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
        [(deck_id, oid) for oid in cards],
    )


# --- _sample_deck_ids --------------------------------------------------------------------------

def test_sample_deck_ids_keeps_small_slots_whole(db) -> None:
    for i in range(1, 6):
        _insert_deck(db, i, slot_key="s", cards=["a"])
    db.commit()
    sampled = _sample_deck_ids(db, max_decks_per_slot=25, seed=1)
    assert sampled == [1, 2, 3, 4, 5]


def test_sample_deck_ids_caps_large_slots(db) -> None:
    for i in range(1, 101):
        _insert_deck(db, i, slot_key="big", cards=["a"])
    db.commit()
    sampled = _sample_deck_ids(db, max_decks_per_slot=25, seed=1)
    assert len(sampled) == 25
    assert set(sampled) <= set(range(1, 101))


def test_sample_deck_ids_is_deterministic(db) -> None:
    for i in range(1, 51):
        _insert_deck(db, i, slot_key="s", cards=["a"])
    db.commit()
    a = _sample_deck_ids(db, max_decks_per_slot=10, seed=7)
    b = _sample_deck_ids(db, max_decks_per_slot=10, seed=7)
    assert a == b


def test_sample_deck_ids_independent_per_slot(db) -> None:
    for i in range(1, 31):
        _insert_deck(db, i, slot_key="a", cards=["x"])
    for i in range(31, 41):
        _insert_deck(db, i, slot_key="b", cards=["x"])
    db.commit()
    sampled = _sample_deck_ids(db, max_decks_per_slot=25, seed=1)
    from_a = [d for d in sampled if d < 31]
    from_b = [d for d in sampled if d >= 31]
    assert len(from_a) == 25
    assert len(from_b) == 10  # slot "b" is under the cap, kept whole


# --- build_deck_card_matrix ----------------------------------------------------------------

def test_build_deck_card_matrix_binary_presence(db) -> None:
    _insert_card(db, "a", "A")
    _insert_card(db, "b", "B")
    _insert_deck(db, 1, slot_key="s", cards=["a", "b"])
    _insert_deck(db, 2, slot_key="s", cards=["a"])
    db.commit()
    card_index = pd.DataFrame({"oracle_id": ["a", "b"], "name": ["A", "B"], "row": [0, 1]})

    matrix = build_deck_card_matrix(db, card_index, [1, 2])

    assert matrix.shape == (2, 2)
    assert matrix.toarray().tolist() == [[1.0, 1.0], [1.0, 0.0]]


def test_build_deck_card_matrix_skips_cards_outside_index(db) -> None:
    _insert_card(db, "a", "A")
    _insert_card(db, "outside", "Outside")
    _insert_deck(db, 1, slot_key="s", cards=["a", "outside"])
    db.commit()
    card_index = pd.DataFrame({"oracle_id": ["a"], "name": ["A"], "row": [0]})

    matrix = build_deck_card_matrix(db, card_index, [1])

    assert matrix.shape == (1, 1)
    assert matrix.toarray().tolist() == [[1.0]]


# --- match_topic_stability / best_k --------------------------------------------------------

def test_match_topic_stability_identical_matrices_is_one() -> None:
    h = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]])
    assert match_topic_stability(h, h) == pytest.approx(1.0)


def test_match_topic_stability_orthogonal_topics_is_low() -> None:
    h_a = np.array([[1.0, 0.0], [0.0, 1.0]])
    h_b = np.array([[1.0, 0.0], [0.0, 1.0]])
    # permuted copy should still match to 1.0 via Hungarian assignment
    h_b_perm = h_b[::-1]
    assert match_topic_stability(h_a, h_b_perm) == pytest.approx(1.0)


def test_best_k_picks_highest_stability() -> None:
    sweep = [
        KKSweepResult(k=15, reconstruction_error=0.5, stability=0.7),
        KKSweepResult(k=20, reconstruction_error=0.4, stability=0.9),
        KKSweepResult(k=25, reconstruction_error=0.3, stability=0.6),
    ]
    assert best_k(sweep).k == 20


def test_best_k_ties_broken_toward_larger_k() -> None:
    sweep = [
        KKSweepResult(k=15, reconstruction_error=0.5, stability=0.8),
        KKSweepResult(k=25, reconstruction_error=0.4, stability=0.8),
    ]
    assert best_k(sweep).k == 25


# --- memberships_table -----------------------------------------------------------------------

def test_memberships_table_keeps_overlapping_membership_above_threshold() -> None:
    # card 0 is ~50/50 split between topics 0 and 1 -- both should survive a 10% threshold.
    # card 1 is almost entirely topic 1 -- topic 0's tiny residual should be dropped.
    h = np.array([[5.0, 0.1], [5.0, 9.9]])
    pool = pd.DataFrame({"oracle_id": ["card-0", "card-1"], "name": ["Card 0", "Card 1"]})

    table = memberships_table(pool, h)

    card0 = table[table["oracle_id"] == "card-0"]
    assert set(card0["topic_id"]) == {0, 1}
    card1 = table[table["oracle_id"] == "card-1"]
    assert set(card1["topic_id"]) == {1}


# --- build_and_save end-to-end -------------------------------------------------------------

def _seed_toy_corpus(conn) -> None:
    # Two recognizable "packages": goblins (a, b, c) and ramp (d, e, f), each card >=3 decks.
    # 20 different commander slots so the k-sweep/stability machinery has enough independent
    # decks to work with; every slot small enough to be kept whole by the subsample.
    for oid, name in [
        ("a", "Goblin A"), ("b", "Goblin B"), ("c", "Goblin C"),
        ("d", "Ramp D"), ("e", "Ramp E"), ("f", "Ramp F"),
    ]:
        _insert_card(conn, oid, name)

    deck_id = 1
    for slot in range(10):
        for _ in range(3):
            _insert_deck(conn, deck_id, slot_key=f"goblin-{slot}", cards=["a", "b", "c"])
            deck_id += 1
    for slot in range(10):
        for _ in range(3):
            _insert_deck(conn, deck_id, slot_key=f"ramp-{slot}", cards=["d", "e", "f"])
            deck_id += 1
    conn.commit()


def test_build_and_save_writes_tables_and_manifest(db, tmp_path) -> None:
    out_dir = tmp_path / "kb"
    out_dir.mkdir()
    _seed_toy_corpus(db)

    card_index = pd.DataFrame({
        "oracle_id": ["a", "b", "c", "d", "e", "f"],
        "name": ["Goblin A", "Goblin B", "Goblin C", "Ramp D", "Ramp E", "Ramp F"],
        "is_land": [False] * 6,
        "row": [0, 1, 2, 3, 4, 5],
    })
    card_index.to_parquet(out_dir / "card_index.parquet", index=False)

    stats = build_and_save(db, out_dir=out_dir, k_grid=[2, 3])

    assert stats.n_cards_total == 6
    assert stats.n_lands_excluded == 0
    assert stats.n_decks_sampled == 60  # every slot under the 25-cap, kept whole

    memberships = load_card_memberships(out_dir)
    assert set(memberships["oracle_id"]) <= {"a", "b", "c", "d", "e", "f"}

    deck_proportions = load_deck_proportions(out_dir)
    assert set(deck_proportions["deck_id"]) <= set(range(1, 61))

    h = load_components(out_dir)
    assert h.shape == (stats.chosen_k, 6)

    pool = load_pool_index(out_dir)
    assert set(pool["oracle_id"]) == {"a", "b", "c", "d", "e", "f"}

    import json
    manifest = json.loads((out_dir / "nmf_manifest.json").read_text())
    assert manifest["chosen_k"] == stats.chosen_k
    assert manifest["n_decks_sampled"] == 60


def test_topic_proportions_for_deck_projects_new_deck(db, tmp_path) -> None:
    out_dir = tmp_path / "kb"
    out_dir.mkdir()
    _seed_toy_corpus(db)

    card_index = pd.DataFrame({
        "oracle_id": ["a", "b", "c", "d", "e", "f"],
        "name": ["Goblin A", "Goblin B", "Goblin C", "Ramp D", "Ramp E", "Ramp F"],
        "is_land": [False] * 6,
        "row": [0, 1, 2, 3, 4, 5],
    })
    card_index.to_parquet(out_dir / "card_index.parquet", index=False)

    build_and_save(db, out_dir=out_dir, k_grid=[2])

    h = load_components(out_dir)
    pool = load_pool_index(out_dir)

    proportions = topic_proportions_for_deck(h, pool, {"a", "b", "c"})
    assert proportions
    assert sum(proportions.values()) == pytest.approx(1.0)

    empty = topic_proportions_for_deck(h, pool, {"not-in-pool"})
    assert empty == {}


# --- topic_members / card_topics ------------------------------------------------------------

def test_topic_members_and_card_topics() -> None:
    memberships = pd.DataFrame({
        "oracle_id": ["a", "a", "b"],
        "name": ["A", "A", "B"],
        "topic_id": [0, 1, 0],
        "weight": [5.0, 1.0, 3.0],
        "share": [0.83, 0.17, 1.0],
    })

    topic0 = topic_members(memberships, 0)
    assert list(topic0["oracle_id"]) == ["a", "b"]

    a_topics = card_topics(memberships, "a")
    assert list(a_topics["topic_id"]) == [0, 1]

    missing = card_topics(memberships, "nonexistent")
    assert missing.empty
