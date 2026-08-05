import math

import numpy as np
import pytest

from edhcut.analysis.cooccurrence import (
    PRECON_SIMILARITY_THRESHOLD,
    PRECON_WEIGHT_FLOOR,
    CooccurrenceResult,
    _slot_label,
    build_card_index,
    build_cooccurrence,
    compute_lift,
    compute_pmi,
    deck_weight,
    top_associated,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _ensure_cards(conn, *oracle_ids: str) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO cards (oracle_id, name) VALUES (?, ?)",
        [(oid, oid) for oid in oracle_ids],
    )


def _insert_deck(conn, deck_id, *, slot_key, cards, precon_similarity=None, source_id=None):
    _ensure_cards(conn, *cards)
    conn.execute(
        "INSERT INTO decks (deck_id, source, source_id, slot_key, precon_similarity) "
        "VALUES (?, 'archidekt', ?, ?, ?)",
        (deck_id, source_id or f"d{deck_id}", slot_key, precon_similarity),
    )
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
        [(deck_id, oid) for oid in cards],
    )
    conn.commit()


# --- deck_weight ---------------------------------------------------------------------------

def test_deck_weight_full_below_threshold() -> None:
    assert deck_weight(None) == 1.0
    assert deck_weight(0.5) == 1.0
    assert deck_weight(PRECON_SIMILARITY_THRESHOLD) == 1.0


def test_deck_weight_floors_at_exact_copy() -> None:
    assert deck_weight(1.0) == pytest.approx(PRECON_WEIGHT_FLOOR)


def test_deck_weight_decays_linearly_between_threshold_and_one() -> None:
    midpoint = PRECON_SIMILARITY_THRESHOLD + (1.0 - PRECON_SIMILARITY_THRESHOLD) / 2
    expected = (1.0 + PRECON_WEIGHT_FLOOR) / 2
    assert deck_weight(midpoint) == pytest.approx(expected)


def test_deck_weight_monotonically_decreasing_above_threshold() -> None:
    values = [deck_weight(s) for s in (0.9, 0.93, 0.96, 0.99, 1.0)]
    assert values == sorted(values, reverse=True)


# --- _slot_label -----------------------------------------------------------------------------

def test_slot_label_single_commander() -> None:
    assert _slot_label(["Krenko, Mob Boss"]) == "krenko"


def test_slot_label_partner_pair() -> None:
    assert (
        _slot_label(["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"])
        == "yoshimaru_bruse_tarl"
    )


# --- build_card_index --------------------------------------------------------------------------

def test_build_card_index_excludes_cards_below_min_decks(db) -> None:
    _insert_deck(db, 1, slot_key="s", cards=["a", "b"])
    _insert_deck(db, 2, slot_key="s", cards=["a", "b"])
    _insert_deck(db, 3, slot_key="s", cards=["a"])  # b only in 2 decks, a in 3

    index = build_card_index(db, min_decks=3)
    assert set(index["oracle_id"]) == {"a"}


def test_build_card_index_rows_are_oracle_id_sorted(db) -> None:
    _insert_deck(db, 1, slot_key="s", cards=["z", "a", "m"])
    _insert_deck(db, 2, slot_key="s", cards=["z", "a", "m"])
    _insert_deck(db, 3, slot_key="s", cards=["z", "a", "m"])

    index = build_card_index(db, min_decks=3).sort_values("row")
    assert list(index["oracle_id"]) == ["a", "m", "z"]
    assert list(index["row"]) == [0, 1, 2]


# --- build_cooccurrence --------------------------------------------------------------------

def test_build_cooccurrence_counts_pairs_across_decks(db) -> None:
    _insert_deck(db, 1, slot_key="s", cards=["a", "b", "c"])
    _insert_deck(db, 2, slot_key="s", cards=["a", "b"])
    index = build_card_index(db, min_decks=1)

    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))
    assert result.pair_raw[tuple(sorted((row["a"], row["b"])))] == 2
    assert result.pair_raw[tuple(sorted((row["a"], row["c"])))] == 1
    assert result.deck_count == 2
    assert result.total_weight == pytest.approx(2.0)


def test_build_cooccurrence_scoped_to_slot_key(db) -> None:
    _insert_deck(db, 1, slot_key="slot-a", cards=["x", "y"])
    _insert_deck(db, 2, slot_key="slot-b", cards=["x", "y"])
    index = build_card_index(db, min_decks=1)

    result = build_cooccurrence(db, index, slot_key="slot-a")
    assert result.deck_count == 1


def test_build_cooccurrence_applies_novelty_weight(db) -> None:
    _insert_deck(db, 1, slot_key="s", cards=["a", "b"], precon_similarity=1.0)
    index = build_card_index(db, min_decks=1)

    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))
    pair = tuple(sorted((row["a"], row["b"])))
    assert result.pair_weighted[pair] == pytest.approx(PRECON_WEIGHT_FLOOR)
    assert result.pair_raw[pair] == 1  # raw count is unaffected by weighting
    assert result.total_weight == pytest.approx(PRECON_WEIGHT_FLOOR)


def test_weighted_matrix_is_symmetric_with_zero_diagonal(db) -> None:
    _insert_deck(db, 1, slot_key="s", cards=["a", "b"])
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)

    m = result.weighted_matrix().toarray()
    assert np.array_equal(m, m.T)
    assert np.all(np.diag(m) == 0)


# --- compute_pmi / compute_lift ----------------------------------------------------------------

def test_pmi_masks_pairs_below_min_pair_count(db) -> None:
    _insert_deck(db, 1, slot_key="s", cards=["a", "b"])
    _insert_deck(db, 2, slot_key="s", cards=["a", "b"])
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))
    pair = tuple(sorted((row["a"], row["b"])))

    pmi = compute_pmi(result, min_pair_count=3)
    assert pmi[pair] == 0.0  # only seen twice, masked out at min_pair_count=3

    pmi_lenient = compute_pmi(result, min_pair_count=2)
    assert pmi_lenient[pair] != 0.0


def test_pmi_higher_for_more_exclusive_pair(db) -> None:
    # a & b always appear together and rarely with anything else -> high PMI.
    # c appears in almost every deck (with everything) -> low/negative PMI with a.
    for i in range(5):
        _insert_deck(db, i, slot_key="s", cards=["a", "b", "c"])
    for i in range(5, 15):
        _insert_deck(db, i, slot_key="s", cards=["c", "filler"])
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))

    pmi = compute_pmi(result, min_pair_count=3)
    pmi_ab = pmi[tuple(sorted((row["a"], row["b"])))]
    pmi_ac = pmi[tuple(sorted((row["a"], row["c"])))]
    assert pmi_ab > pmi_ac


def test_lift_greater_than_one_for_positively_associated_pair(db) -> None:
    for i in range(5):
        _insert_deck(db, i, slot_key="s", cards=["a", "b"])
    for i in range(5, 10):
        _insert_deck(db, i, slot_key="s", cards=["z"])  # unrelated decks so P(a), P(b) < 1
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))

    lift = compute_lift(result, min_pair_count=3)
    assert lift[tuple(sorted((row["a"], row["b"])))] > 1.0


def test_pmi_discount_promotes_well_supported_pair_over_low_count_tie() -> None:
    # Hand-constructed scope (matches the real live-corpus artifact found for e.g. Purphoros,
    # God of the Forge — see docs/devlog/6.1-cooccurrence.md): card 0 = "hub" (marginal 100),
    # card 1 = "ceiling" (marginal 3, joint 3 -> a rare card that happens to co-occur with hub
    # in *every* one of its 3 appearances, pure coincidence at that sample size), card 2 =
    # "better" (marginal 25, joint 20 -> a much better-supported, if individually
    # smaller-ratio, association). Undiscounted PMI ranks the 3-deck coincidence above the
    # 20-deck association; discounting should invert that.
    result = CooccurrenceResult(
        n_cards=3,
        pair_raw={(0, 1): 3, (0, 2): 20},
        pair_weighted={(0, 1): 3.0, (0, 2): 20.0},
        raw_marginal=np.array([100, 3, 25]),
        weighted_marginal=np.array([100.0, 3.0, 25.0]),
        total_weight=200.0,
        deck_count=200,
    )
    undiscounted = compute_pmi(result, k=0.0, min_pair_count=1, discount=False)
    discounted = compute_pmi(result, k=0.0, min_pair_count=1, discount=True)

    assert undiscounted[(0, 1)] > undiscounted[(0, 2)]  # raw PMI favors the coincidence
    assert discounted[(0, 2)] > discounted[(0, 1)]  # discounting favors the real association


def test_pmi_discount_barely_touches_well_supported_pairs(db) -> None:
    for i in range(50):
        _insert_deck(db, i, slot_key="s", cards=["a", "b"])
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))
    pair = tuple(sorted((row["a"], row["b"])))

    undiscounted = compute_pmi(result, min_pair_count=1, discount=False)
    discounted = compute_pmi(result, min_pair_count=1, discount=True)
    assert discounted[pair] == pytest.approx(undiscounted[pair], rel=0.05)


def test_pmi_matrix_is_symmetric(db) -> None:
    for i in range(4):
        _insert_deck(db, i, slot_key="s", cards=["a", "b", "c"])
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)

    pmi = compute_pmi(result, min_pair_count=1).toarray()
    assert np.allclose(pmi, pmi.T)


# --- top_associated ------------------------------------------------------------------------

def test_top_associated_excludes_self_and_zero_entries(db) -> None:
    for i in range(4):
        _insert_deck(db, i, slot_key="s", cards=["a", "b"])
    _insert_deck(db, 100, slot_key="s", cards=["a", "z"])  # only seen once with z
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    pmi = compute_pmi(result, min_pair_count=3)

    a_oracle_id = index.loc[index["name"] == "a", "oracle_id"].iloc[0]
    top = top_associated(pmi, index, a_oracle_id, k=5)
    names = list(top["name"])
    assert "a" not in names
    assert "z" not in names  # masked out (seen <3 times), so PMI entry is absent/zero
    assert "b" in names


def test_top_associated_raises_for_unknown_card(db) -> None:
    _insert_deck(db, 1, slot_key="s", cards=["a", "b"])
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    pmi = compute_pmi(result, min_pair_count=1)

    with pytest.raises(KeyError):
        top_associated(pmi, index, "nonexistent-oracle-id")
