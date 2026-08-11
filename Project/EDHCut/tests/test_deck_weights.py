import pytest

from edhcut.analysis.deck_weights import (
    DEFAULT_WEIGHT,
    compute_deck_weights,
    compute_near_uniform_weights,
    deck_weight,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_meta_commander(conn, slot_key, *, num_decks) -> None:
    conn.execute("INSERT OR IGNORE INTO cards (oracle_id, name) VALUES (?, ?)", (slot_key, slot_key))
    conn.execute(
        "INSERT INTO meta_commanders "
        "(slot_key, commander_oracle_id, name, color_identity, edhrec_num_decks, sample_target, fetched_at) "
        "VALUES (?, ?, ?, 'r', ?, 5, '2026-08-09')",
        (slot_key, slot_key, slot_key, num_decks),
    )


def _insert_decks(conn, slot_key, count) -> None:
    conn.executemany(
        "INSERT INTO decks (source, source_id, slot_key) VALUES ('archidekt', ?, ?)",
        [(f"{slot_key}-{i}", slot_key) for i in range(count)],
    )


def test_no_meta_commanders_returns_empty(db) -> None:
    assert compute_deck_weights(db) == {}


def test_meta_commander_with_no_harvested_decks_excluded(db) -> None:
    _insert_meta_commander(db, "a", num_decks=1000)
    db.commit()
    assert compute_deck_weights(db) == {}


def test_oversampled_commander_gets_weight_below_one(db) -> None:
    # Two commanders, equal true share (1000 decks each -> 50/50), but "a" is harvested 10x
    # more heavily than "b" -> "a" is oversampled relative to its true share, weight < 1.
    _insert_meta_commander(db, "a", num_decks=1000)
    _insert_meta_commander(db, "b", num_decks=1000)
    _insert_decks(db, "a", 100)
    _insert_decks(db, "b", 10)
    db.commit()

    weights = compute_deck_weights(db)

    assert weights["a"] < 1.0
    assert weights["b"] > 1.0
    # true_share equal (0.5/0.5); sample_share a=100/110, b=10/110 -> w_a = 0.5/(100/110), etc.
    assert weights["a"] == pytest.approx(0.5 / (100 / 110))
    assert weights["b"] == pytest.approx(0.5 / (10 / 110))


def test_weights_reflect_true_share_not_just_sample_ratio(db) -> None:
    # "a" is truly 9x more popular than "b" (9000 vs 1000 decks) but both got harvested equally
    # (50 decks each) -> "a" is *undersampled* relative to its true dominance, weight > 1.
    _insert_meta_commander(db, "a", num_decks=9000)
    _insert_meta_commander(db, "b", num_decks=1000)
    _insert_decks(db, "a", 50)
    _insert_decks(db, "b", 50)
    db.commit()

    weights = compute_deck_weights(db)

    assert weights["a"] > 1.0
    assert weights["b"] < 1.0
    assert weights["a"] == pytest.approx(0.9 / 0.5)
    assert weights["b"] == pytest.approx(0.1 / 0.5)


def test_roster_style_full_corpus_counts_regardless_of_cohort(db) -> None:
    """A slot's harvested-deck count isn't restricted by cohort -- a roster commander's full
    (large) roster-cohort corpus is exactly what should be recognized as oversampled, without
    needing a separate meta_sample-cohort harvest for the same commander."""
    _insert_meta_commander(db, "krenko", num_decks=42881)
    _insert_meta_commander(db, "obscure", num_decks=2500)
    db.executemany(
        "INSERT INTO decks (source, source_id, slot_key, cohort) VALUES ('archidekt', ?, 'krenko', 'roster')",
        [(f"k{i}",) for i in range(2000)],
    )
    _insert_decks(db, "obscure", 6)
    db.commit()

    weights = compute_deck_weights(db)

    assert weights["krenko"] < 1.0  # 2,000 harvested decks vs. a ~5% true share -> way oversampled


def test_deck_weight_defaults_missing_slot_to_neutral(db) -> None:
    _insert_meta_commander(db, "a", num_decks=1000)
    _insert_decks(db, "a", 10)
    db.commit()
    weights = compute_deck_weights(db)

    assert deck_weight(weights, "some-other-slot") == DEFAULT_WEIGHT
    assert deck_weight(weights, None) == DEFAULT_WEIGHT
    assert deck_weight(weights, "a") == weights["a"]


def test_deck_weight_custom_default(db) -> None:
    assert deck_weight({}, "unknown", default=0.0) == 0.0


# --- compute_near_uniform_weights -------------------------------------------------------------

def test_near_uniform_no_decks_returns_empty(db) -> None:
    assert compute_near_uniform_weights(db) == {}


def test_near_uniform_flattens_oversampled_slot_regardless_of_meta_commanders(db) -> None:
    # "a" is not in meta_commanders at all (e.g. a roster slot below the harvest threshold) --
    # compute_near_uniform_weights still down-weights it, unlike compute_deck_weights.
    _insert_decks(db, "a", 300)
    _insert_decks(db, "b", 10)
    _insert_decks(db, "c", 8)
    db.commit()

    weights = compute_near_uniform_weights(db)

    median_n = 10  # sorted counts [8, 10, 300] -> median 10
    assert weights["a"] == pytest.approx(median_n / 300)
    assert weights["b"] == pytest.approx(median_n / 10)
    assert weights["c"] == pytest.approx(median_n / 8)
    assert weights["a"] < weights["c"]  # the oversampled slot ends up with the smallest weight


def test_near_uniform_every_slot_weighted_deck_count_equal(db) -> None:
    """The whole point of the formula: after weighting, every slot's *total* contribution
    (n_s * weight_s) converges on the same value (the median), regardless of raw deck count."""
    _insert_decks(db, "a", 300)
    _insert_decks(db, "b", 10)
    _insert_decks(db, "c", 8)
    db.commit()

    weights = compute_near_uniform_weights(db)

    totals = {slot: weights[slot] * n for slot, n in [("a", 300), ("b", 10), ("c", 8)]}
    assert totals["a"] == pytest.approx(totals["b"]) == pytest.approx(totals["c"])


def test_near_uniform_slot_at_median_gets_weight_one(db) -> None:
    _insert_decks(db, "a", 5)
    _insert_decks(db, "b", 9)
    _insert_decks(db, "c", 20)
    db.commit()

    weights = compute_near_uniform_weights(db)

    assert weights["b"] == pytest.approx(1.0)
