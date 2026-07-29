import json

import pytest

from edhcut.db import connect
from edhcut.ingest.precon_similarity import (
    backfill_precon_similarity,
    compute_deck_precon_similarity,
    jaccard_similarity,
    matching_precon_ids,
)


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _ensure_cards(conn, *oracle_ids: str) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO cards (oracle_id, name) VALUES (?, ?)",
        [(oid, oid) for oid in oracle_ids if oid],
    )


def _insert_precon(
    conn, precon_id, *, commander, partner=None, alternatives=None, cards
):
    _ensure_cards(conn, commander, partner, *(alternatives or []), *cards)
    conn.execute(
        "INSERT INTO precons (precon_id, set_name, deck_name, commander_oracle_id, "
        "partner_oracle_id, alternative_commander_oracle_ids) VALUES (?, ?, ?, ?, ?, ?)",
        (precon_id, "Test Set", f"precon-{precon_id}", commander, partner,
         json.dumps(alternatives or [])),
    )
    conn.executemany(
        "INSERT INTO precon_cards (precon_id, oracle_id, qty) VALUES (?, ?, ?)",
        [(precon_id, oid, qty) for oid, qty in cards.items()],
    )
    conn.commit()


def _insert_deck(conn, deck_id, *, source_id, commander, partner=None, cards):
    _ensure_cards(conn, commander, partner, *cards)
    conn.execute(
        "INSERT INTO decks (deck_id, source, source_id, commander_oracle_id, "
        "partner_oracle_id, slot_key) VALUES (?, 'archidekt', ?, ?, ?, ?)",
        (deck_id, source_id, commander, partner, commander),
    )
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, ?)",
        [(deck_id, oid, qty) for oid, qty in cards.items()],
    )
    conn.commit()


def test_jaccard_similarity_basic_overlap() -> None:
    assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2 / 4)


def test_jaccard_similarity_identical_sets_is_one() -> None:
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_similarity_disjoint_sets_is_zero() -> None:
    assert jaccard_similarity({"a"}, {"b"}) == 0.0


def test_jaccard_similarity_both_empty_is_none_not_zero() -> None:
    assert jaccard_similarity(set(), set()) is None


def test_matching_precon_ids_finds_declared_commander(db) -> None:
    _insert_precon(db, 1, commander="leinore-uid", cards={"leinore-uid": 1, "sol-ring-uid": 1})
    assert matching_precon_ids(db, ["leinore-uid"]) == [1]


def test_matching_precon_ids_finds_declared_partner(db) -> None:
    _insert_precon(
        db, 1, commander="leinore-uid", partner="kyler-uid",
        cards={"leinore-uid": 1, "kyler-uid": 1},
    )
    assert matching_precon_ids(db, ["kyler-uid"]) == [1]


def test_matching_precon_ids_finds_alternative_commander(db) -> None:
    _insert_precon(
        db, 1, commander="leinore-uid", alternatives=["kyler-uid"],
        cards={"leinore-uid": 1, "kyler-uid": 1},
    )
    # Kyler never shipped as anyone's *declared* commander — only as an alternative.
    assert matching_precon_ids(db, ["kyler-uid"]) == [1]


def test_matching_precon_ids_no_match_returns_empty(db) -> None:
    _insert_precon(db, 1, commander="leinore-uid", cards={"leinore-uid": 1})
    assert matching_precon_ids(db, ["someone-else-uid"]) == []


def test_matching_precon_ids_matches_across_multiple_precons(db) -> None:
    _insert_precon(db, 1, commander="krenko-uid", cards={"krenko-uid": 1})
    _insert_precon(db, 2, commander="other-uid", alternatives=["krenko-uid"],
                    cards={"other-uid": 1, "krenko-uid": 1})
    assert matching_precon_ids(db, ["krenko-uid"]) == [1, 2]


def test_compute_deck_precon_similarity_excludes_deck_commander_from_both_sides(db) -> None:
    # Krenko is only an *alternative* in this precon (declared commander is someone else),
    # so Krenko itself is still a regular card in precon_cards. It must be subtracted out of
    # the precon side too, or a deck running Krenko as commander would be unfairly penalized
    # for "missing" a card it structurally can never have in its own library.
    _insert_precon(
        db, 1, commander="other-uid", alternatives=["krenko-uid"],
        cards={"other-uid": 1, "krenko-uid": 1, "sol-ring-uid": 1, "goblin-uid": 1},
    )
    _insert_deck(
        db, 100, source_id="d100", commander="krenko-uid",
        cards={"sol-ring-uid": 1, "goblin-uid": 1},
    )
    similarity = compute_deck_precon_similarity(db, 100, ["krenko-uid"])
    # Deck library {sol-ring, goblin} vs precon library minus krenko: {other-uid, sol-ring,
    # goblin} -> intersection 2, union 3.
    assert similarity == pytest.approx(2 / 3)


def test_compute_deck_precon_similarity_takes_best_match_across_precons(db) -> None:
    _insert_precon(db, 1, commander="krenko-uid", cards={"krenko-uid": 1, "a-uid": 1})
    _insert_precon(
        db, 2, commander="other-uid", alternatives=["krenko-uid"],
        cards={"other-uid": 1, "krenko-uid": 1, "a-uid": 1, "b-uid": 1},
    )
    _insert_deck(db, 100, source_id="d100", commander="krenko-uid", cards={"a-uid": 1, "b-uid": 1})
    # vs precon 1 library {a-uid}: intersection 1, union 2 -> 0.5
    # vs precon 2 library {other-uid, a-uid, b-uid}: intersection 2, union 3 -> 0.667
    similarity = compute_deck_precon_similarity(db, 100, ["krenko-uid"])
    assert similarity == pytest.approx(2 / 3)


def test_compute_deck_precon_similarity_returns_none_without_a_match(db) -> None:
    _insert_deck(db, 100, source_id="d100", commander="nobody-uid", cards={"a-uid": 1})
    assert compute_deck_precon_similarity(db, 100, ["nobody-uid"]) is None


def test_backfill_writes_similarity_and_leaves_unmatched_decks_null(db) -> None:
    _insert_precon(db, 1, commander="krenko-uid", cards={"krenko-uid": 1, "a-uid": 1})
    _insert_deck(db, 100, source_id="d100", commander="krenko-uid", cards={"a-uid": 1})
    _insert_deck(db, 101, source_id="d101", commander="nobody-uid", cards={"z-uid": 1})

    stats = backfill_precon_similarity(db)
    assert stats.decks_checked == 2
    assert stats.decks_matched == 1

    row1 = db.execute("SELECT precon_similarity FROM decks WHERE deck_id = 100").fetchone()
    row2 = db.execute("SELECT precon_similarity FROM decks WHERE deck_id = 101").fetchone()
    assert row1[0] == pytest.approx(1.0)  # identical libraries: {a-uid} vs {a-uid}
    assert row2[0] is None


def test_backfill_can_be_scoped_to_a_single_slot_key(db) -> None:
    _insert_precon(db, 1, commander="krenko-uid", cards={"krenko-uid": 1, "a-uid": 1})
    _insert_deck(db, 100, source_id="d100", commander="krenko-uid", cards={"a-uid": 1})
    # A different slot's deck, also matchable, but scoped out of this backfill call.
    _insert_deck(db, 200, source_id="d200", commander="krenko-uid", cards={"a-uid": 1})
    db.execute("UPDATE decks SET slot_key = 'other-slot' WHERE deck_id = 200")
    db.commit()

    stats = backfill_precon_similarity(db, slot_key="krenko-uid")
    assert stats.decks_checked == 1

    row200 = db.execute("SELECT precon_similarity FROM decks WHERE deck_id = 200").fetchone()
    assert row200[0] is None  # untouched by the scoped run


def test_backfill_recomputes_stale_similarity_back_to_null(db) -> None:
    # A deck previously matched (similarity set), but the precon it matched no longer exists
    # (e.g. dropped in a precons re-harvest) -> re-running must clear the stale value rather
    # than leaving old data behind.
    _insert_deck(db, 100, source_id="d100", commander="krenko-uid", cards={"a-uid": 1})
    db.execute("UPDATE decks SET precon_similarity = 0.75 WHERE deck_id = 100")
    db.commit()

    stats = backfill_precon_similarity(db)
    assert stats.decks_matched == 0
    row = db.execute("SELECT precon_similarity FROM decks WHERE deck_id = 100").fetchone()
    assert row[0] is None
