import json

import pytest

from edhcut.db import connect
from edhcut.ingest.precon_retention import (
    DIFF_THRESHOLD,
    PRECON_DIFF_FULL_TRUST_CEILING,
    PRECON_OVERLAP_FLOOR,
    backfill_precon_card_retention,
    best_matching_precon,
    card_difference_count,
    cut_confidence,
    deck_precon_trust,
    multiset_overlap,
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


def _insert_precon(conn, precon_id, *, commander, partner=None, alternatives=None, cards):
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


# --- deck_precon_trust (pure) ----------------------------------------------------------------

def test_deck_precon_trust_full_at_or_below_diff_ceiling() -> None:
    deck_total = 99
    assert deck_precon_trust(0, deck_total) == 1.0
    assert deck_precon_trust(PRECON_DIFF_FULL_TRUST_CEILING, deck_total) == 1.0


def test_deck_precon_trust_zero_at_or_below_overlap_floor() -> None:
    deck_total = 99
    floor_diff = deck_total - PRECON_OVERLAP_FLOOR  # only PRECON_OVERLAP_FLOOR cards shared
    assert deck_precon_trust(floor_diff, deck_total) == 0.0
    assert deck_precon_trust(deck_total, deck_total) == 0.0  # 0 shared cards at all


def test_deck_precon_trust_linear_between_ceiling_and_floor() -> None:
    deck_total = 99
    floor_diff = deck_total - PRECON_OVERLAP_FLOOR
    midpoint = PRECON_DIFF_FULL_TRUST_CEILING + (floor_diff - PRECON_DIFF_FULL_TRUST_CEILING) / 2
    assert deck_precon_trust(midpoint, deck_total) == pytest.approx(0.5)


def test_deck_precon_trust_monotonically_decreasing() -> None:
    deck_total = 99
    values = [deck_precon_trust(d, deck_total) for d in (0, 10, 30, 50, 79, 99)]
    assert values == sorted(values, reverse=True)


# --- multiset_overlap / card_difference_count (pure) ----------------------------------------

# --- cut_confidence (pure) -------------------------------------------------------------------

def test_cut_confidence_midpoint_is_exactly_half() -> None:
    assert cut_confidence(20) == pytest.approx(0.5)


def test_cut_confidence_near_one_for_barely_touched_deck() -> None:
    assert cut_confidence(1) == pytest.approx(0.87, abs=0.01)


def test_cut_confidence_near_zero_for_heavily_rebuilt_deck() -> None:
    assert cut_confidence(50) == pytest.approx(0.047, abs=0.005)


def test_cut_confidence_monotonically_decreasing() -> None:
    values = [cut_confidence(d) for d in (0, 1, 5, 10, 20, 30, 50, 90)]
    assert values == sorted(values, reverse=True)


def test_cut_confidence_and_complement_sum_to_one() -> None:
    for diff in (0, 5, 20, 50, 99):
        kept_weight = 1.0 - cut_confidence(diff)
        assert cut_confidence(diff) + kept_weight == pytest.approx(1.0)


def test_multiset_overlap_takes_min_per_card() -> None:
    assert multiset_overlap({"forest": 15, "sol-ring": 1}, {"forest": 8, "sol-ring": 1}) == 9


def test_multiset_overlap_ignores_cards_only_on_one_side() -> None:
    assert multiset_overlap({"a": 3}, {"b": 5}) == 0


def test_card_difference_count_zero_for_exact_copy() -> None:
    cards = {"forest": 15, "sol-ring": 1, "a": 1}
    assert card_difference_count(cards, cards) == 0


def test_card_difference_count_counts_unmatched_deck_cards() -> None:
    deck = {"forest": 10, "new-card-1": 1, "new-card-2": 1}
    precon = {"forest": 15, "old-card": 1}
    # overlap = min(10,15) + 0 + 0 = 10; deck total = 12 -> difference = 2
    assert card_difference_count(deck, precon) == 2


def test_card_difference_count_basic_land_stacking_does_not_inflate_difference() -> None:
    # A precon with 20 stacked Forests (1 distinct row) vs a deck running only 5 Forests plus
    # otherwise-identical cards should read as "15 different," not blow up just because
    # Forest is only "1 distinct card" on the precon side.
    deck = {"forest": 5, "sol-ring": 1}
    precon = {"forest": 20, "sol-ring": 1}
    assert card_difference_count(deck, precon) == 0  # deck's 5+1=6 cards are all matched


# --- best_matching_precon --------------------------------------------------------------------

def test_best_matching_precon_returns_none_without_any_match(db) -> None:
    deck_cards = {"a": 1}
    assert best_matching_precon(db, deck_cards, ["nobody-uid"]) is None


def test_best_matching_precon_picks_lowest_difference(db) -> None:
    _insert_precon(db, 1, commander="krenko-uid", cards={"krenko-uid": 1, "a": 1, "b": 1})
    _insert_precon(
        db, 2, commander="other-uid", alternatives=["krenko-uid"],
        cards={"other-uid": 1, "krenko-uid": 1, "a": 1, "b": 1, "c": 1},
    )
    deck_cards = {"a": 1, "b": 1, "c": 1}
    # vs precon 1 (excl. krenko): library {a,b} -> overlap 2, deck total 3, diff 1
    # vs precon 2 (excl. krenko): library {other-uid,a,b,c} -> overlap 3, deck total 3, diff 0
    result = best_matching_precon(db, deck_cards, ["krenko-uid"])
    assert result == (2, 0)


def test_best_matching_precon_excludes_own_commander_from_precon_side(db) -> None:
    _insert_precon(db, 1, commander="krenko-uid", cards={"krenko-uid": 1, "a": 1})
    deck_cards = {"a": 1}  # commander correctly absent from deck_cards
    result = best_matching_precon(db, deck_cards, ["krenko-uid"])
    assert result == (1, 0)  # krenko-uid excluded from precon side too -> exact match


# --- backfill_precon_card_retention ----------------------------------------------------------

def test_backfill_counts_kept_and_cut_within_similar_decks(db) -> None:
    _insert_precon(
        db, 1, commander="krenko-uid",
        cards={"krenko-uid": 1, "staple": 1, "filler": 1, **{f"pad{i}": 1 for i in range(18)}},
    )
    # Deck A: exact copy (diff 0) -> similar, keeps both staple and filler.
    _insert_deck(
        db, 100, source_id="d100", commander="krenko-uid",
        cards={"staple": 1, "filler": 1, **{f"pad{i}": 1 for i in range(18)}},
    )
    # Deck B: swapped filler out for something else (diff 1, still <= 20) -> similar, cut filler.
    _insert_deck(
        db, 101, source_id="d101", commander="krenko-uid",
        cards={"staple": 1, "new-card": 1, **{f"pad{i}": 1 for i in range(18)}},
    )
    stats = backfill_precon_card_retention(db, diff_threshold=DIFF_THRESHOLD)
    assert stats.decks_checked == 2
    assert stats.decks_similar == 2
    assert stats.groups == 1

    rows = {
        oracle_id: (similar_deck_count, kept_count)
        for oracle_id, similar_deck_count, kept_count in db.execute(
            "SELECT oracle_id, similar_deck_count, kept_count FROM precon_card_retention "
            "WHERE precon_id = 1 AND commander_key = 'krenko-uid'"
        ).fetchall()
    }
    assert rows["staple"] == (2, 2)  # kept in both similar decks
    assert rows["filler"] == (2, 1)  # kept in A, cut in B
    assert "krenko-uid" not in rows  # commander itself never tracked

    # weighted_cut for "filler": only deck B (diff 1) cut it -> weight = cut_confidence(1).
    # weighted_kept for "staple": both decks kept it, at their own diff's kept-weight each.
    weighted = {
        oracle_id: (weighted_cut, weighted_kept)
        for oracle_id, weighted_cut, weighted_kept in db.execute(
            "SELECT oracle_id, weighted_cut, weighted_kept FROM precon_card_retention "
            "WHERE precon_id = 1 AND commander_key = 'krenko-uid'"
        ).fetchall()
    }
    assert weighted["filler"][0] == pytest.approx(cut_confidence(1))
    assert weighted["staple"][1] == pytest.approx(
        (1.0 - cut_confidence(0)) + (1.0 - cut_confidence(1))
    )


def test_backfill_excludes_decks_beyond_diff_threshold_from_raw_counts(db) -> None:
    _insert_precon(
        db, 1, commander="krenko-uid",
        cards={"krenko-uid": 1, **{f"card{i}": 1 for i in range(30)}},
    )
    # A totally different deck (0 shared cards with the precon library, 30 of its own 30 cards
    # unmatched) -> not "similar" for the threshold-gated raw counts. It still matches the
    # commander, but deck_precon_trust(diff=30, deck_total=30) is 0 (floor_diff = 30-20 = 10,
    # and 30 >= 10) -- too little overlap to assume this deck is even precon-derived, so it
    # contributes nothing to weighted_cut/weighted_kept either, not just the raw counts.
    _insert_deck(
        db, 100, source_id="d100", commander="krenko-uid",
        cards={f"other{i}": 1 for i in range(30)},
    )
    stats = backfill_precon_card_retention(db, diff_threshold=DIFF_THRESHOLD)
    assert stats.decks_matched_any == 1
    assert stats.decks_similar == 0
    assert stats.groups == 1

    row = db.execute(
        "SELECT similar_deck_count, kept_count, weighted_cut, weighted_kept "
        "FROM precon_card_retention WHERE precon_id = 1 AND oracle_id = 'card0'"
    ).fetchone()
    similar_deck_count, kept_count, weighted_cut, weighted_kept = row
    assert (similar_deck_count, kept_count) == (0, 0)  # excluded from the raw/threshold view
    assert weighted_cut == pytest.approx(0.0)  # trust 0 -- no evidence either way
    assert weighted_kept == pytest.approx(0.0)


def test_backfill_scales_weighted_evidence_by_deck_precon_trust(db) -> None:
    # A deck with 50 total cards, 25 of which match the precon (staple + 24 pads) and 25 of
    # which don't (25 new player cards) -> overlap=25, deck_total=50, diff=25. Not similar
    # enough for full trust (floor_diff = 50-20 = 30, so trust is in the linear zone, not 0 or
    # 1) -- trust = 1 - (25-10)/(30-10) = 0.25. Its "kept" contribution for the shared "staple"
    # card should be exactly a quarter of what a diff-25 deck at full trust would contribute.
    _insert_precon(
        db, 1, commander="krenko-uid",
        cards={"krenko-uid": 1, "staple": 1, **{f"pad{i}": 1 for i in range(49)}},
    )
    _insert_deck(
        db, 100, source_id="d100", commander="krenko-uid",
        cards={
            "staple": 1, **{f"pad{i}": 1 for i in range(24)},  # 25 cards shared with the precon
            **{f"new{i}": 1 for i in range(25)},  # 25 cards not in the precon at all
        },
    )
    backfill_precon_card_retention(db, diff_threshold=DIFF_THRESHOLD)

    row = db.execute(
        "SELECT weighted_kept FROM precon_card_retention "
        "WHERE precon_id = 1 AND oracle_id = 'staple'"
    ).fetchone()
    trust = 0.25
    assert row[0] == pytest.approx(trust * (1.0 - cut_confidence(25)))


def test_backfill_scopes_by_commander_key_not_just_precon(db) -> None:
    # A card that's good to keep for one commander but not another must not be conflated --
    # two different commanders both matching the same precon get separate rows.
    _insert_precon(
        db, 1, commander="leinore-uid", alternatives=["kyler-uid"],
        cards={"leinore-uid": 1, "staple": 1, **{f"pad{i}": 1 for i in range(18)}},
    )
    _insert_deck(
        db, 100, source_id="d100", commander="kyler-uid",
        cards={"staple": 1, **{f"pad{i}": 1 for i in range(18)}},
    )
    _insert_deck(
        db, 101, source_id="d101", commander="leinore-uid",
        cards={f"pad{i}": 1 for i in range(18)},  # cut "staple" under Leinore
    )
    stats = backfill_precon_card_retention(db, diff_threshold=DIFF_THRESHOLD)
    assert stats.groups == 2

    kyler_row = db.execute(
        "SELECT kept_count, similar_deck_count FROM precon_card_retention "
        "WHERE precon_id = 1 AND commander_key = 'kyler-uid' AND oracle_id = 'staple'"
    ).fetchone()
    leinore_row = db.execute(
        "SELECT kept_count, similar_deck_count FROM precon_card_retention "
        "WHERE precon_id = 1 AND commander_key = 'leinore-uid' AND oracle_id = 'staple'"
    ).fetchone()
    assert kyler_row == (1, 1)
    assert leinore_row == (0, 1)


def test_backfill_contributes_to_only_the_single_best_matching_precon(db) -> None:
    _insert_precon(db, 1, commander="krenko-uid", cards={"krenko-uid": 1, "a": 1})
    _insert_precon(
        db, 2, commander="other-uid", alternatives=["krenko-uid"],
        cards={"other-uid": 1, "krenko-uid": 1, "a": 1, "b": 1},
    )
    _insert_deck(db, 100, source_id="d100", commander="krenko-uid", cards={"a": 1, "b": 1})
    # closer to precon 2 (diff 0) than precon 1 (diff 1) -- must only count toward precon 2.
    stats = backfill_precon_card_retention(db, diff_threshold=DIFF_THRESHOLD)
    assert stats.groups == 1
    assert db.execute(
        "SELECT COUNT(*) FROM precon_card_retention WHERE precon_id = 1"
    ).fetchone()[0] == 0
    assert db.execute(
        "SELECT COUNT(*) FROM precon_card_retention WHERE precon_id = 2"
    ).fetchone()[0] == 3  # other-uid, a, b (krenko-uid excluded as own commander)


def test_backfill_is_idempotent_and_clears_stale_rows(db) -> None:
    _insert_precon(db, 1, commander="krenko-uid", cards={"krenko-uid": 1, "a": 1})
    _insert_deck(db, 100, source_id="d100", commander="krenko-uid", cards={"a": 1})
    backfill_precon_card_retention(db)
    # Remove the deck entirely and re-run -- old rows must not linger.
    db.execute("DELETE FROM deck_cards WHERE deck_id = 100")
    db.execute("DELETE FROM decks WHERE deck_id = 100")
    db.commit()
    stats = backfill_precon_card_retention(db)
    assert stats.rows_written == 0
    assert db.execute("SELECT COUNT(*) FROM precon_card_retention").fetchone()[0] == 0
