import json

import pandas as pd
import pytest

from edhcut.analysis.playrate import (
    ALL_COLOR_IDENTITIES,
    build_and_save,
    build_card_play_rates,
    build_color_identity_deck_counts,
    build_deck_color_identities,
    canonical_identity,
    eligible_deck_count,
    load_card_play_rates,
    load_color_identity_deck_counts,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, *, name=None, color_identity=(), legal_commander=1) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cards (oracle_id, name, color_identity, legal_commander) VALUES (?, ?, ?, ?)",
        (oracle_id, name or oracle_id, json.dumps(list(color_identity)), legal_commander),
    )


def _insert_deck(conn, deck_id, *, commander_oracle_id, partner_oracle_id=None, slot_key=None, cards=()) -> None:
    conn.execute(
        "INSERT INTO decks (deck_id, source, source_id, commander_oracle_id, partner_oracle_id, slot_key) "
        "VALUES (?, 'archidekt', ?, ?, ?, ?)",
        (deck_id, f"d{deck_id}", commander_oracle_id, partner_oracle_id, slot_key),
    )
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
        [(deck_id, oid) for oid in cards],
    )
    conn.commit()


# --- canonical_identity ------------------------------------------------------------------------

def test_canonical_identity_orders_wubrg() -> None:
    assert canonical_identity({"G", "U"}) == "UG"
    assert canonical_identity(["r", "w"]) == "WR"
    assert canonical_identity([]) == ""
    assert canonical_identity(["G", "U", "B", "R", "W"]) == "WUBRG"


def test_all_color_identities_has_32_unique_entries() -> None:
    assert len(ALL_COLOR_IDENTITIES) == 32
    assert len(set(ALL_COLOR_IDENTITIES)) == 32
    assert "" in ALL_COLOR_IDENTITIES
    assert "WUBRG" in ALL_COLOR_IDENTITIES


# --- build_deck_color_identities -----------------------------------------------------------

def test_deck_color_identity_single_commander(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko")
    df = build_deck_color_identities(db)
    assert df.set_index("deck_id").loc[1, "color_identity"] == "R"


def test_deck_color_identity_partners_union(db) -> None:
    _insert_card(db, "yoshimaru", color_identity=["W"])
    _insert_card(db, "bruse_tarl", color_identity=["R", "W"])
    _insert_deck(db, 1, commander_oracle_id="yoshimaru", partner_oracle_id="bruse_tarl")
    df = build_deck_color_identities(db)
    assert df.set_index("deck_id").loc[1, "color_identity"] == "WR"


def test_deck_color_identity_colorless_commander(db) -> None:
    _insert_card(db, "kozilek", color_identity=[])
    _insert_deck(db, 1, commander_oracle_id="kozilek")
    df = build_deck_color_identities(db)
    assert df.set_index("deck_id").loc[1, "color_identity"] == ""


# --- build_color_identity_deck_counts -------------------------------------------------------

def test_color_identity_deck_counts_is_full_32_row_zero_filled(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko")
    df = build_color_identity_deck_counts(db)
    assert len(df) == 32
    assert set(df["color_identity"]) == set(ALL_COLOR_IDENTITIES)
    assert df.set_index("color_identity").loc["R", "deck_count"] == 1
    assert df.set_index("color_identity").loc["WUBRG", "deck_count"] == 0


def test_color_identity_deck_counts_groups_matching_identities(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "krenko2", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko")
    _insert_deck(db, 2, commander_oracle_id="krenko2")
    df = build_color_identity_deck_counts(db)
    assert df.set_index("color_identity").loc["R", "deck_count"] == 2


def test_deck_color_identities_slot_key_scopes_to_one_slot(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "kyler", color_identity=["G", "W"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="s-krenko")
    _insert_deck(db, 2, commander_oracle_id="krenko", slot_key="s-krenko")
    _insert_deck(db, 3, commander_oracle_id="kyler", slot_key="s-kyler")

    df = build_deck_color_identities(db, slot_key="s-krenko")

    assert set(df["deck_id"]) == {1, 2}
    assert set(df["color_identity"]) == {"R"}


def test_color_identity_deck_counts_slot_key_scopes_to_one_slot(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "kyler", color_identity=["G", "W"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="s-krenko")
    _insert_deck(db, 2, commander_oracle_id="kyler", slot_key="s-kyler")

    df = build_color_identity_deck_counts(db, slot_key="s-krenko")

    assert df.set_index("color_identity").loc["R", "deck_count"] == 1
    assert df.set_index("color_identity").loc["WG", "deck_count"] == 0  # excluded, different slot


def test_color_identity_deck_counts_no_slot_key_covers_every_deck(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "kyler", color_identity=["G", "W"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="s-krenko")
    _insert_deck(db, 2, commander_oracle_id="kyler", slot_key="s-kyler")

    df = build_color_identity_deck_counts(db)

    assert df.set_index("color_identity").loc["R", "deck_count"] == 1
    assert df.set_index("color_identity").loc["WG", "deck_count"] == 1


def test_deck_color_identities_carries_each_decks_own_slot_key(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="s-krenko")
    df = build_deck_color_identities(db)
    assert df.set_index("deck_id").loc[1, "slot_key"] == "s-krenko"


# --- build_color_identity_deck_counts: deck_slot_weights ----------------------------------------

def test_color_identity_deck_counts_no_deck_slot_weights_matches_default(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="s-krenko")
    unweighted = build_color_identity_deck_counts(db)
    explicit_none = build_color_identity_deck_counts(db, deck_slot_weights=None)
    pd.testing.assert_frame_equal(unweighted, explicit_none)


def test_color_identity_deck_counts_deck_slot_weights_scales_each_decks_contribution(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="heavy")
    _insert_deck(db, 2, commander_oracle_id="krenko", slot_key="light")

    df = build_color_identity_deck_counts(db, deck_slot_weights={"heavy": 0.25, "light": 3.0})

    assert df.set_index("color_identity").loc["R", "deck_count"] == pytest.approx(0.25 + 3.0)


def test_color_identity_deck_counts_deck_slot_weights_missing_slot_defaults_to_one(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="known")
    _insert_deck(db, 2, commander_oracle_id="krenko", slot_key="unmapped")

    df = build_color_identity_deck_counts(db, deck_slot_weights={"known": 0.5})

    assert df.set_index("color_identity").loc["R", "deck_count"] == pytest.approx(0.5 + 1.0)


def test_color_identity_deck_counts_deck_slot_weights_uniform_one_matches_unweighted_values(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "kyler", color_identity=["G", "W"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="s-krenko")
    _insert_deck(db, 2, commander_oracle_id="kyler", slot_key="s-kyler")

    unweighted = build_color_identity_deck_counts(db)
    weighted = build_color_identity_deck_counts(db, deck_slot_weights={"s-krenko": 1.0, "s-kyler": 1.0})

    pd.testing.assert_series_equal(
        unweighted["deck_count"].astype(float), weighted["deck_count"], check_names=False
    )


def test_color_identity_deck_counts_matches_build_cooccurrence_total_weight(db) -> None:
    """Integration check for the units bug the plan flags: summing the weighted 32-row table
    must land on exactly the same total a `deck_slot_weights`-weighted `build_cooccurrence` call
    computes for its own null-model `N`, over the same (global) deck population."""
    from edhcut.analysis.cooccurrence import build_card_index, build_cooccurrence

    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "kyler", color_identity=["G", "W"])
    _insert_card(db, "goblin1", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko", slot_key="s-krenko", cards=["goblin1"])
    _insert_deck(db, 2, commander_oracle_id="krenko", slot_key="s-krenko", cards=["goblin1"])
    _insert_deck(db, 3, commander_oracle_id="kyler", slot_key="s-kyler", cards=["goblin1"])

    weights = {"s-krenko": 0.5, "s-kyler": 4.0}
    card_index = build_card_index(db, min_decks=1)
    cooc_result = build_cooccurrence(db, card_index, slot_key=None, deck_slot_weights=weights)

    counts = build_color_identity_deck_counts(db, deck_slot_weights=weights)
    assert counts["deck_count"].sum() == pytest.approx(cooc_result.total_weight)


# --- eligible_deck_count ---------------------------------------------------------------------

def test_eligible_deck_count_sums_supersets_only() -> None:
    counts = pd.DataFrame(
        {
            "color_identity": ["", "R", "W", "WR", "WUBRG"],
            "deck_count": [10, 20, 5, 7, 1],
        }
    )
    # A mono-red card is eligible in R decks and WR decks (both are supersets of {R}), plus
    # any other identity containing R -- but not in "", "W", which lack R.
    assert eligible_deck_count(counts, "R") == 20 + 7 + 1
    assert eligible_deck_count(counts, "") == 10 + 20 + 5 + 7 + 1
    assert isinstance(eligible_deck_count(counts, "R"), int)


def test_eligible_deck_count_zero_when_no_deck_has_a_superset() -> None:
    counts = pd.DataFrame({"color_identity": ["", "R"], "deck_count": [10, 20]})
    assert eligible_deck_count(counts, "G") == 0


def test_eligible_deck_count_returns_float_for_weighted_deck_counts() -> None:
    counts = pd.DataFrame({"color_identity": ["", "R"], "deck_count": [10.5, 20.25]})
    total = eligible_deck_count(counts, "")
    assert isinstance(total, float)
    assert total == pytest.approx(30.75)


# --- build_card_play_rates -------------------------------------------------------------------

def test_card_play_rate_basic(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "goblin1", color_identity=["R"])
    _insert_card(db, "goblin2", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko", cards=["goblin1"])
    _insert_deck(db, 2, commander_oracle_id="krenko", cards=["goblin1"])
    _insert_deck(db, 3, commander_oracle_id="krenko", cards=[])

    table = build_card_play_rates(db).set_index("oracle_id")
    # goblin1 played in 2 of 3 R-identity decks (the only decks eligible for a mono-R card).
    assert table.loc["goblin1", "deck_count"] == 2
    assert table.loc["goblin1", "eligible_deck_count"] == 3
    assert table.loc["goblin1", "play_rate"] == pytest.approx(2 / 3)
    # goblin2 never played, but still has a well-defined (zero) rate.
    assert table.loc["goblin2", "deck_count"] == 0
    assert table.loc["goblin2", "play_rate"] == 0.0


def test_card_play_rate_excludes_non_commander_legal_cards(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "banned_card", color_identity=["R"], legal_commander=0)
    _insert_deck(db, 1, commander_oracle_id="krenko")
    table = build_card_play_rates(db)
    assert "banned_card" not in set(table["oracle_id"])


def test_card_play_rate_na_when_color_identity_has_no_eligible_deck(db) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "green_card", color_identity=["G"])
    _insert_deck(db, 1, commander_oracle_id="krenko")

    table = build_card_play_rates(db).set_index("oracle_id")
    assert table.loc["green_card", "eligible_deck_count"] == 0
    assert pd.isna(table.loc["green_card", "play_rate"])


def test_card_play_rate_multicolor_card_eligible_only_in_superset_decks(db) -> None:
    _insert_card(db, "yoshimaru", color_identity=["W"])
    _insert_card(db, "bruse_tarl", color_identity=["R", "W"])
    _insert_card(db, "mono_white_cmdr", color_identity=["W"])
    _insert_card(db, "boros_card", color_identity=["W", "R"])

    _insert_deck(db, 1, commander_oracle_id="yoshimaru", partner_oracle_id="bruse_tarl", cards=["boros_card"])
    _insert_deck(db, 2, commander_oracle_id="mono_white_cmdr")

    table = build_card_play_rates(db).set_index("oracle_id")
    # boros_card (WR) is only eligible in the WR partner deck, not the mono-white one.
    assert table.loc["boros_card", "eligible_deck_count"] == 1
    assert table.loc["boros_card", "deck_count"] == 1
    assert table.loc["boros_card", "play_rate"] == 1.0


# --- build_and_save / load round trip ---------------------------------------------------------

def test_build_and_save_round_trip(db, tmp_path) -> None:
    _insert_card(db, "krenko", color_identity=["R"])
    _insert_card(db, "goblin1", color_identity=["R"])
    _insert_deck(db, 1, commander_oracle_id="krenko", cards=["goblin1"])

    out_dir = tmp_path / "kb"
    stats = build_and_save(db, out_dir=out_dir)
    assert stats["n_color_identities"] == 32
    assert stats["n_decks"] == 1
    assert stats["n_cards"] == 2

    counts = load_color_identity_deck_counts(out_dir)
    assert len(counts) == 32

    rates = load_card_play_rates(out_dir)
    assert set(rates["oracle_id"]) == {"krenko", "goblin1"}
