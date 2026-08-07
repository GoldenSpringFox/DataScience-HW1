import math

import numpy as np
import pytest

from edhcut.analysis.cooccurrence import (
    PRECON_CUT_RATE_FULL_WEIGHT,
    CooccurrenceResult,
    _slot_label,
    build_card_index,
    build_cooccurrence,
    compute_lift,
    compute_pmi,
    compute_tscore,
    precon_card_weight,
    top_associated,
)
from edhcut.db import connect
from edhcut.ingest.archidekt import slot_key_for


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


def _insert_deck(
    conn, deck_id, *, slot_key, cards, source_id=None, commander_oracle_id=None, partner_oracle_id=None
):
    _ensure_cards(conn, *cards)
    conn.execute(
        "INSERT INTO decks (deck_id, source, source_id, slot_key, commander_oracle_id, partner_oracle_id) "
        "VALUES (?, 'archidekt', ?, ?, ?, ?)",
        (deck_id, source_id or f"d{deck_id}", slot_key, commander_oracle_id, partner_oracle_id),
    )
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
        [(deck_id, oid) for oid in cards],
    )
    conn.commit()


def _insert_precon(conn, precon_id, *, commander_oracle_id, cards: dict[str, int]) -> None:
    _ensure_cards(conn, commander_oracle_id, *cards)
    conn.execute(
        "INSERT INTO precons (precon_id, set_name, deck_name, commander_oracle_id) "
        "VALUES (?, 'Test Set', 'Test Precon', ?)",
        (precon_id, commander_oracle_id),
    )
    conn.executemany(
        "INSERT INTO precon_cards (precon_id, oracle_id, qty) VALUES (?, ?, ?)",
        [(precon_id, oid, qty) for oid, qty in cards.items()],
    )
    conn.commit()


def _insert_retention(
    conn, precon_id, commander_key, oracle_id, *, weighted_cut, weighted_kept
) -> None:
    conn.execute(
        "INSERT INTO precon_card_retention "
        "(precon_id, commander_key, oracle_id, similar_deck_count, kept_count, weighted_cut, weighted_kept) "
        "VALUES (?, ?, ?, 0, 0, ?, ?)",
        (precon_id, commander_key, oracle_id, weighted_cut, weighted_kept),
    )
    conn.commit()


# --- precon_card_weight ----------------------------------------------------------------------

def test_precon_card_weight_full_at_or_below_threshold() -> None:
    assert precon_card_weight(0.0) == 1.0
    assert precon_card_weight(0.02) == 1.0
    assert precon_card_weight(PRECON_CUT_RATE_FULL_WEIGHT) == 1.0


def test_precon_card_weight_zero_at_always_cut() -> None:
    assert precon_card_weight(1.0) == 0.0


def test_precon_card_weight_decays_linearly_above_threshold() -> None:
    midpoint = PRECON_CUT_RATE_FULL_WEIGHT + (1.0 - PRECON_CUT_RATE_FULL_WEIGHT) / 2
    assert precon_card_weight(midpoint) == pytest.approx(0.5)


def test_precon_card_weight_monotonically_decreasing_above_threshold() -> None:
    values = [precon_card_weight(r) for r in (0.4, 0.6, 0.8, 0.9, 1.0)]
    assert values == sorted(values, reverse=True)


# Note: deck_precon_trust() itself now lives in edhcut.ingest.precon_retention (shared with
# that module's own weighted_cut/weighted_kept computation) -- its pure-function unit tests
# live in tests/test_precon_retention.py. The tests below exercise it indirectly through
# build_cooccurrence().


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


def test_build_cooccurrence_weighs_down_only_frequently_cut_precon_cards(db) -> None:
    # "kept_card" is almost always kept (cut_rate 0.03, below the 0.05 full-weight floor) ->
    # weight 1.0. "cut_card" is usually cut (cut_rate 0.8) -> some real discount, computed via
    # precon_card_weight() directly rather than a hand-simplified fraction. "player_added" isn't
    # part of the precon's own card list at all -> always weight 1.0, regardless of how often
    # precon cards get cut in this same deck. Deck has 3 cards, 2 of which (kept/cut) are the
    # precon's own -> diff=1 (only "player_added" doesn't match) -> well under
    # PRECON_DIFF_FULL_TRUST_CEILING, trust=1.0, so the raw precon_card_weight applies undiluted.
    commander_key = slot_key_for(["cmdr"])
    cut_card_weight = precon_card_weight(0.8)
    _insert_precon(db, 1, commander_oracle_id="cmdr", cards={"kept_card": 1, "cut_card": 1})
    _insert_retention(db, 1, commander_key, "kept_card", weighted_cut=0.03, weighted_kept=0.97)
    _insert_retention(db, 1, commander_key, "cut_card", weighted_cut=0.8, weighted_kept=0.2)
    _insert_deck(
        db, 1, slot_key="s", cards=["kept_card", "cut_card", "player_added"],
        commander_oracle_id="cmdr",
    )
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))

    assert result.weighted_marginal[row["kept_card"]] == pytest.approx(1.0)
    assert result.weighted_marginal[row["cut_card"]] == pytest.approx(cut_card_weight)
    assert result.weighted_marginal[row["player_added"]] == pytest.approx(1.0)

    kept_cut_pair = tuple(sorted((row["kept_card"], row["cut_card"])))
    kept_added_pair = tuple(sorted((row["kept_card"], row["player_added"])))
    assert result.pair_weighted[kept_cut_pair] == pytest.approx(1.0 * cut_card_weight)
    assert result.pair_weighted[kept_added_pair] == pytest.approx(1.0)
    assert result.pair_raw[kept_cut_pair] == 1  # raw count is unaffected by weighting
    # total_weight is just the deck count now -- down-weighting lives entirely in individual
    # cards' contributions, not in shrinking the effective sample size itself.
    assert result.total_weight == pytest.approx(1.0)


def test_build_cooccurrence_low_overlap_deck_gets_full_weight_despite_high_cut_rate(db) -> None:
    # Mirrors the live example (a Kyler deck sharing only 12/77 cards with its matched precon).
    # A precon with 10 tracked cards; "cut_card" has cut_rate 0.8. Two decks, same commander:
    #   Deck A: exactly those 10 tracked cards, nothing else -> deck_total=10, overlap=10,
    #     diff=0 -> well under PRECON_DIFF_FULL_TRUST_CEILING, trust=1.0, raw weight applies.
    #   Deck B: the same 10 tracked cards *plus* 20 player-added cards not in the precon at all
    #     -> deck_total=30, overlap=10, diff=20. floor_diff = 30-20=10, and diff(20) >= 10 ->
    #     trust=0, so cut_card reverts to full weight despite the *same* aggregate cut rate.
    commander_key = slot_key_for(["cmdr"])
    fillers = {f"filler{i}": 1 for i in range(8)}
    tracked_cards = {"kept_card": 1, "cut_card": 1, **fillers}
    cut_card_weight = precon_card_weight(0.8)
    _insert_precon(db, 1, commander_oracle_id="cmdr", cards=tracked_cards)
    _insert_retention(db, 1, commander_key, "kept_card", weighted_cut=0.03, weighted_kept=0.97)
    _insert_retention(db, 1, commander_key, "cut_card", weighted_cut=0.8, weighted_kept=0.2)
    for filler in fillers:
        _insert_retention(db, 1, commander_key, filler, weighted_cut=0.02, weighted_kept=0.98)

    # Deck A (its own slot, so its weighting can be checked in isolation): exactly the 10
    # tracked cards, diff=0 -> trust 1.0, cut_card gets the raw weight applied undiluted.
    _insert_deck(
        db, 1, slot_key="full_trust", cards=list(tracked_cards),
        commander_oracle_id="cmdr",
    )
    # Deck B (separate slot): the same 10 tracked cards plus 20 cards not in the precon at all
    # -> diff=20, deck_total=30 -> trust 0, cut_card reverts to full weight.
    _insert_deck(
        db, 2, slot_key="low_trust",
        cards=[*tracked_cards, *(f"extra{i}" for i in range(20))],
        commander_oracle_id="cmdr", source_id="deck2",
    )
    index = build_card_index(db, min_decks=1)
    row = dict(zip(index["oracle_id"], index["row"]))

    full_trust_result = build_cooccurrence(db, index, slot_key="full_trust")
    low_trust_result = build_cooccurrence(db, index, slot_key="low_trust")

    assert full_trust_result.weighted_marginal[row["cut_card"]] == pytest.approx(cut_card_weight)
    assert low_trust_result.weighted_marginal[row["cut_card"]] == pytest.approx(1.0)


def test_build_cooccurrence_full_weight_when_no_precon_match(db) -> None:
    # A deck with a commander that doesn't ship in any precon at all -- best_matching_precon
    # returns None, so every card (even ones that happen to share a name with some other
    # precon's cards) stays at full weight.
    _insert_deck(db, 1, slot_key="s", cards=["a", "b"], commander_oracle_id="no-such-precon-cmdr")
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))
    pair = tuple(sorted((row["a"], row["b"])))
    assert result.pair_weighted[pair] == pytest.approx(1.0)


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


# --- compute_tscore -------------------------------------------------------------------------

def test_tscore_masks_pairs_below_min_pair_count(db) -> None:
    _insert_deck(db, 1, slot_key="s", cards=["a", "b"])
    _insert_deck(db, 2, slot_key="s", cards=["a", "b"])
    # A third, unrelated deck so a/b aren't literally in 100% of the whole scope -- if they
    # were, observed would exactly equal expected-under-independence and t would be 0 even
    # unmasked, which would make this fixture unable to tell "masked" apart from "just 0".
    _insert_deck(db, 3, slot_key="s", cards=["filler"])
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))
    pair = tuple(sorted((row["a"], row["b"])))

    tscore = compute_tscore(result, min_pair_count=3)
    assert tscore[pair] == 0.0  # only seen twice, masked out at min_pair_count=3

    tscore_lenient = compute_tscore(result, min_pair_count=2)
    assert tscore_lenient[pair] != 0.0


def test_tscore_positive_for_positively_associated_pair(db) -> None:
    for i in range(5):
        _insert_deck(db, i, slot_key="s", cards=["a", "b"])
    for i in range(5, 10):
        _insert_deck(db, i, slot_key="s", cards=["z"])  # unrelated decks so P(a), P(b) < 1
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)
    row = dict(zip(index["oracle_id"], index["row"]))

    tscore = compute_tscore(result, min_pair_count=3)
    assert tscore[tuple(sorted((row["a"], row["b"])))] > 0.0


def test_tscore_matrix_is_symmetric(db) -> None:
    for i in range(4):
        _insert_deck(db, i, slot_key="s", cards=["a", "b", "c"])
    index = build_card_index(db, min_decks=1)
    result = build_cooccurrence(db, index, slot_key=None)

    tscore = compute_tscore(result, min_pair_count=1).toarray()
    assert np.allclose(tscore, tscore.T)


def test_tscore_favors_well_supported_pair_over_low_count_coincidence_with_no_discount_needed() -> None:
    # Same hand-constructed scope as test_pmi_discount_promotes_well_supported_pair_over_low_count_tie
    # (the live Purphoros, God of the Forge case from docs/devlog/6.1-cooccurrence.md): card 1 is
    # a rare "ceiling" coincidence (marginal 3, joint 3 -- co-occurs with the hub in literally
    # 100% of its own appearances, easy by chance at that sample size); card 2 is a much
    # better-supported, if individually smaller-ratio, association (marginal 25, joint 20).
    # Unlike undiscounted PMI, t-score ranks the well-supported pair above the low-count
    # coincidence with no separate discount factor -- sqrt(joint) alone does it.
    result = CooccurrenceResult(
        n_cards=3,
        pair_raw={(0, 1): 3, (0, 2): 20},
        pair_weighted={(0, 1): 3.0, (0, 2): 20.0},
        raw_marginal=np.array([100, 3, 25]),
        weighted_marginal=np.array([100.0, 3.0, 25.0]),
        total_weight=200.0,
        deck_count=200,
    )
    tscore = compute_tscore(result, min_pair_count=1)
    assert tscore[(0, 2)] > tscore[(0, 1)]


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
