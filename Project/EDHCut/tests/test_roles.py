"""Functional role classification (`edhcut.analysis.roles`). Two halves: the rule table's own
invariants (every mapped role is declared, no zero-weight rules, the documented precedence order
holds) and the classifier's behaviour on real cards, including the held-out spot check."""

import pandas as pd
import pytest

from edhcut.analysis.roles import (
    BIG_BODY_SCORE,
    MIN_ROLE_SCORE,
    NON_SECONDARY_ROLES,
    ROLES,
    SECONDARY_MIN_SCORE,
    SPOT_CHECK,
    TAG_COMBOS,
    TAG_RULES,
    TEXT_RULES,
    assign_roles,
    build_and_save,
    classify,
    layer_agreement,
    load_card_tags,
    load_roles,
    normalize_oracle_text,
    role_distribution,
    score_tags,
    score_text,
    unknown_tags,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, *, name=None, type_line="Artifact", oracle_text="", tags=()):
    conn.execute(
        "INSERT OR REPLACE INTO cards (oracle_id, name, type_line, oracle_text) VALUES (?, ?, ?, ?)",
        (oracle_id, name or oracle_id, type_line, oracle_text),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO card_tags (oracle_id, tag, source) VALUES (?, ?, 'tagger_bulk')",
        [(oracle_id, tag) for tag in tags],
    )


# --------------------------------------------------------------------------------------
# The mapping itself
# --------------------------------------------------------------------------------------


def test_every_mapped_role_is_a_declared_role():
    assert set(TAG_RULES) <= set(ROLES)
    assert set(TEXT_RULES) <= set(ROLES)


def test_no_rule_has_zero_weight():
    """A zero weight is always a mistake -- it reads as a rule but contributes nothing."""
    for role, rules in TAG_RULES.items():
        assert all(weight != 0 for weight in rules.values()), role
    for role, rules in TEXT_RULES.items():
        assert all(weight != 0 for _, weight in rules), role


def test_other_is_the_only_default_and_is_never_scored():
    """`other` comes only from layer 3, so no rule may award it directly -- and it is the only
    such bucket (`synergy_piece` was dropped in the 2026-08-23 vocabulary revision)."""
    assert "other" not in TAG_RULES and "other" not in TEXT_RULES
    assert "synergy_piece" not in ROLES


def test_land_is_never_a_scored_role():
    """`land` is a type-line override (LAND_SCORE), not something tags can earn."""
    assert "land" not in TAG_RULES and "land" not in TEXT_RULES
    assert NON_SECONDARY_ROLES == frozenset({"land", "other"})


def test_board_presence_is_the_most_generic_role():
    """It sits last before the default, so anything more specific outscores it."""
    assert ROLES.index("board_presence") == len(ROLES) - 2
    assert ROLES[-1] == "other"


def test_declared_precedence_stax_boardwipe_graveyard_hate():
    """Set against Decree of Annihilation: it does all three, and should read in that order."""
    assert ROLES.index("stax") < ROLES.index("boardwipe") < ROLES.index("graveyard_hate")


def test_wincon_outranks_board_presence():
    """An evasion-granting lord beats a plain one on a tie, and those weights are set to tie."""
    assert ROLES.index("wincon") < ROLES.index("board_presence")


def test_spot_check_expectations_use_real_roles():
    for name, primary, secondary, _calibrated in SPOT_CHECK:
        assert primary in ROLES, name
        assert secondary is None or secondary in ROLES, name
        assert secondary not in NON_SECONDARY_ROLES, name


# --------------------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------------------


def test_normalize_replaces_card_name_and_strips_reminder_text():
    text = normalize_oracle_text(
        "Counterspell", "Counterspell has flying (It can't be blocked except by fliers.)"
    )
    assert "counterspell" not in text
    assert "~ has flying" in text
    assert "except by fliers" not in text


def test_normalize_replaces_each_face_of_a_multi_faced_name():
    """Scryfall writes each face's oracle text with that face's own short name, so replacing
    only the combined `A // B` string would leave the face names in the text."""
    text = normalize_oracle_text("Alpha // Beta", "Beta deals damage. Alpha draws a card.")
    assert "alpha" not in text and "beta" not in text
    assert text.count("~") == 2


def test_normalize_handles_missing_oracle_text():
    assert normalize_oracle_text("Mountain", None) == ""


# --------------------------------------------------------------------------------------
# Layer 1 -- tag scoring
# --------------------------------------------------------------------------------------


def test_anchor_tag_scores_its_role():
    assert score_tags(["sweeper"])["boardwipe"] == TAG_RULES["boardwipe"]["sweeper"]


def test_corroborators_add_to_the_anchor():
    anchor_only = score_tags(["sweeper"])["boardwipe"]
    assert score_tags(["sweeper", "sweeper-one-sided"])["boardwipe"] > anchor_only


def test_negative_weight_cancels_a_parent_tag():
    """Cultivate's tag set: the `tutor` parent is real, but fetching a basic land onto the
    battlefield is ramp, so `tutor` must not survive as a role."""
    scores = score_tags(
        ["ramp", "land-ramp", "tutor", "tutor-land", "tutor-land-basic",
         "tutor-land-to-battlefield", "tutor-to-battlefield"]
    )
    assert "tutor" not in scores
    assert scores["mana_acceleration"] > 0


def test_land_denial_and_mass_land_destruction_are_both_stax():
    """Decided 2026-08-23. Convenient, because Tagger uses one tag for both: `mass-land-denial`
    sits on Winter Orb and on Armageddon alike, and now needs no disambiguation."""
    denial = classify(name="Winter Orb", type_line="Artifact", oracle_text="",
                      tags=["mass-land-denial", "stasis"])
    destruction = classify(name="Armageddon", type_line="Sorcery",
                           oracle_text="Destroy all lands.",
                           tags=["mass-land-denial", "removal-land", "sweeper"])
    assert denial.primary == "stax"
    assert destruction.primary == "stax"
    assert destruction.secondary is None      # it wipes no board


def test_mass_land_destruction_is_kept_out_of_boardwipe():
    """Armageddon is tagged `sweeper` too, but wipes no board; Jokulhaups genuinely does both
    and keeps boardwipe as its secondary. `removal-creature` is what separates them."""
    armageddon = classify(name="Armageddon", type_line="Sorcery", oracle_text="Destroy all lands.",
                          tags=["mass-land-denial", "removal-land", "sweeper"])
    jokulhaups = classify(name="Jokulhaups", type_line="Sorcery",
                          oracle_text="Destroy all artifacts, creatures, and lands.",
                          tags=["mass-land-denial", "removal-land", "sweeper", "removal-creature"])
    assert armageddon.primary == "stax" and armageddon.secondary is None
    assert jokulhaups.primary == "stax" and jokulhaups.secondary == "boardwipe"


def test_land_destruction_with_nothing_given_back_is_stax():
    """Decision 2026-08-23: destroying a land and offering nothing attacks the opponent's ability
    to play, which is stax. Stone Rain and Wasteland."""
    stone_rain = score_tags(["removal-land", "removal-destroy", "spot-removal"])
    assert stone_rain["stax"] > stone_rain["spot_removal"]


def test_land_destruction_that_replaces_the_land_is_spot_removal():
    """Ghost Quarter and Demolition Field hand back a basic -- that is a trade, so removal.
    `swap-removal` is the tag that marks it."""
    ghost_quarter = score_tags(
        ["removal-land", "removal-destroy", "spot-removal", "swap-removal"])
    assert ghost_quarter["spot_removal"] > ghost_quarter.get("stax", 0)


def test_land_destruction_that_could_hit_a_nonland_is_spot_removal():
    """Acidic Slime lists land among several targets -- ordinary multi-mode removal."""
    acidic_slime = score_tags(
        ["removal-land", "removal-artifact", "removal-enchantment", "removal-destroy",
         "spot-removal"])
    assert acidic_slime["spot_removal"] > acidic_slime.get("stax", 0)


def test_tax_tag_alone_is_not_stax():
    """Tagger's `tax` also sits on Soul Warden and Roaming Throne -- "something happens when an
    opponent does something", not "a cost is imposed"."""
    assert score_tags(["tax"]).get("stax", 0) < MIN_ROLE_SCORE
    assert score_tags(["cost-increaser"])["stax"] >= MIN_ROLE_SCORE


def test_anthems_score_wincon():
    """Decided 2026-08-23: a static team pump is how a go-wide deck closes."""
    assert score_tags(["anthem"])["wincon"] >= MIN_ROLE_SCORE


def test_impulse_draw_is_an_anchor_not_a_corroborator():
    """Light Up the Stage is never tagged `draw` -- it never draws."""
    assert score_tags(["impulsive-draw"])["draw"] >= MIN_ROLE_SCORE


def test_unmapped_tags_score_nothing():
    assert score_tags(["typal-goblin", "alliteration"]) == {}


# --------------------------------------------------------------------------------------
# Layer 2 -- text scoring
# --------------------------------------------------------------------------------------


def test_land_fetch_reads_as_ramp_not_tutor_even_without_the_word_land():
    """Farseek names basic land *types*, never the word "land" -- the case that made the two
    rules disagree before `_LAND_TARGET` was shared between them."""
    scores = score_text(
        "Farseek", "Sorcery", "Search your library for a Plains, Island, Swamp, or Mountain card."
    )
    assert "mana_acceleration" in scores
    assert "tutor" not in scores


def test_non_land_search_reads_as_tutor():
    scores = score_text("Worldly Tutor", "Instant", "Search your library for a creature card.")
    assert "tutor" in scores
    assert "mana_acceleration" not in scores


def test_ramp_text_rules_are_skipped_on_lands():
    """A fetchland's own ability is mana base, not a ramp spell in a nonland slot."""
    text = "Sacrifice this land: Search your library for a Forest card and put it onto the battlefield."
    assert "mana_acceleration" not in score_text("Windswept Heath", "Land", text)
    assert "mana_acceleration" in score_text("Land Grant", "Sorcery", text)


def test_graveyard_hate_ignores_your_own_graveyard():
    """Exiling from *your* graveyard is fuel or a cost, not hate (Necropotence)."""
    own = score_text("Necro", "Enchantment", "Whenever you discard a card, exile that card from your graveyard.")
    theirs = score_text("Relic", "Artifact", "Exile all cards from each opponent's graveyard.")
    assert "graveyard_hate" not in own
    assert "graveyard_hate" in theirs


def test_removal_text_ignores_your_own_permanents_and_graveyard_cards():
    """Same principle as above: hitting your own resource is a cost, not removal."""
    own = score_text("Extraplanar Lens", "Artifact", "You may exile target land you control.")
    graveyard = score_text("Deathrite Shaman", "Creature", "Exile target land card from a graveyard.")
    theirs = score_text("Stone Rain", "Sorcery", "Destroy target land.")
    assert "spot_removal" not in own
    assert "spot_removal" not in graveyard
    assert "spot_removal" in theirs


def test_draw_rules_do_not_double_count_one_clause():
    """The second- and third-person draw patterns must be mutually exclusive."""
    second = score_text("X", "Instant", "Draw a card.")["draw"]
    third = score_text("Y", "Instant", "Each player draws a card.")["draw"]
    assert second == 3.0
    assert third == 1.0


def test_evasion_grant_scores_once():
    """`gains?` covers both "gains trample" and "creatures gain trample"; an earlier duplicate
    pattern scored the same clause twice and pushed Craterhoof Behemoth off `wincon`."""
    assert score_text("X", "Creature", "Creatures you control gain trample.")["evasion"] == 2.0
    assert score_text("Y", "Creature", "Target creature gains trample.")["evasion"] == 2.0


def test_goad_reads_as_defensive_from_text_alone():
    """There is no usable Tagger tag for goad, so this role depends on the text layer."""
    assert "defensive" in score_text("X", "Instant", "Goad each creature target opponent controls.")


# --------------------------------------------------------------------------------------
# Combining the layers
# --------------------------------------------------------------------------------------


def test_land_type_line_always_wins_primary():
    result = classify(
        name="Bojuka Bog",
        type_line="Land",
        oracle_text="When this land enters, exile target player's graveyard.",
        tags=["hate-graveyard"],
    )
    assert result.primary == "land"
    assert result.primary_source == "type_line"
    assert result.secondary == "graveyard_hate"


def test_layers_combine_by_max_not_sum():
    """One ability seen by both layers is one piece of evidence, not two."""
    result = classify(
        name="X", type_line="Instant", oracle_text="Draw a card.", tags=["draw"]
    )
    assert result.primary == "draw"
    assert result.primary_score == max(
        score_tags(["draw"])["draw"], score_text("X", "Instant", "Draw a card.")["draw"]
    )
    assert result.primary_source == "both"


def test_role_below_the_evidence_floor_is_not_assigned():
    """A lone 0.5 corroborator is not enough to name a role -- the bug that made every anthem
    effect a `wincon` before `anthem` itself became the anchor."""
    weak = classify(name="X", type_line="Enchantment", oracle_text="", tags=["removal-destroy"])
    assert weak.primary == "other"
    assert score_tags(["removal-destroy"])["spot_removal"] < MIN_ROLE_SCORE
    # `power-boost-to-all` is the tag this floor was originally added for: it made every anthem
    # effect a `wincon`. It now scores no `wincon` at all -- the role's own anchors do that work.
    assert "wincon" not in score_tags(["power-boost-to-all"])


def test_secondary_needs_more_than_a_single_incidental_signal():
    result = classify(
        name="Craterhoof",
        type_line="Creature",
        oracle_text="",
        tags=["overrun", "gives-trample"],
    )
    assert result.primary == "wincon"
    # Craterhoof DOES grant trample, and since 2026-08-23 the instruction is to be liberal with
    # secondaries -- one real signal is enough. What must not happen is a *sub*-threshold signal
    # becoming a secondary.
    assert result.secondary == "evasion"
    assert score_tags(["gives-flying"])["evasion"] < SECONDARY_MIN_SCORE


def test_ties_are_broken_deterministically_by_role_priority():
    """Whispersilk Cloak scores protection and evasion at exactly 4.0."""
    tags = ["protection", "protects-creature", "gives-shroud", "gives-evasion", "gives-unblockable"]
    result = classify(name="Whispersilk Cloak", type_line="Artifact", oracle_text="", tags=tags)
    assert result.primary == "protection"    # earlier in ROLES than evasion
    assert result.secondary == "evasion"     # the runner-up is still recorded


def test_default_is_other_even_when_functional_tags_exist():
    """No `synergy_piece` hedge: Krenko makes tokens, but token generation is not a role on
    this list, so the honest answer is `other`."""
    result = classify(
        name="Impact Tremors", type_line="Enchantment", oracle_text="",
        tags=["typal-goblin", "alliteration"],
    )
    assert result.primary == "other"
    assert result.primary_source == "default"
    assert result.secondary is None


def test_default_is_other_with_no_tags_at_all():
    assert classify(name="X", type_line="Enchantment", oracle_text="", tags=[]).primary == "other"


def test_a_plain_small_creature_is_not_board_presence():
    """Narrowed 2026-08-23: an earlier build gave every creature a floor score and the role
    swallowed 10,028 cards. A vanilla 2/2 is not what `board_presence` is for."""
    bear = classify(name="Grizzly Bears", type_line="Creature - Bear", oracle_text="",
                    tags=[], power="2", toughness="2")
    assert bear.primary == "other"


def test_a_big_body_is_board_presence():
    """Ghalta: the card's own size is the point, and no tag says so."""
    ghalta = classify(name="Ghalta", type_line="Legendary Creature - Dinosaur", oracle_text="",
                      tags=[], power="12", toughness="12")
    assert ghalta.primary == "board_presence"
    assert ghalta.primary_score == BIG_BODY_SCORE
    assert ghalta.primary_source == "type_line"


def test_non_numeric_power_does_not_count_as_a_big_body():
    """Power/toughness are raw Scryfall strings; `*` is a property of the board, not the card."""
    star = classify(name="X", type_line="Creature - Elemental", oracle_text="",
                    tags=[], power="*", toughness="*")
    assert star.primary == "other"


def test_token_generation_is_board_presence():
    krenko = classify(name="Krenko, Mob Boss", type_line="Legendary Creature - Goblin",
                      oracle_text="",
                      tags=["repeatable-creature-tokens", "repeatable-token-generator"],
                      power="3", toughness="3")
    assert krenko.primary == "board_presence"
    assert krenko.primary_score > BIG_BODY_SCORE


def test_treasure_tokens_are_not_board_presence():
    """A Treasure cannot block -- only `repeatable-creature-tokens` counts."""
    assert "board_presence" not in score_tags(["repeatable-artifact-tokens"])


def test_a_land_creature_is_still_a_land():
    """Dryad Arbor: the land override wins, and board_presence must not displace it."""
    arbor = classify(name="Dryad Arbor", type_line="Land Creature - Forest Dryad",
                     oracle_text="", tags=[], power="1", toughness="1")
    assert arbor.primary == "land"


# ------------------------------------------------------------------------------------
# Cross-layer penalties and tag conjunctions (both added 2026-08-23)
# ------------------------------------------------------------------------------------


def test_a_tag_penalty_also_cancels_the_text_layer():
    """Ponder is the case this exists for: its tag-layer `draw` is cancelled by `cantrip`, but the
    text layer reads a plain "Draw a card." and carried the role through until penalties were
    applied *after* the two layers were combined."""
    ponder = classify(name="Ponder", type_line="Sorcery",
                      oracle_text="Look at the top three cards of your library... Draw a card.",
                      tags=["cantrip", "hand-neutral", "draw", "pure-draw", "card-advantage"])
    assert ponder.primary != "draw"
    assert "draw" in score_text("Ponder", "Sorcery", "Draw a card.")     # the layer did see it


def test_loot_survives_the_hand_neutral_penalty():
    """Merfolk Looter is `hand-neutral` too, but loot IS draw by decision -- its own anchors are
    weighted to outlast the penalty."""
    looter = classify(name="Merfolk Looter", type_line="Creature - Merfolk Rogue",
                      oracle_text="Draw a card, then discard a card.",
                      tags=["draw", "loot", "repeatable-loot", "hand-neutral", "card-advantage"],
                      power="1", toughness="1")
    assert looter.primary == "draw"


def test_a_plain_lord_is_board_presence_but_an_evasive_one_is_a_wincon():
    """The conjunction TAG_COMBOS exists for. Doing this with plain weights would need a large
    `gives-evasion` weight on `wincon`, which would fire on every evasion granter in the pool."""
    plain = classify(name="Imperious Perfect", type_line="Creature - Elf Warrior", oracle_text="",
                     tags=["anthem", "power-boost-to-all", "repeatable-creature-tokens"],
                     power="2", toughness="2")
    evasive = classify(name="Lord of the Accursed", type_line="Creature - Zombie", oracle_text="",
                       tags=["anthem", "power-boost-to-all", "gives-evasion", "gives-menace"],
                       power="2", toughness="3")
    assert plain.primary == "board_presence"
    assert evasive.primary == "wincon"
    assert "wincon" not in score_tags(["gives-evasion"])       # evasion alone is not a wincon


def test_tag_combos_reference_real_roles_and_real_tags():
    live_tags = {tag for rules in TAG_RULES.values() for tag in rules}
    for role, combo, bonus in TAG_COMBOS:
        assert role in ROLES, role
        assert bonus > 0, role
        assert combo <= live_tags, combo - live_tags


def test_per_layer_verdicts_are_recorded_independently():
    """`tagger_primary`/`heuristic_primary` are each layer's own argmax before combining --
    what the agreement metric is computed from."""
    result = classify(
        name="X", type_line="Instant", oracle_text="Counter target spell.", tags=["sweeper"]
    )
    assert result.tagger_primary == "boardwipe"
    assert result.heuristic_primary == "stack_interaction"


# --------------------------------------------------------------------------------------
# Table building
# --------------------------------------------------------------------------------------


def test_load_card_tags_is_scoped_to_one_source(db):
    _insert_card(db, "a", tags=["sweeper"])
    db.execute("INSERT INTO card_tags (oracle_id, tag, source) VALUES ('a', 'mined', 'textmine')")
    assert load_card_tags(db) == {"a": ["sweeper"]}
    assert load_card_tags(db, source="textmine") == {"a": ["mined"]}


def test_assign_roles_covers_every_card(db):
    _insert_card(db, "wipe", name="Wrath", type_line="Sorcery", tags=["sweeper"])
    _insert_card(db, "bog", name="Bog", type_line="Land", tags=["hate-graveyard"])
    _insert_card(db, "bear", name="Bear", type_line="Creature - Bear")
    _insert_card(db, "blank", name="Blank", type_line="Enchantment")

    roles = assign_roles(db)
    assert len(roles) == 4
    by_name = roles.set_index("name")
    assert by_name.loc["Wrath", "primary_role"] == "boardwipe"
    assert by_name.loc["Bog", "primary_role"] == "land"
    assert by_name.loc["Bog", "secondary_role"] == "graveyard_hate"
    # A plain creature with no power/toughness set and no tags earns nothing -- `board_presence`
    # is not a fallback for "is a creature".
    assert by_name.loc["Bear", "primary_role"] == "other"
    assert by_name.loc["Blank", "primary_role"] == "other"


def test_role_distribution_lists_every_role(db):
    _insert_card(db, "wipe", type_line="Sorcery", tags=["sweeper"])
    table = role_distribution(assign_roles(db))
    assert list(table["role"]) == list(ROLES)
    assert table.loc[table["role"] == "boardwipe", "as_primary"].item() == 1
    assert table.loc[table["role"] == "stack_interaction", "as_primary"].item() == 0


def test_layer_agreement_only_counts_cards_both_layers_saw(db):
    _insert_card(db, "agree", type_line="Instant", oracle_text="Counter target spell.", tags=["counterspell"])
    _insert_card(db, "disagree", type_line="Instant", oracle_text="Counter target spell.", tags=["sweeper"])
    _insert_card(db, "tags_only", type_line="Sorcery", tags=["sweeper"])
    _insert_card(db, "text_only", type_line="Sorcery", oracle_text="Destroy all creatures.")

    stats = layer_agreement(assign_roles(db))
    assert stats["n_both"] == 2
    assert stats["n_agree"] == 1
    assert stats["agreement_rate"] == 0.5
    assert stats["n_tagger_only"] == 1
    assert stats["n_heuristic_only"] == 1


def test_unknown_tags_flags_a_slug_missing_from_the_vocabulary(db):
    """Every mapped tag must exist in `card_tags`; a typo would silently contribute nothing.
    Against an empty DB every mapped tag is unknown, which is what makes the check meaningful
    -- the live-vocabulary run is `python -m edhcut.analysis.roles build`."""
    missing = unknown_tags(db)
    assert "sweeper" in missing

    _insert_card(db, "a", tags=sorted({tag for rules in TAG_RULES.values() for tag in rules}))
    assert not {tag for rules in TAG_RULES.values() for tag in rules} & unknown_tags(db)


def test_build_and_save_round_trips(db, tmp_path):
    _insert_card(db, "wipe", name="Wrath", type_line="Sorcery", tags=["sweeper"])
    _insert_card(db, "bog", name="Bog", type_line="Land", tags=["hate-graveyard"])

    stats = build_and_save(db, out_dir=tmp_path)
    assert stats["n_cards"] == 2
    assert stats["n_with_secondary"] == 1

    reloaded = load_roles(tmp_path)
    assert (tmp_path / "roles.parquet").exists()
    assert set(reloaded["primary_role"]) == {"boardwipe", "land"}
    assert pd.isna(reloaded.set_index("name").loc["Wrath", "secondary_role"])
