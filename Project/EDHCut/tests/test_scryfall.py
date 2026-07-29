from edhcut.ingest.scryfall import (
    build_card_row,
    can_be_commander,
    is_commander_legal,
    normalize_name,
)

SOL_RING = {
    "oracle_id": "abc-123",
    "name": "Sol Ring",
    "mana_cost": "{1}",
    "cmc": 1.0,
    "type_line": "Artifact",
    "oracle_text": "{T}: Add {C}{C}.",
    "colors": [],
    "color_identity": [],
    "keywords": [],
    "rarity": "uncommon",
    "edhrec_rank": 1,
    "prices": {"usd": "2.50"},
    "game_changer": False,
    "legalities": {"commander": "legal"},
    "layout": "normal",
    "produced_mana": ["C"],
    "games": ["paper", "mtgo"],
}

TRANSFORM_CARD = {
    "oracle_id": "def-456",
    "name": "Ulvenwald Captive // Ulvenwald Abomination",
    "cmc": 2.0,
    "type_line": "Creature — Werewolf Horror // Creature — Eldrazi Werewolf",
    "colors": None,
    "color_identity": ["G"],
    "keywords": [],
    "rarity": "uncommon",
    "layout": "transform",
    "legalities": {"commander": "legal"},
    "games": ["paper"],
    "card_faces": [
        {
            "name": "Ulvenwald Captive",
            "mana_cost": "{2}",
            "oracle_text": "At the beginning of your upkeep...",
            "colors": [],
        },
        {
            "name": "Ulvenwald Abomination",
            "mana_cost": "",
            "oracle_text": "Trample",
            "colors": ["G"],
        },
    ],
}


def test_normalize_name_lowercases_and_strips_punctuation() -> None:
    assert normalize_name("Sol Ring") == "sol ring"
    assert normalize_name("Urza's Saga") == "urza s saga"
    assert normalize_name("Kutzil, Malamet Exemplar") == "kutzil malamet exemplar"


def test_normalize_name_strips_diacritics_and_ligatures() -> None:
    assert normalize_name("Jötun Grunt") == "jotun grunt"
    assert normalize_name("Ætherize") == "aetherize"


def test_build_card_row_single_face() -> None:
    row = build_card_row(SOL_RING)
    assert row["oracle_id"] == "abc-123"
    assert row["price_usd"] == 2.5
    assert row["is_land"] is False
    assert row["can_be_commander"] is False


def test_build_card_row_falls_back_to_face_text_for_multi_face_cards() -> None:
    row = build_card_row(TRANSFORM_CARD)
    assert row["mana_cost"] == "{2}"
    assert "Trample" in row["oracle_text"]
    assert row["colors"] == "[\"G\"]"


def test_can_be_commander_legendary_creature() -> None:
    legendary = {"type_line": "Legendary Creature — Human Wizard", "oracle_text": ""}
    assert can_be_commander(legendary) is True


def test_can_be_commander_explicit_text() -> None:
    background_partner = {
        "type_line": "Legendary Enchantment — Background",
        "oracle_text": "Whenever you cast..., can be your commander.",
    }
    assert can_be_commander(background_partner) is True
    assert can_be_commander({"type_line": "Sorcery", "oracle_text": ""}) is False


def test_is_commander_legal_filter() -> None:
    assert is_commander_legal(SOL_RING) is True
    assert is_commander_legal({"legalities": {"commander": "not_legal"}}) is False


def test_commander_legal_card_kept_regardless_of_representative_printing_games() -> None:
    # Regression test: a card whose oracle_cards *representative* printing happens to be an
    # MTGO-only reprint (e.g. Goblin Chirurgeon, via a "Masters Edition" set) is still a real,
    # paper-playable, commander-legal card — legalities.commander already accounts for this
    # correctly, so it must not be excluded just because this one printing's `games` doesn't
    # include "paper". See module docstring for the 1,048-card bug this used to cause.
    mtgo_only_representative_printing = {
        "oracle_id": "chirurgeon-uid",
        "name": "Goblin Chirurgeon",
        "type_line": "Creature — Goblin",
        "legalities": {"commander": "legal"},
        "games": ["mtgo"],
    }
    assert is_commander_legal(mtgo_only_representative_printing) is True
