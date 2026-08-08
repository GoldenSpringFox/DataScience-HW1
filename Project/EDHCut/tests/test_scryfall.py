from edhcut.ingest.scryfall import (
    _image_uri,
    build_card_row,
    build_name_aliases,
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
    "power": None,
    "toughness": None,
    "prices": {"usd": "2.50"},
    "game_changer": False,
    "legalities": {"commander": "legal"},
    "layout": "normal",
    "produced_mana": ["C"],
    "games": ["paper", "mtgo"],
    "image_uris": {"normal": "https://cards.scryfall.io/normal/front/a/b/sol-ring.jpg"},
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
            "power": "2",
            "toughness": "3",
            "image_uris": {"normal": "https://cards.scryfall.io/normal/front/d/e/captive.jpg"},
        },
        {
            "name": "Ulvenwald Abomination",
            "mana_cost": "",
            "oracle_text": "Trample",
            "colors": ["G"],
            "image_uris": {"normal": "https://cards.scryfall.io/normal/back/d/e/abomination.jpg"},
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
    assert row["image_uri"] == "https://cards.scryfall.io/normal/front/a/b/sol-ring.jpg"
    assert row["power"] is None
    assert row["toughness"] is None


def test_build_card_row_falls_back_to_face_text_for_multi_face_cards() -> None:
    row = build_card_row(TRANSFORM_CARD)
    assert row["mana_cost"] == "{2}"
    assert "Trample" in row["oracle_text"]
    assert row["colors"] == "[\"G\"]"
    assert row["image_uri"] == "https://cards.scryfall.io/normal/front/d/e/captive.jpg"
    assert row["power"] == "2"
    assert row["toughness"] == "3"


def test_image_uri_prefers_top_level() -> None:
    assert _image_uri(SOL_RING) == "https://cards.scryfall.io/normal/front/a/b/sol-ring.jpg"


def test_image_uri_falls_back_to_front_face_for_multi_face_cards() -> None:
    # Front face specifically, not the back -- one representative image per card.
    assert _image_uri(TRANSFORM_CARD) == "https://cards.scryfall.io/normal/front/d/e/captive.jpg"


def test_image_uri_respects_requested_size() -> None:
    card = {"image_uris": {"normal": "normal.jpg", "large": "large.jpg"}}
    assert _image_uri(card, size="large") == "large.jpg"


def test_image_uri_none_when_missing_entirely() -> None:
    assert _image_uri({"name": "No Image Card"}) is None


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


def test_build_name_aliases_prefers_a_cards_own_full_name_over_another_cards_face_name() -> None:
    # Regression test: a real bug found live -- "Emeritus of Truce // Swords to Plowshares"
    # (a "prepare"-layout card) has a face literally named "Swords to Plowshares", the same
    # name as the real, separately-printed classic instant. A flat last-write-wins dict built
    # in file order let whichever card came later silently steal the alias from the other,
    # regardless of which one a person actually means by that name.
    entries = [
        ("classic-stp-id", "Swords to Plowshares", []),
        ("prepare-card-id", "Emeritus of Truce // Swords to Plowshares", ["Emeritus of Truce", "Swords to Plowshares"]),
    ]
    aliases = build_name_aliases(entries)
    assert aliases["swords to plowshares"] == "classic-stp-id"
    assert aliases["emeritus of truce"] == "prepare-card-id"
    assert aliases["emeritus of truce swords to plowshares"] == "prepare-card-id"


def test_build_name_aliases_order_independent() -> None:
    # Same scenario, cards processed in the opposite order -- the classic card's own name must
    # still win regardless of which one the bulk file happens to list first.
    entries = [
        ("prepare-card-id", "Emeritus of Truce // Swords to Plowshares", ["Emeritus of Truce", "Swords to Plowshares"]),
        ("classic-stp-id", "Swords to Plowshares", []),
    ]
    aliases = build_name_aliases(entries)
    assert aliases["swords to plowshares"] == "classic-stp-id"


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
