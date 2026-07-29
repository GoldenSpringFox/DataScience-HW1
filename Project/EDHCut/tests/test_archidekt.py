from edhcut.db import connect
from edhcut.ingest.archidekt import (
    _parse_next_data,
    card_oracle_id,
    combined_color_identity,
    commander_cards,
    deck_log_line,
    included_cards,
    is_token,
    slot_key_for,
)


def _card(name: str, uid: str, categories: list[str], quantity: int = 1, layout: str = "normal") -> dict:
    return {
        "categories": categories,
        "quantity": quantity,
        "card": {
            "collectorNumber": "1",
            "edition": {"editioncode": "tst"},
            "oracleCard": {"uid": uid, "name": name, "layout": layout},
        },
    }


FIXTURE_DECK = {
    "id": 12345,
    "name": "Test Deck",
    "viewCount": 42,
    "updatedAt": "2026-07-01T00:00:00Z",
    "createdAt": "2026-01-01T00:00:00Z",
    "deckFormat": 3,
    "categories": [
        {"name": "Commander", "includedInDeck": True},
        {"name": "Creature", "includedInDeck": True},
        {"name": "Ramp", "includedInDeck": True},
        {"name": "Maybeboard", "includedInDeck": False},
        {"name": "Upgrades", "includedInDeck": False},
        # Archidekt's own API reports "Sideboard" as includedInDeck: true on every real deck
        # checked, as if it were a normal included category — but it's always genuinely
        # excluded on the site regardless (see `_HARDCODED_EXCLUDED_FIRST_CATEGORY`).
        {"name": "Sideboard", "includedInDeck": True},
    ],
    "cards": [
        _card("Krenko, Mob Boss", "krenko-uid", ["Commander"]),
        _card("Sol Ring", "sol-ring-uid", ["Ramp"]),
        _card("Goblin Recruiter", "recruiter-uid", ["Creature"]),
        _card("Some Land", "land-uid", []),  # no categories -> still included
        _card("Cut Candidate", "cut-uid", ["Maybeboard"]),  # excluded board, first category
        # "Sideboard" is hardcoded-excluded even though the deck's own flag says
        # includedInDeck: true for it — this is the one exception to "flag is authoritative".
        _card("Bench Warmer", "bench-uid", ["Sideboard"]),
        # Real inclusion co-tagged with a holding category for personal organization — only
        # the *first* category (an included one) decides; "Maybeboard" being second doesn't
        # bench it. Mirrors real decks confirmed live (e.g. "Goblin Chirurgeon" tagged
        # `["Goblin", "Maybeboard"]`).
        _card("Organized Keeper", "organized-uid", ["Creature", "Maybeboard"]),
        # First category is a not-included one -> excluded, even though a later category is
        # included. This used to be "included" under an earlier (wrong) any-category-matches
        # rule; Archidekt itself only looks at the first category.
        _card("Wishlist Card", "wishlist-uid", ["Upgrades", "Creature"]),
        # Archidekt token-tracking entry: tagged Maybeboard, but ALSO Creature — would leak
        # into the deck via the category check alone (real bug, caught live).
        _card("Goblin Token", "token-uid", ["Maybeboard", "Creature"], layout="token"),
    ],
}


def test_included_cards_excludes_maybeboard_only_cards() -> None:
    included = {card_oracle_id(c) for c in included_cards(FIXTURE_DECK)}
    assert "cut-uid" not in included


def test_included_cards_excludes_sideboard_even_when_included_flag_is_true() -> None:
    included = {card_oracle_id(c) for c in included_cards(FIXTURE_DECK)}
    assert "bench-uid" not in included


def test_included_cards_keeps_cards_with_no_categories() -> None:
    included = {card_oracle_id(c) for c in included_cards(FIXTURE_DECK)}
    assert "land-uid" in included


def test_included_cards_keeps_card_with_included_first_category_despite_later_maybeboard_tag() -> None:
    included = {card_oracle_id(c) for c in included_cards(FIXTURE_DECK)}
    assert "organized-uid" in included


def test_included_cards_excludes_card_with_not_included_first_category() -> None:
    # "Wishlist Card"'s first category ("Upgrades") isn't included, even though its second
    # ("Creature") is -> only the first category counts.
    included = {card_oracle_id(c) for c in included_cards(FIXTURE_DECK)}
    assert "wishlist-uid" not in included


def test_included_cards_keeps_ordinary_deck_cards() -> None:
    included = {card_oracle_id(c) for c in included_cards(FIXTURE_DECK)}
    assert {"sol-ring-uid", "recruiter-uid"} <= included


def test_included_cards_excludes_tokens_even_when_sharing_an_included_category() -> None:
    # "Goblin Token" is tagged both Maybeboard and Creature — the category check alone would
    # keep it (Maybeboard is first, but even ignoring that, is_token() must override).
    included = {card_oracle_id(c) for c in included_cards(FIXTURE_DECK)}
    assert "token-uid" not in included


def test_is_token_detects_token_layout() -> None:
    token_card = _card("Goblin Token", "token-uid", ["Creature"], layout="token")
    normal_card = _card("Sol Ring", "sol-ring-uid", ["Ramp"], layout="normal")
    assert is_token(token_card) is True
    assert is_token(normal_card) is False


def test_commander_cards_finds_commander_tagged_card() -> None:
    commanders = commander_cards(FIXTURE_DECK)
    assert len(commanders) == 1
    assert card_oracle_id(commanders[0]) == "krenko-uid"


def test_commander_cards_finds_both_partners() -> None:
    deck = {
        "categories": [{"name": "Commander", "includedInDeck": True}],
        "cards": [
            _card("Yoshimaru, Ever Faithful", "yoshi-uid", ["Commander"]),
            _card("Bruse Tarl, Boorish Herder", "bruse-uid", ["Commander"]),
            _card("Bruse Tarl Lookalike", "not-bruse-uid", ["Creature"]),
        ],
    }
    commander_ids = {card_oracle_id(c) for c in commander_cards(deck)}
    assert commander_ids == {"yoshi-uid", "bruse-uid"}


def test_parse_next_data_extracts_embedded_json() -> None:
    html = """
    <html><body>
    <script id="__NEXT_DATA__" type="application/json">{"props": {"pageProps": {"deckResults": {"count": 1, "next": null, "results": []}}}}</script>
    </body></html>
    """
    data = _parse_next_data(html)
    assert data["props"]["pageProps"]["deckResults"]["count"] == 1


def test_deck_log_line_no_problems() -> None:
    line = deck_log_line("Krenko, Mob Boss", "https://archidekt.com/decks/1/")
    assert line == "Krenko, Mob Boss, problems: none, https://archidekt.com/decks/1/"


def test_deck_log_line_single_problem() -> None:
    line = deck_log_line("Krenko, Mob Boss", "https://archidekt.com/decks/1/", stale=True)
    assert line == "Krenko, Mob Boss, problems: stale, https://archidekt.com/decks/1/"


def test_deck_log_line_illegal_cards_shows_count_and_names() -> None:
    line = deck_log_line(
        "Krenko, Mob Boss", "https://archidekt.com/decks/1/",
        illegal_count=2, illegal_names=["Mana Crypt", "Dockside Extortionist"],
    )
    assert line == (
        "Krenko, Mob Boss, problems: illegal_cards [2] (Mana Crypt, Dockside Extortionist), "
        "https://archidekt.com/decks/1/"
    )


def test_deck_log_line_illegal_cards_truncates_names_at_five() -> None:
    line = deck_log_line(
        "Krenko, Mob Boss", "https://archidekt.com/decks/1/",
        illegal_count=7, illegal_names=[f"Card {i}" for i in range(7)],
    )
    assert line == (
        "Krenko, Mob Boss, problems: illegal_cards [7] "
        "(Card 0, Card 1, Card 2, Card 3, Card 4), "
        "https://archidekt.com/decks/1/"
    )


def test_slot_key_for_single_commander_is_the_bare_oracle_id() -> None:
    assert slot_key_for(["krenko-uid"]) == "krenko-uid"


def test_slot_key_for_partner_pair_joins_in_slot_order() -> None:
    assert slot_key_for(["yoshi-uid", "bruse-uid"]) == "yoshi-uid+bruse-uid"


def test_combined_color_identity_unions_partners_in_wubrg_order(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO cards (oracle_id, name, color_identity) VALUES (?, ?, ?)",
            [
                ("yoshi-uid", "Yoshimaru, Ever Faithful", '["W"]'),
                # Deliberately not in WUBRG order in the source data.
                ("bruse-uid", "Bruse Tarl, Boorish Herder", '["R", "W"]'),
                ("colorless-uid", "Sol Ring", "[]"),
            ],
        )
        conn.commit()
        assert combined_color_identity(conn, ["yoshi-uid"]) == "W"
        assert combined_color_identity(conn, ["bruse-uid"]) == "WR"
        assert combined_color_identity(conn, ["yoshi-uid", "bruse-uid"]) == "WR"
        assert combined_color_identity(conn, ["colorless-uid"]) == ""


def test_deck_log_line_multiple_problems_comma_separated() -> None:
    line = deck_log_line(
        "Krenko, Mob Boss", "https://archidekt.com/decks/1/",
        wrong_size=True, illegal_count=1, illegal_names=["Mana Crypt"],
    )
    assert line == (
        "Krenko, Mob Boss, problems: wrong_size, illegal_cards [1] (Mana Crypt), "
        "https://archidekt.com/decks/1/"
    )
