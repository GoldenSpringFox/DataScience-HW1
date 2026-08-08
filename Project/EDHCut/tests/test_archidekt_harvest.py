"""harvest_slot() integration tests against a fake session — no network, no real HTTP cache."""

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from edhcut.db import connect
from edhcut.ingest.archidekt import harvest_slot

NOW = datetime.now(timezone.utc)
RECENT = (NOW - timedelta(days=30)).isoformat().replace("+00:00", "Z")
STALE = (NOW - timedelta(days=800)).isoformat().replace("+00:00", "Z")


def _listing(deck_id: int, updated_at: str, view_count: int = 100) -> dict:
    return {"id": deck_id, "name": f"deck-{deck_id}", "viewCount": view_count, "updatedAt": updated_at}


def _next_data_html(results: list[dict], has_next: bool) -> str:
    payload = {
        "props": {
            "pageProps": {
                "deckResults": {
                    "count": len(results),
                    "next": "http://10.0.0.1/api/decks/v3/?page=2" if has_next else None,
                    "results": results,
                }
            }
        }
    }
    return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'


def _deck_json(
    deck_id: int,
    *,
    commander_uid: str,
    commander_name: str,
    partner_uid: str | None = None,
    partner_name: str | None = None,
    extra_cards: list[dict] | None = None,
    updated_at: str = RECENT,
) -> dict:
    cards = [
        {
            "categories": ["Commander"],
            "quantity": 1,
            "card": {"oracleCard": {"uid": commander_uid, "name": commander_name}},
        }
    ]
    if partner_uid:
        cards.append({
            "categories": ["Commander"],
            "quantity": 1,
            "card": {"oracleCard": {"uid": partner_uid, "name": partner_name}},
        })
    cards.extend(extra_cards or [])
    return {
        "id": deck_id,
        "name": f"deck-{deck_id}",
        "viewCount": 100,
        "updatedAt": updated_at,
        "createdAt": updated_at,
        "deckFormat": 3,
        "categories": [
            {"name": "Commander", "includedInDeck": True},
            {"name": "Creature", "includedInDeck": True},
            {"name": "Land", "includedInDeck": True},
            {"name": "Maybeboard", "includedInDeck": False},
        ],
        "cards": cards,
    }


def _library_card(name: str, uid: str, categories: list[str] | None = None, quantity: int = 1) -> dict:
    return {
        "categories": categories if categories is not None else ["Creature"],
        "quantity": quantity,
        "card": {"oracleCard": {"uid": uid, "name": name}},
    }


def _filler(quantity: int) -> dict:
    """A generic land entry used to pad a fixture deck up to an exact physical card count."""
    return _library_card("Filler Land", "filler-uid", categories=["Land"], quantity=quantity)


class _FakeResponse:
    def __init__(self, *, json_data=None, text=None, status_code=200):
        self._json = json_data
        self.text = text or ""
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"{self.status_code} error")
            err.response = self
            raise err

    def json(self):
        return self._json


class FakeArchidektSession:
    """Serves canned search-page HTML / deck JSON. Implements the subset request_with_retry uses."""

    def __init__(self, search_pages: list[str], decks: dict[int, dict]):
        self._search_pages = search_pages
        self._decks = decks
        self.deck_requests: list[int] = []

    def request(self, method, url, params=None, **kwargs):
        if "search/decks" in url:
            page = params["page"]
            return _FakeResponse(text=self._search_pages[page - 1])
        deck_id = int(url.rstrip("/").rsplit("/", 1)[-1])
        self.deck_requests.append(deck_id)
        # A deck_id absent from the fixture dict simulates Archidekt 404ing a deck that
        # showed up in search results but was deleted/made private since being indexed.
        if deck_id not in self._decks:
            return _FakeResponse(status_code=404)
        return _FakeResponse(json_data=self._decks[deck_id])


class FakeMultiSearchSession:
    """Like FakeArchidektSession, but routes each search to its own page list by the params
    that distinguish the harvester's passes: `cardName` (exact-pair pass) and `colors`
    (single-partner fallback pass).

    `searches` is keyed by `(commanderName, cardName, colors)` — use None for an absent
    param — and maps to that search's list of page HTML strings.
    """

    def __init__(self, searches: dict[tuple, list[str]], decks: dict[int, dict]):
        self._searches = searches
        self._decks = decks
        self.deck_requests: list[int] = []
        self.search_requests: list[tuple] = []

    def request(self, method, url, params=None, **kwargs):
        if "search/decks" in url:
            key = (params.get("commanderName"), params.get("cardName"), params.get("colors"))
            self.search_requests.append(key)
            pages = self._searches.get(key)
            if pages is None:
                raise AssertionError(f"unexpected search {key!r}; known: {list(self._searches)}")
            page = params["page"]
            # Out-of-range page = search exhausted; mirror Archidekt returning no results.
            if page > len(pages):
                return _FakeResponse(text=_next_data_html([], has_next=False))
            return _FakeResponse(text=pages[page - 1])
        deck_id = int(url.rstrip("/").rsplit("/", 1)[-1])
        self.deck_requests.append(deck_id)
        return _FakeResponse(json_data=self._decks[deck_id])


@pytest.fixture
def db_with_cards(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO cards (oracle_id, name, color_identity, legal_commander) "
            "VALUES (?, ?, ?, 1)",
            [
                ("krenko-uid", "Krenko, Mob Boss", '["R"]'),
                ("yoshi-uid", "Yoshimaru, Ever Faithful", '["W"]'),
                ("bruse-uid", "Bruse Tarl, Boorish Herder", '["R", "W"]'),
                # Stand-in partners a fallback pass can turn up alongside Yoshimaru/Bruse.
                ("kediss-uid", "Kediss, Emberclaw Familiar", '["R"]'),
                ("jeska-uid", "Jeska, Thrice Reborn", '["R"]'),
                ("sol-ring-uid", "Sol Ring", "[]"),
                ("recruiter-uid", "Goblin Recruiter", '["R"]'),
                ("filler-uid", "Filler Land", "[]"),
                ("mountain-uid", "Mountain", "[]"),
            ],
        )
        conn.executemany(
            "INSERT INTO card_names (name_normalized, oracle_id) VALUES (?, ?)",
            [
                ("krenko mob boss", "krenko-uid"),
                ("yoshimaru ever faithful", "yoshi-uid"),
                ("bruse tarl boorish herder", "bruse-uid"),
            ],
        )
        conn.commit()
    with connect(db_path) as conn:
        yield conn


def test_skips_stale_decks_without_fetching_them(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(1, STALE), _listing(2, RECENT)], has_next=False)],
        decks={2: _deck_json(
            2, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
            extra_cards=[_filler(99)],
        )},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "filler-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.stale_skipped == 1
    assert stats.decks_kept == 1
    assert session.deck_requests == [2]  # never fetched the stale one


def test_deck_that_404s_is_skipped_not_fatal(db_with_cards) -> None:
    """A deck can surface in search results and then 404 by fetch time (deleted/made private
    since being indexed) — the harvester should skip it and keep going, not abort the slot."""
    conn = db_with_cards
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(1, RECENT), _listing(2, RECENT)], has_next=False)],
        decks={2: _deck_json(
            2, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
            extra_cards=[_filler(99)],
        )},  # deck 1 deliberately absent -> FakeArchidektSession serves it as a 404
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "filler-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.fetch_failed == 1
    assert stats.decks_kept == 1
    assert stats.error is None
    assert session.deck_requests == [1, 2]  # both attempted; the 404 didn't stop the second


def test_single_commander_mismatch_is_flagged(db_with_cards) -> None:
    conn = db_with_cards
    mismatched_deck = _deck_json(5, commander_uid="sol-ring-uid", commander_name="Sol Ring")
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(5, RECENT)], has_next=False)],
        decks={5: mismatched_deck},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "sol-ring-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.commander_mismatch_rejected == 1
    assert stats.decks_kept == 0
    assert len(stats.flagged) == 1
    assert "archidekt.com/decks/5/" in stats.flagged[0].url


def test_partner_mismatch_is_rejected_but_not_flagged(db_with_cards) -> None:
    conn = db_with_cards
    # Yoshimaru is commander, but no Bruse Tarl partner -> should reject without flagging
    # (routine outcome for partner slots per docs/archidekt_api.md, not worth surfacing).
    solo_deck = _deck_json(9, commander_uid="yoshi-uid", commander_name="Yoshimaru, Ever Faithful")
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(9, RECENT)], has_next=False)],
        decks={9: solo_deck},
    )
    stats = harvest_slot(
        conn, session, ["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"],
        {"yoshi-uid", "bruse-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.commander_mismatch_rejected == 1
    assert stats.decks_kept == 0
    assert stats.flagged == []


def test_genuine_partner_deck_is_kept_and_commanders_excluded_from_deck_cards(db_with_cards) -> None:
    conn = db_with_cards
    deck = _deck_json(
        11, commander_uid="yoshi-uid", commander_name="Yoshimaru, Ever Faithful",
        partner_uid="bruse-uid", partner_name="Bruse Tarl, Boorish Herder",
        extra_cards=[_library_card("Sol Ring", "sol-ring-uid"), _filler(97)],  # 1 + 97 = 98
    )
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(11, RECENT)], has_next=False)],
        decks={11: deck},
    )
    stats = harvest_slot(
        conn, session, ["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"],
        {"yoshi-uid", "bruse-uid", "sol-ring-uid", "filler-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 1
    assert stats.cards_written == 2  # Sol Ring + filler — both commanders excluded

    deck_row = conn.execute(
        "SELECT commander_oracle_id, partner_oracle_id FROM decks WHERE source_id = '11'"
    ).fetchone()
    assert deck_row == ("yoshi-uid", "bruse-uid")

    card_oracle_ids = {r[0] for r in conn.execute("SELECT oracle_id FROM deck_cards")}
    assert card_oracle_ids == {"sol-ring-uid", "filler-uid"}


def test_maybeboard_and_unresolved_cards_are_excluded_from_deck_cards(db_with_cards) -> None:
    conn = db_with_cards
    deck = _deck_json(
        13, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[
            _library_card("Sol Ring", "sol-ring-uid"),
            _library_card("Cut Candidate", "cut-uid", categories=["Maybeboard"]),
            _library_card("Banned Staple", "banned-uid"),  # not in known_oracle_ids
            _filler(97),  # Sol Ring(1) + Banned Staple(1) + filler(97) = 99 physical cards
        ],
    )
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(13, RECENT)], has_next=False)],
        decks={13: deck},
    )
    known_oracle_ids = {"krenko-uid", "sol-ring-uid", "filler-uid"}  # excludes "banned-uid"
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], known_oracle_ids,
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 1
    assert stats.cards_written == 2  # Sol Ring + filler (Cut Candidate excluded, Banned unresolved)
    assert stats.unresolved_oracle_ids == 1
    assert stats.flagged == []  # 1/99 unresolved is well under the flag threshold
    card_oracle_ids = {r[0] for r in conn.execute("SELECT oracle_id FROM deck_cards")}
    assert card_oracle_ids == {"sol-ring-uid", "filler-uid"}


def test_re_harvesting_a_deck_replaces_its_cards_not_duplicates(db_with_cards) -> None:
    conn = db_with_cards
    known_oracle_ids = {"krenko-uid", "sol-ring-uid", "recruiter-uid", "filler-uid"}
    deck_v1 = _deck_json(
        21, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[_library_card("Sol Ring", "sol-ring-uid"), _filler(98)],
    )
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(21, RECENT)], has_next=False)],
        decks={21: deck_v1},
    )
    harvest_slot(
        conn, session, ["Krenko, Mob Boss"], known_oracle_ids,
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )

    # Re-harvest the same deck, now with Sol Ring cut and Goblin Recruiter added instead.
    deck_v2 = _deck_json(
        21, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[_library_card("Goblin Recruiter", "recruiter-uid"), _filler(98)],
    )
    session2 = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(21, RECENT)], has_next=False)],
        decks={21: deck_v2},
    )
    harvest_slot(
        conn, session2, ["Krenko, Mob Boss"], known_oracle_ids,
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )

    deck_count = conn.execute("SELECT COUNT(*) FROM decks WHERE source_id = '21'").fetchone()[0]
    assert deck_count == 1  # upserted, not duplicated

    card_oracle_ids = {r[0] for r in conn.execute("SELECT oracle_id FROM deck_cards")}
    assert card_oracle_ids == {"recruiter-uid", "filler-uid"}  # old Sol Ring row gone


def test_wrong_card_count_is_silently_rejected_not_flagged(db_with_cards) -> None:
    # A "68 Mountain" entry + Sol Ring = 69 physical library cards from just 2 deck_cards
    # rows — well short of 99. Confirms (a) an incomplete/wrong-size deck is excluded
    # entirely per the user's explicit instruction (not merely flagged), and (b) the count
    # driving that decision is the quantity-summed physical total, not the distinct row
    # count (a "68 Mountain" entry is 1 row but 68 of the would-be 99 cards).
    conn = db_with_cards
    mountain_card = _library_card("Mountain", "mountain-uid", categories=["Land"], quantity=68)
    deck = _deck_json(
        31, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[_library_card("Sol Ring", "sol-ring-uid"), mountain_card],
    )
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(31, RECENT)], has_next=False)],
        decks={31: deck},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "sol-ring-uid", "mountain-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.invalid_size_rejected == 1
    assert stats.decks_kept == 0
    assert stats.flagged == []  # silently excluded, not surfaced for manual review
    assert conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM deck_cards").fetchone()[0] == 0


def test_exact_size_deck_with_high_quantity_land_is_kept(db_with_cards) -> None:
    conn = db_with_cards
    mountain_card = _library_card("Mountain", "mountain-uid", categories=["Land"], quantity=99)
    deck = _deck_json(
        32, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[mountain_card],
    )
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(32, RECENT)], has_next=False)],
        decks={32: deck},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "mountain-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 1
    assert stats.invalid_size_rejected == 0
    assert stats.cards_written == 1  # one distinct row, 99 physical cards
    assert stats.flagged == []


def test_maybeboard_card_kept_when_included_flag_is_true_despite_cotagging(db_with_cards) -> None:
    # Real-world pattern found live (deck 3085756, "(WIP) no krenkombos"): a deck marks
    # "Maybeboard" as includedInDeck: true and tags every card there with an additional
    # functional category too (e.g. ["Maybeboard", "Ramp"]). Unlike the built-in "Sideboard"
    # category, "Maybeboard" isn't hardcoded — it's an ordinary category, so its own flag is
    # authoritative and cards whose *first* category is "Maybeboard" count as real inclusions
    # here. (An earlier version of this rule hardcoded "Maybeboard" out unconditionally too —
    # that happened to match the decks tested at the time but broke on this one, discovered
    # when the user pasted this real deck's card list and it was 20 cards short.)
    # Uses "sol-ring-uid" (pre-seeded by db_with_cards) rather than an invented uid, since
    # deck_cards.oracle_id has a foreign key onto cards.oracle_id.
    conn = db_with_cards
    maybeboard_card = _library_card(
        "Sol Ring", "sol-ring-uid", categories=["Maybeboard", "Ramp"]
    )
    deck = _deck_json(
        34, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[maybeboard_card, _filler(98)],
    )
    # Override the deck's category list: Maybeboard marked includedInDeck: true.
    deck["categories"] = [
        {"name": "Commander", "includedInDeck": True},
        {"name": "Ramp", "includedInDeck": True},
        {"name": "Maybeboard", "includedInDeck": True},
        {"name": "Land", "includedInDeck": True},
    ]
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(34, RECENT)], has_next=False)],
        decks={34: deck},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "sol-ring-uid", "filler-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    # Sol Ring counted -> 98 filler + 1 Sol Ring = 99 -> exact size -> kept.
    assert stats.invalid_size_rejected == 0
    assert stats.decks_kept == 1
    card_oracle_ids = {r[0] for r in conn.execute("SELECT oracle_id FROM deck_cards")}
    assert "sol-ring-uid" in card_oracle_ids


def test_sideboard_card_excluded_even_when_included_flag_is_true(db_with_cards) -> None:
    # The built-in "Sideboard" category is the one hardcoded exception: Archidekt's API
    # reports includedInDeck: true for it on every real deck checked (as if it were a normal
    # category, same as "Commander" always being true), yet a card whose first category is
    # "Sideboard" never actually counts toward the deck on the site itself — confirmed on
    # decks 18694948 and 3085756.
    conn = db_with_cards
    sideboard_card = _library_card(
        "Shared Animosity", "animosity-uid", categories=["Sideboard"]
    )
    deck = _deck_json(
        36, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[sideboard_card, _filler(99)],
    )
    deck["categories"] = [
        {"name": "Commander", "includedInDeck": True},
        {"name": "Sideboard", "includedInDeck": True},
        {"name": "Land", "includedInDeck": True},
    ]
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(36, RECENT)], has_next=False)],
        decks={36: deck},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "animosity-uid", "filler-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    # Shared Animosity excluded -> exactly 99 filler cards -> kept.
    assert stats.decks_kept == 1
    assert stats.invalid_size_rejected == 0
    card_oracle_ids = {r[0] for r in conn.execute("SELECT oracle_id FROM deck_cards")}
    assert "animosity-uid" not in card_oracle_ids
    assert card_oracle_ids == {"filler-uid"}


def test_custom_maybeboard_subboard_name_is_excluded(db_with_cards) -> None:
    # Real-world pattern: a deck uses a custom sub-board name like "Maybeboard - Goblin"
    # instead of the literal "Maybeboard". Custom category names aren't hardcoded specially
    # (only "Sideboard" is) — this one is excluded simply because its own includedInDeck flag
    # is false, same as any other ordinary not-included category would be.
    conn = db_with_cards
    custom_maybeboard_card = _library_card(
        "Goblin Chirurgeon", "chirurgeon-uid", categories=["Maybeboard - Goblin", "Creature"]
    )
    deck = _deck_json(
        35, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[custom_maybeboard_card, _filler(99)],
    )
    deck["categories"] = [
        {"name": "Commander", "includedInDeck": True},
        {"name": "Creature", "includedInDeck": True},
        {"name": "Maybeboard - Goblin", "includedInDeck": False},
        {"name": "Land", "includedInDeck": True},
    ]
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(35, RECENT)], has_next=False)],
        decks={35: deck},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "chirurgeon-uid", "filler-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    # Goblin Chirurgeon excluded -> exactly 99 filler cards -> kept.
    assert stats.decks_kept == 1
    assert stats.invalid_size_rejected == 0
    card_oracle_ids = {r[0] for r in conn.execute("SELECT oracle_id FROM deck_cards")}
    assert "chirurgeon-uid" not in card_oracle_ids
    assert card_oracle_ids == {"filler-uid"}


def test_high_unresolved_ratio_on_valid_size_deck_is_flagged(db_with_cards) -> None:
    conn = db_with_cards
    # Exactly 99 physical cards (valid size), but 20 of them (>15%) fail resolution — should
    # be kept (right size) but flagged (unusual composition), unlike the wrong-size case.
    unresolved_cards = [_library_card(f"Banned {i}", f"banned-{i}-uid") for i in range(20)]
    deck = _deck_json(
        41, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[*unresolved_cards, _filler(79)],  # 20 + 79 = 99
    )
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(41, RECENT)], has_next=False)],
        decks={41: deck},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "filler-uid"},  # banned-* unresolved
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 1
    assert stats.unresolved_oracle_ids == 20
    assert len(stats.flagged) == 1
    assert "20/99" in stats.flagged[0].reason


def test_deck_log_file_records_one_line_per_candidate(db_with_cards, tmp_path) -> None:
    conn = db_with_cards
    # deck 1: stale -> skipped before fetch
    # deck 2: kept cleanly (exactly 99 cards, nothing unresolved)
    # deck 3: commander mismatch
    # deck 4: kept, but with 1 unresolved card among its 99 physical cards
    deck2 = _deck_json(
        2, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[_filler(99)],
    )
    deck3 = _deck_json(3, commander_uid="sol-ring-uid", commander_name="Sol Ring")
    deck4 = _deck_json(
        4, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
        extra_cards=[_library_card("Banned Staple", "banned-uid"), _filler(98)],
    )
    session = FakeArchidektSession(
        search_pages=[_next_data_html(
            [_listing(1, STALE), _listing(2, RECENT), _listing(3, RECENT), _listing(4, RECENT)],
            has_next=False,
        )],
        decks={2: deck2, 3: deck3, 4: deck4},
    )
    log_path = tmp_path / "harvest_log.txt"
    with open(log_path, "a", encoding="utf-8") as log_file:
        harvest_slot(
            conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "sol-ring-uid", "filler-uid"},
            decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
            log_file=log_file,
        )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines == [
        "Krenko, Mob Boss, problems: stale, https://archidekt.com/decks/1/",
        "Krenko, Mob Boss, problems: none, https://archidekt.com/decks/2/",
        "Krenko, Mob Boss, problems: mismatch, https://archidekt.com/decks/3/",
        "Krenko, Mob Boss, problems: illegal_cards [1] (Banned Staple), "
        "https://archidekt.com/decks/4/",
    ]


def test_error_mid_slot_is_captured_not_raised(db_with_cards) -> None:
    conn = db_with_cards

    class _ExplodingSession(FakeArchidektSession):
        def request(self, method, url, params=None, **kwargs):
            if "search/decks" in url:
                raise ConnectionError("simulated network failure")
            return super().request(method, url, params=params, **kwargs)

    session = _ExplodingSession(search_pages=[], decks={})
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.error is not None
    assert "Krenko" in stats.error
    assert "simulated network failure" in stats.error


# --- Partner-slot fallback: when too few decks run the exact pair, top up with decks running
# --- one partner at the pair's combined color identity. See harvest_slot()'s docstring.

YOSHI_BRUSE = ["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"]
YOSHI_BRUSE_SLOT_KEY = "yoshi-uid+bruse-uid"
# Search keys for FakeMultiSearchSession: (commanderName, cardName, colors).
EXACT_PAIR_SEARCH = ("Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder", None)
YOSHI_FALLBACK_SEARCH = ("Yoshimaru, Ever Faithful", None, "WR")
BRUSE_FALLBACK_SEARCH = ("Bruse Tarl, Boorish Herder", None, "WR")

PARTNER_KNOWN_IDS = {
    "yoshi-uid", "bruse-uid", "kediss-uid", "jeska-uid", "filler-uid", "sol-ring-uid",
}


def _pair_deck(deck_id: int) -> dict:
    """A genuine Yoshimaru + Bruse Tarl deck (2 commanders + 98 library cards)."""
    return _deck_json(
        deck_id,
        commander_uid="yoshi-uid", commander_name="Yoshimaru, Ever Faithful",
        partner_uid="bruse-uid", partner_name="Bruse Tarl, Boorish Herder",
        extra_cards=[_filler(98)],
    )


def _yoshi_fallback_deck(deck_id: int) -> dict:
    """Yoshimaru alongside a *different* red partner — what the fallback search turns up."""
    return _deck_json(
        deck_id,
        commander_uid="yoshi-uid", commander_name="Yoshimaru, Ever Faithful",
        partner_uid="kediss-uid", partner_name="Kediss, Emberclaw Familiar",
        extra_cards=[_filler(98)],
    )


def _bruse_fallback_deck(deck_id: int) -> dict:
    return _deck_json(
        deck_id,
        commander_uid="bruse-uid", commander_name="Bruse Tarl, Boorish Herder",
        partner_uid="jeska-uid", partner_name="Jeska, Thrice Reborn",
        extra_cards=[_filler(98)],
    )


def _one_page(deck_ids: list[int]) -> list[str]:
    return [_next_data_html([_listing(i, RECENT) for i in deck_ids], has_next=False)]


def test_partner_slot_does_not_use_fallback_when_exact_pairs_suffice(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeMultiSearchSession(
        searches={EXACT_PAIR_SEARCH: _one_page([1, 2, 3, 4])},
        decks={i: _pair_deck(i) for i in (1, 2, 3, 4)},
    )
    stats = harvest_slot(
        conn, session, YOSHI_BRUSE, PARTNER_KNOWN_IDS,
        decks_per_commander=4, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 4
    # Quota met by the exact pair alone -> no fallback search issued at all.
    assert stats.exact_pair_decks is None
    assert stats.fallback_decks == 0
    assert session.search_requests == [EXACT_PAIR_SEARCH]


def test_partner_fallback_splits_remaining_quota_evenly_between_partners(db_with_cards) -> None:
    # 2 exact pairs found against a quota of 10 -> 8 remaining -> 4 to each partner.
    conn = db_with_cards
    decks = {1: _pair_deck(1), 2: _pair_deck(2)}
    decks.update({i: _yoshi_fallback_deck(i) for i in range(10, 20)})
    decks.update({i: _bruse_fallback_deck(i) for i in range(30, 40)})
    session = FakeMultiSearchSession(
        searches={
            EXACT_PAIR_SEARCH: _one_page([1, 2]),
            YOSHI_FALLBACK_SEARCH: _one_page(list(range(10, 20))),
            BRUSE_FALLBACK_SEARCH: _one_page(list(range(30, 40))),
        },
        decks=decks,
    )
    stats = harvest_slot(
        conn, session, YOSHI_BRUSE, PARTNER_KNOWN_IDS,
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 10
    assert stats.exact_pair_decks == 2
    assert stats.fallback_decks == 8

    kept_ids = {int(r[0]) for r in conn.execute(
        "SELECT source_id FROM decks WHERE slot_key = ?", (YOSHI_BRUSE_SLOT_KEY,)
    )}
    assert len(kept_ids) == 10
    # Neither partner exceeded its half: ids 10-19 are Yoshimaru's, 30-39 are Bruse's.
    assert len(kept_ids & set(range(10, 20))) == 4
    assert len(kept_ids & set(range(30, 40))) == 4


def test_partner_fallback_rolls_shortfall_over_to_the_other_partner(db_with_cards) -> None:
    # Quota 10, 2 exact pairs -> 8 remaining -> 4 each. Yoshimaru's search holds only 1 deck,
    # so Bruse must absorb the other 7 rather than stopping at its own half of 4.
    conn = db_with_cards
    decks = {1: _pair_deck(1), 2: _pair_deck(2), 10: _yoshi_fallback_deck(10)}
    decks.update({i: _bruse_fallback_deck(i) for i in range(30, 45)})
    session = FakeMultiSearchSession(
        searches={
            EXACT_PAIR_SEARCH: _one_page([1, 2]),
            YOSHI_FALLBACK_SEARCH: _one_page([10]),
            BRUSE_FALLBACK_SEARCH: _one_page(list(range(30, 45))),
        },
        decks=decks,
    )
    stats = harvest_slot(
        conn, session, YOSHI_BRUSE, PARTNER_KNOWN_IDS,
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 10
    assert stats.exact_pair_decks == 2
    assert stats.fallback_decks == 8
    kept_ids = {int(r[0]) for r in conn.execute("SELECT source_id FROM decks")}
    assert 10 in kept_ids  # Yoshimaru's only fallback deck was taken
    assert len(kept_ids & set(range(30, 45))) == 7  # Bruse covered the rest


def test_partner_fallback_stores_real_commanders_and_slot_key(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeMultiSearchSession(
        searches={
            EXACT_PAIR_SEARCH: _one_page([]),
            YOSHI_FALLBACK_SEARCH: _one_page([10]),
            BRUSE_FALLBACK_SEARCH: _one_page([]),
        },
        decks={10: _yoshi_fallback_deck(10)},
    )
    stats = harvest_slot(
        conn, session, YOSHI_BRUSE, PARTNER_KNOWN_IDS,
        decks_per_commander=2, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 1
    row = conn.execute(
        "SELECT commander_oracle_id, partner_oracle_id, slot_key FROM decks"
    ).fetchone()
    # The deck's *real* commanders are recorded (Kediss, not Bruse Tarl); slot_key is what
    # ties it back to the Yoshimaru + Bruse Tarl corpus.
    assert row == ("yoshi-uid", "kediss-uid", YOSHI_BRUSE_SLOT_KEY)
    # The stand-in partner is a commander, so it must not also count as a library card.
    card_ids = {r[0] for r in conn.execute("SELECT oracle_id FROM deck_cards")}
    assert card_ids == {"filler-uid"}


def test_exact_pair_deck_is_not_double_counted_by_fallback_searches(db_with_cards) -> None:
    # A genuine pair deck matches all three searches. It must be kept once, and the fallback
    # must not re-fetch it (dedup happens on the listing, before the deck request).
    conn = db_with_cards
    session = FakeMultiSearchSession(
        searches={
            EXACT_PAIR_SEARCH: _one_page([1]),
            YOSHI_FALLBACK_SEARCH: _one_page([1, 10]),
            BRUSE_FALLBACK_SEARCH: _one_page([1]),
        },
        decks={1: _pair_deck(1), 10: _yoshi_fallback_deck(10)},
    )
    stats = harvest_slot(
        conn, session, YOSHI_BRUSE, PARTNER_KNOWN_IDS,
        decks_per_commander=5, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 2
    assert conn.execute("SELECT COUNT(*) FROM decks").fetchone()[0] == 2
    assert session.deck_requests.count(1) == 1


def test_single_commander_slot_stores_bare_oracle_id_as_slot_key(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeArchidektSession(
        search_pages=[_next_data_html([_listing(2, RECENT)], has_next=False)],
        decks={2: _deck_json(
            2, commander_uid="krenko-uid", commander_name="Krenko, Mob Boss",
            extra_cards=[_filler(99)],
        )},
    )
    stats = harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid", "filler-uid"},
        decks_per_commander=10, staleness_cutoff_days=730, show_progress=False,
    )
    assert stats.decks_kept == 1
    assert stats.exact_pair_decks is None  # not a partner slot -> no fallback bookkeeping
    row = conn.execute("SELECT slot_key, partner_oracle_id FROM decks").fetchone()
    assert row == ("krenko-uid", None)
