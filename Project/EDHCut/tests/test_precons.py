"""harvest_precons() tests against a fake session — no network, no real HTTP cache."""

import json

import pytest

from edhcut.db import connect
from edhcut.ingest.precons import (
    alternative_commander_oracle_ids,
    harvest_precons,
    parse_set_code,
)


def _card(name: str, uid: str, categories: list[str], quantity: int = 1) -> dict:
    return {
        "categories": categories,
        "quantity": quantity,
        "card": {"oracleCard": {"uid": uid, "name": name, "layout": "normal"}},
    }


def _next_data_html(precons: dict) -> str:
    payload = {"props": {"pageProps": {"precons": precons}}}
    return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'


class _FakeResponse:
    def __init__(self, *, json_data=None, text=None):
        self._json = json_data
        self.text = text or ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class FakePreconSession:
    def __init__(self, index_html: str, decks: dict[int, dict]):
        self._index_html = index_html
        self._decks = decks
        self.deck_requests: list[int] = []

    def request(self, method, url, params=None, **kwargs):
        if "commander-precons" in url:
            return _FakeResponse(text=self._index_html)
        deck_id = int(url.rstrip("/").rsplit("/", 1)[-1])
        self.deck_requests.append(deck_id)
        return _FakeResponse(json_data=self._decks[deck_id])


@pytest.fixture
def db_with_cards(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO cards (oracle_id, name, color_identity, can_be_commander, "
            "legal_commander) VALUES (?, ?, ?, ?, 1)",
            [
                ("leinore-uid", "Leinore, Autumn Sovereign", '["G", "W"]', 1),
                ("kyler-uid", "Kyler, Sigardian Emissary", '["G", "W"]', 1),
                # Same colors as the commander but not a legendary/can't-be-commander card —
                # must NOT show up as an alternative.
                ("ordinary-uid", "Ordinary Human", '["G", "W"]', 0),
                # Legendary, but a different color identity — must NOT show up either.
                ("offcolor-uid", "Offcolor Legend", '["U"]', 1),
                ("filler-uid", "Filler Land", "[]", 0),
            ],
        )
        conn.commit()
    with connect(db_path) as conn:
        yield conn


def test_parse_set_code_extracts_trailing_parens() -> None:
    assert parse_set_code("Midnight Hunt Commander (MIC)") == "MIC"


def test_parse_set_code_returns_none_without_parens() -> None:
    assert parse_set_code("Starter Commander Decks") is None


def test_alternative_commander_oracle_ids_matches_exact_color_identity(db_with_cards) -> None:
    conn = db_with_cards
    precon_cards = {"leinore-uid", "kyler-uid", "ordinary-uid", "offcolor-uid", "filler-uid"}
    result = alternative_commander_oracle_ids(conn, ["leinore-uid"], precon_cards)
    assert result == ["kyler-uid"]


def test_alternative_commander_oracle_ids_excludes_declared_commanders(db_with_cards) -> None:
    conn = db_with_cards
    result = alternative_commander_oracle_ids(
        conn, ["leinore-uid", "kyler-uid"], {"leinore-uid", "kyler-uid"}
    )
    assert result == []


def _precon_deck(deck_id: int, *, commander_uid: str, commander_name: str,
                  extra_cards: list[dict]) -> dict:
    cards = [_card(commander_name, commander_uid, ["Commander"])] + extra_cards
    return {
        "id": deck_id,
        "name": f"precon-{deck_id}",
        "deckFormat": 3,
        "categories": [
            {"name": "Commander", "includedInDeck": True},
            {"name": "Creature", "includedInDeck": True},
            {"name": "Land", "includedInDeck": True},
        ],
        "cards": cards,
    }


def test_harvest_precons_stores_commander_and_full_card_list_including_commander(
    db_with_cards,
) -> None:
    conn = db_with_cards
    deck = _precon_deck(
        2209041, commander_uid="leinore-uid", commander_name="Leinore, Autumn Sovereign",
        extra_cards=[_card("Kyler, Sigardian Emissary", "kyler-uid", ["Creature"]),
                     _card("Filler Land", "filler-uid", ["Land"], quantity=98)],
    )
    session = FakePreconSession(
        index_html=_next_data_html({"Midnight Hunt Commander (MIC)": [
            {"id": 2209041, "name": "Coven Counters - Midnight Hunt Commander"},
        ]}),
        decks={2209041: deck},
    )
    stats = harvest_precons(
        conn, session, {"leinore-uid", "kyler-uid", "filler-uid"}, show_progress=False
    )

    assert stats.precons_checked == 1
    assert stats.precons_kept == 1
    assert stats.errors == []

    row = conn.execute(
        "SELECT set_name, set_code, deck_name, commander_oracle_id, partner_oracle_id, "
        "alternative_commander_oracle_ids FROM precons WHERE precon_id = 2209041"
    ).fetchone()
    assert row[0] == "Midnight Hunt Commander (MIC)"
    assert row[1] == "MIC"
    assert row[2] == "Coven Counters - Midnight Hunt Commander"
    assert row[3] == "leinore-uid"
    assert row[4] is None
    assert json.loads(row[5]) == ["kyler-uid"]

    # The card list includes the commander itself, unlike deck_cards.
    quantities = dict(conn.execute(
        "SELECT oracle_id, qty FROM precon_cards WHERE precon_id = 2209041"
    ))
    assert quantities == {"leinore-uid": 1, "kyler-uid": 1, "filler-uid": 98}
    assert sum(quantities.values()) == 100


def test_harvest_precons_records_partner_precons(db_with_cards) -> None:
    conn = db_with_cards
    deck = {
        "id": 99,
        "name": "partner-precon",
        "deckFormat": 3,
        "categories": [{"name": "Commander", "includedInDeck": True}],
        "cards": [
            _card("Leinore, Autumn Sovereign", "leinore-uid", ["Commander"]),
            _card("Kyler, Sigardian Emissary", "kyler-uid", ["Commander"]),
            _card("Filler Land", "filler-uid", ["Land"], quantity=98),
        ],
    }
    session = FakePreconSession(
        index_html=_next_data_html({"Some Set (SET)": [{"id": 99, "name": "Partner Precon"}]}),
        decks={99: deck},
    )
    harvest_precons(conn, session, {"leinore-uid", "kyler-uid", "filler-uid"}, show_progress=False)

    row = conn.execute(
        "SELECT commander_oracle_id, partner_oracle_id FROM precons WHERE precon_id = 99"
    ).fetchone()
    assert row == ("leinore-uid", "kyler-uid")


def test_harvest_precons_counts_unresolved_cards_without_writing_them(db_with_cards) -> None:
    conn = db_with_cards
    deck = _precon_deck(
        5, commander_uid="leinore-uid", commander_name="Leinore, Autumn Sovereign",
        extra_cards=[_card("Banned Staple", "banned-uid", ["Creature"]),
                     _card("Filler Land", "filler-uid", ["Land"], quantity=98)],
    )
    session = FakePreconSession(
        index_html=_next_data_html({"Some Set (SET)": [{"id": 5, "name": "Some Precon"}]}),
        decks={5: deck},
    )
    stats = harvest_precons(
        conn, session, {"leinore-uid", "filler-uid"}, show_progress=False  # banned-uid excluded
    )
    assert stats.unresolved_oracle_ids == 1
    card_ids = {r[0] for r in conn.execute("SELECT oracle_id FROM precon_cards WHERE precon_id = 5")}
    assert "banned-uid" not in card_ids


def test_harvest_precons_reharvest_replaces_cards_not_duplicates(db_with_cards) -> None:
    conn = db_with_cards
    deck_v1 = _precon_deck(
        7, commander_uid="leinore-uid", commander_name="Leinore, Autumn Sovereign",
        extra_cards=[_card("Filler Land", "filler-uid", ["Land"], quantity=99)],
    )
    session = FakePreconSession(
        index_html=_next_data_html({"Some Set (SET)": [{"id": 7, "name": "Some Precon"}]}),
        decks={7: deck_v1},
    )
    harvest_precons(conn, session, {"leinore-uid", "filler-uid"}, show_progress=False)
    harvest_precons(conn, session, {"leinore-uid", "filler-uid"}, show_progress=False)

    assert conn.execute("SELECT COUNT(*) FROM precons").fetchone()[0] == 1
    quantities = dict(conn.execute("SELECT oracle_id, qty FROM precon_cards WHERE precon_id = 7"))
    # Not duplicated/doubled by the second run.
    assert quantities == {"leinore-uid": 1, "filler-uid": 99}


def test_harvest_precons_one_bad_listing_does_not_abort_the_run(db_with_cards) -> None:
    conn = db_with_cards
    good_deck = _precon_deck(
        1, commander_uid="leinore-uid", commander_name="Leinore, Autumn Sovereign",
        extra_cards=[_card("Filler Land", "filler-uid", ["Land"], quantity=99)],
    )

    class _PartlyBrokenSession(FakePreconSession):
        def request(self, method, url, params=None, **kwargs):
            if "commander-precons" in url:
                return super().request(method, url, params=params, **kwargs)
            deck_id = int(url.rstrip("/").rsplit("/", 1)[-1])
            if deck_id == 2:
                raise ConnectionError("simulated network failure")
            return super().request(method, url, params=params, **kwargs)

    session = _PartlyBrokenSession(
        index_html=_next_data_html({"Some Set (SET)": [
            {"id": 1, "name": "Good Precon"},
            {"id": 2, "name": "Broken Precon"},
        ]}),
        decks={1: good_deck},
    )
    stats = harvest_precons(conn, session, {"leinore-uid", "filler-uid"}, show_progress=False)

    assert stats.precons_checked == 2
    assert stats.precons_kept == 1
    assert len(stats.errors) == 1
    assert "simulated network failure" in stats.errors[0]
    assert conn.execute("SELECT COUNT(*) FROM precons").fetchone()[0] == 1
