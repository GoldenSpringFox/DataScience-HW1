"""Task 5.7 meta-sample harvest mode: order_by threading, cohort storage, the roster-priority
upsert guard, already_seen_source_ids resumability, and run_meta_sample orchestration — all
against a fake session, no network."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from edhcut.db import connect
from edhcut.ingest.archidekt import (
    _commander_names_for,
    _upsert_deck,
    harvest_slot,
    run_meta_sample,
)
from edhcut.ingest.scryfall import normalize_name

NOW = datetime.now(timezone.utc)
RECENT = (NOW - timedelta(days=30)).isoformat().replace("+00:00", "Z")


def _listing(deck_id: int, updated_at: str = RECENT) -> dict:
    return {"id": deck_id, "name": f"deck-{deck_id}", "viewCount": 100, "updatedAt": updated_at}


def _next_data_html(results: list[dict]) -> str:
    payload = {"props": {"pageProps": {"deckResults": {"count": len(results), "next": None, "results": results}}}}
    return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'


def _deck_json(deck_id: int, *, commander_uid: str, commander_name: str, updated_at: str = RECENT) -> dict:
    return {
        "id": deck_id,
        "name": f"deck-{deck_id}",
        "viewCount": 100,
        "updatedAt": updated_at,
        "createdAt": updated_at,
        "deckFormat": 3,
        "categories": [
            {"name": "Commander", "includedInDeck": True},
            {"name": "Land", "includedInDeck": True},
        ],
        "cards": [
            {"categories": ["Commander"], "quantity": 1, "card": {"oracleCard": {"uid": commander_uid, "name": commander_name}}},
            {"categories": ["Land"], "quantity": 99, "card": {"oracleCard": {"uid": "filler-uid", "name": "Filler Land"}}},
        ],
    }


class _FakeResponse:
    def __init__(self, *, json_data=None, text=None, status_code=200):
        self._json = json_data
        self.text = text or ""
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class FakeSession:
    """Serves one page of results per commanderName; records every search's orderBy param."""

    def __init__(self, pages: dict[str, list[dict]], decks: dict[int, dict]):
        self._pages = pages
        self._decks = decks
        self.deck_requests: list[int] = []
        self.search_order_by: list[str] = []

    def request(self, method, url, params=None, **kwargs):
        if "search/decks" in url:
            self.search_order_by.append(params.get("orderBy"))
            commander = params["commanderName"]
            page = params["page"]
            results = self._pages.get(commander, []) if page == 1 else []
            return _FakeResponse(text=_next_data_html(results))
        deck_id = int(url.rstrip("/").rsplit("/", 1)[-1])
        self.deck_requests.append(deck_id)
        return _FakeResponse(json_data=self._decks[deck_id])


@pytest.fixture
def db_with_cards(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        cards = [
            ("krenko-uid", "Krenko, Mob Boss"),
            ("breya-uid", "Breya, Etherium Shaper"),
            ("jarad-uid", "Jarad, Golgari Lich Lord"),
            ("thrasios-uid", "Thrasios, Triton Hero"),
            ("tymna-uid", "Tymna the Weaver"),
            ("filler-uid", "Filler Land"),
            # The rest of the real roster (edhcut.config.CONFIG.commander_slots) -- run_meta_sample
            # resolves all 5 configured slots (not just Krenko) to compute its roster-exclusion
            # set, so every test exercising it needs the full roster present, not just Krenko.
            ("kyler-uid", "Kyler, Sigardian Emissary"),
            ("yoshimaru-uid", "Yoshimaru, Ever Faithful"),
            ("bruse-tarl-uid", "Bruse Tarl, Boorish Herder"),
            ("yenna-uid", "Yenna, Redtooth Regent"),
            ("orysa-uid", "Orysa, Tide Choreographer"),
        ]
        conn.executemany("INSERT INTO cards (oracle_id, name) VALUES (?, ?)", cards)
        conn.executemany(
            "INSERT INTO card_names (name_normalized, oracle_id) VALUES (?, ?)",
            [(normalize_name(name), oracle_id) for oracle_id, name in cards],
        )
        conn.commit()
    with connect(db_path) as conn:
        yield conn


# --- order_by / cohort threading through harvest_slot -----------------------------------------

def test_harvest_slot_threads_order_by_to_search(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeSession(
        pages={"Breya, Etherium Shaper": [_listing(1)]},
        decks={1: _deck_json(1, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper")},
    )
    harvest_slot(
        conn, session, ["Breya, Etherium Shaper"], {"breya-uid", "filler-uid"},
        decks_per_commander=5, staleness_cutoff_days=730, order_by="-updatedAt", show_progress=False,
    )
    assert session.search_order_by == ["-updatedAt"]


def test_harvest_slot_default_order_by_is_view_count(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeSession(pages={"Krenko, Mob Boss": []}, decks={})
    harvest_slot(
        conn, session, ["Krenko, Mob Boss"], {"krenko-uid"},
        decks_per_commander=5, staleness_cutoff_days=730, show_progress=False,
    )
    assert session.search_order_by == ["-viewCount"]


def test_harvest_slot_stores_cohort(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeSession(
        pages={"Breya, Etherium Shaper": [_listing(1)]},
        decks={1: _deck_json(1, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper")},
    )
    harvest_slot(
        conn, session, ["Breya, Etherium Shaper"], {"breya-uid", "filler-uid"},
        decks_per_commander=5, staleness_cutoff_days=730, cohort="meta_sample", show_progress=False,
    )
    assert conn.execute("SELECT cohort FROM decks WHERE source_id = '1'").fetchone() == ("meta_sample",)


# --- resumability: already_seen_source_ids -----------------------------------------------------

def test_already_seen_source_ids_skips_without_refetching(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeSession(
        pages={"Breya, Etherium Shaper": [_listing(1), _listing(2)]},
        decks={2: _deck_json(2, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper")},
        # deck 1 deliberately absent -- if it were fetched, this would KeyError
    )
    stats = harvest_slot(
        conn, session, ["Breya, Etherium Shaper"], {"breya-uid", "filler-uid"},
        decks_per_commander=1, staleness_cutoff_days=730,
        already_seen_source_ids=frozenset({"1"}), show_progress=False,
    )
    assert session.deck_requests == [2]  # deck 1 never fetched
    assert stats.decks_kept == 1
    assert stats.candidates_checked == 1  # deck 1 didn't count as a checked candidate either


# --- roster-priority upsert guard ---------------------------------------------------------------

def test_upsert_never_downgrades_an_existing_roster_deck(db_with_cards) -> None:
    conn = db_with_cards
    deck = _deck_json(1, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper")

    _upsert_deck(conn, deck, "breya-uid", None, "breya-uid", "roster")
    conn.commit()
    _upsert_deck(conn, deck, "jarad-uid", "tymna-uid", "jarad-uid+tymna-uid", "meta_sample")
    conn.commit()

    row = conn.execute(
        "SELECT cohort, commander_oracle_id, partner_oracle_id, slot_key FROM decks WHERE source_id = '1'"
    ).fetchone()
    assert row == ("roster", "breya-uid", None, "breya-uid")


def test_upsert_updates_normally_between_two_non_roster_cohorts(db_with_cards) -> None:
    conn = db_with_cards
    deck = _deck_json(1, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper")

    _upsert_deck(conn, deck, "breya-uid", None, "breya-uid", "meta_sample")
    conn.commit()
    deck["viewCount"] = 999
    _upsert_deck(conn, deck, "breya-uid", None, "breya-uid", "meta_sample")
    conn.commit()

    assert conn.execute("SELECT views FROM decks WHERE source_id = '1'").fetchone() == (999,)


# --- run_meta_sample orchestration ---------------------------------------------------------------

@pytest.fixture
def db_with_meta_commanders(db_with_cards):
    conn = db_with_cards
    conn.executemany(
        "INSERT INTO meta_commanders "
        "(slot_key, commander_oracle_id, partner_oracle_id, name, color_identity, "
        "edhrec_num_decks, edhrec_rank, sample_target, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("breya-uid", "breya-uid", None, "Breya, Etherium Shaper", "wubr", 21877, 50, 3, "2026-08-09"),
            ("jarad-uid", "jarad-uid", None, "Jarad, Golgari Lich Lord", "bg", 5588, 500, 2, "2026-08-09"),
            (
                "thrasios-uid+tymna-uid", "thrasios-uid", "tymna-uid",
                "Thrasios, Triton Hero // Tymna the Weaver", "wubg", 8000, 100, 2, "2026-08-09",
            ),
        ],
    )
    conn.commit()
    return conn


def test_commander_names_for_resolves_single_and_pair(db_with_meta_commanders) -> None:
    conn = db_with_meta_commanders
    assert _commander_names_for(conn, "breya-uid", None) == ["Breya, Etherium Shaper"]
    assert _commander_names_for(conn, "thrasios-uid", "tymna-uid") == [
        "Thrasios, Triton Hero", "Tymna the Weaver",
    ]


def test_run_meta_sample_harvests_each_commander_to_its_own_target(db_with_meta_commanders) -> None:
    conn = db_with_meta_commanders
    session = FakeSession(
        pages={
            "Breya, Etherium Shaper": [_listing(1), _listing(2), _listing(3)],
            "Jarad, Golgari Lich Lord": [_listing(4), _listing(5)],
            "Thrasios, Triton Hero": [_listing(6), _listing(7)],
        },
        decks={
            1: _deck_json(1, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper"),
            2: _deck_json(2, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper"),
            3: _deck_json(3, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper"),
            4: _deck_json(4, commander_uid="jarad-uid", commander_name="Jarad, Golgari Lich Lord"),
            5: _deck_json(5, commander_uid="jarad-uid", commander_name="Jarad, Golgari Lich Lord"),
            6: {**_deck_json(6, commander_uid="thrasios-uid", commander_name="Thrasios, Triton Hero"),
                "cards": [
                    {"categories": ["Commander"], "quantity": 1, "card": {"oracleCard": {"uid": "thrasios-uid", "name": "Thrasios, Triton Hero"}}},
                    {"categories": ["Commander"], "quantity": 1, "card": {"oracleCard": {"uid": "tymna-uid", "name": "Tymna the Weaver"}}},
                    {"categories": ["Land"], "quantity": 98, "card": {"oracleCard": {"uid": "filler-uid", "name": "Filler Land"}}},
                ]},
            7: {**_deck_json(7, commander_uid="thrasios-uid", commander_name="Thrasios, Triton Hero"),
                "cards": [
                    {"categories": ["Commander"], "quantity": 1, "card": {"oracleCard": {"uid": "thrasios-uid", "name": "Thrasios, Triton Hero"}}},
                    {"categories": ["Commander"], "quantity": 1, "card": {"oracleCard": {"uid": "tymna-uid", "name": "Tymna the Weaver"}}},
                    {"categories": ["Land"], "quantity": 98, "card": {"oracleCard": {"uid": "filler-uid", "name": "Filler Land"}}},
                ]},
        },
    )

    results = run_meta_sample(conn, session, show_progress=False)

    assert {r.name: r.total_kept for r in results} == {
        "Breya, Etherium Shaper": 3,
        "Jarad, Golgari Lich Lord": 2,
        "Thrasios, Triton Hero // Tymna the Weaver": 2,
    }
    assert all(r.shortfall == 0 for r in results)
    assert conn.execute("SELECT COUNT(*) FROM decks WHERE cohort = 'meta_sample'").fetchone()[0] == 7
    # every search used -updatedAt, the meta-sample default
    assert set(session.search_order_by) == {"-updatedAt"}


def test_run_meta_sample_is_resumable_across_two_invocations(db_with_meta_commanders) -> None:
    conn = db_with_meta_commanders
    # First run: only enough candidates for Breya to reach 2/3.
    session1 = FakeSession(
        pages={"Breya, Etherium Shaper": [_listing(1), _listing(2)], "Jarad, Golgari Lich Lord": [],
               "Thrasios, Triton Hero": []},
        decks={
            1: _deck_json(1, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper"),
            2: _deck_json(2, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper"),
        },
    )
    run_meta_sample(conn, session1, limit=1, show_progress=False)
    assert conn.execute("SELECT COUNT(*) FROM decks WHERE slot_key = 'breya-uid'").fetchone()[0] == 2

    # Second run: the same 2 decks resurface first (as they would in reality, still near the top
    # of -updatedAt ordering) plus one genuinely new one. Resumability must not double-count or
    # re-fetch the first two, and must stop after exactly 1 new keep (target 3, 2 already kept).
    session2 = FakeSession(
        pages={"Breya, Etherium Shaper": [_listing(1), _listing(2), _listing(3)],
               "Jarad, Golgari Lich Lord": [], "Thrasios, Triton Hero": []},
        decks={3: _deck_json(3, commander_uid="breya-uid", commander_name="Breya, Etherium Shaper")},
    )
    run_meta_sample(conn, session2, limit=1, show_progress=False)

    assert session2.deck_requests == [3]  # 1 and 2 skipped entirely, never refetched
    assert conn.execute("SELECT COUNT(*) FROM decks WHERE slot_key = 'breya-uid'").fetchone()[0] == 3


def test_run_meta_sample_skips_commander_already_at_target(db_with_meta_commanders) -> None:
    conn = db_with_meta_commanders
    conn.executemany(
        "INSERT INTO decks (source, source_id, commander_oracle_id, slot_key, cohort) VALUES (?, ?, ?, ?, ?)",
        [("archidekt", str(i), "breya-uid", "breya-uid", "meta_sample") for i in (1, 2, 3)],
    )
    conn.commit()

    class _ExplodingSession:
        def request(self, *a, **k):
            raise AssertionError("should never be called for an already-complete commander")

    results = run_meta_sample(conn, _ExplodingSession(), limit=1, show_progress=False)

    assert results[0].harvest is None
    assert results[0].already_kept == 3
    assert results[0].shortfall == 0


def test_run_meta_sample_continues_past_a_commander_error(db_with_meta_commanders) -> None:
    conn = db_with_meta_commanders

    class _BrokenThenWorkingSession:
        def __init__(self):
            self.calls = 0

        def request(self, method, url, params=None, **kwargs):
            self.calls += 1
            if "search/decks" in url and params.get("commanderName") == "Breya, Etherium Shaper":
                raise ConnectionError("simulated network failure")
            return _FakeResponse(text=_next_data_html([]))

    results = run_meta_sample(conn, _BrokenThenWorkingSession(), show_progress=False)

    breya_result = next(r for r in results if r.name == "Breya, Etherium Shaper")
    assert breya_result.harvest.error is not None
    # the other two commanders were still processed, not aborted
    other_names = {r.name for r in results if r.name != "Breya, Etherium Shaper"}
    assert other_names == {"Jarad, Golgari Lich Lord", "Thrasios, Triton Hero // Tymna the Weaver"}


def test_run_meta_sample_limit_caps_commanders_processed(db_with_meta_commanders) -> None:
    conn = db_with_meta_commanders
    session = FakeSession(pages={}, decks={})

    results = run_meta_sample(conn, session, limit=1, show_progress=False)

    # edhrec_rank ascending -> Breya (rank 50) is the only one processed
    assert [r.name for r in results] == ["Breya, Etherium Shaper"]


def test_run_meta_sample_skips_roster_commanders_entirely(db_with_meta_commanders) -> None:
    """A roster commander that also clears the meta-sample threshold (real case: Krenko, Kyler,
    Yenna) must never be harvested a second time under cohort='meta_sample' -- it keeps its
    existing, much larger roster corpus at full resolution instead (EDHCut_PLAN.md task 5.7)."""
    conn = db_with_meta_commanders
    conn.execute(
        "INSERT INTO meta_commanders "
        "(slot_key, commander_oracle_id, partner_oracle_id, name, color_identity, "
        "edhrec_num_decks, edhrec_rank, sample_target, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("krenko-uid", "krenko-uid", None, "Krenko, Mob Boss", "r", 42881, 5, 23, "2026-08-09"),
    )
    conn.commit()

    class _ExplodingIfCalledForKrenko:
        def request(self, method, url, params=None, **kwargs):
            if "search/decks" in url and params.get("commanderName") == "Krenko, Mob Boss":
                raise AssertionError("a roster commander must never be searched by run_meta_sample")
            return _FakeResponse(text=_next_data_html([]))

    results = run_meta_sample(conn, _ExplodingIfCalledForKrenko(), show_progress=False)

    assert "Krenko, Mob Boss" not in {r.name for r in results}
    assert conn.execute("SELECT COUNT(*) FROM decks WHERE cohort = 'meta_sample'").fetchone()[0] == 0
