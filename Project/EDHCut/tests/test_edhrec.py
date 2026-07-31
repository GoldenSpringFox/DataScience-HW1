"""EDHREC ingest tests against a fake session — no network, no real HTTP cache."""

import json

import pytest

from edhcut.db import connect
from edhcut.ingest.edhrec import (
    BASE_URL,
    GlobalThemeStat,
    ThemeStat,
    commander_slug,
    extract_card_stats,
    extract_global_tags,
    extract_themes,
    fetch_commander_data,
    fetch_global_tag_page,
    format_card_name,
    get_build_id,
    harvest_global_themes,
    harvest_slot,
    run,
)
from edhcut.ingest.scryfall import normalize_name


def _next_data_html(payload: dict) -> str:
    return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'


def _homepage_html(build_id: str) -> str:
    return _next_data_html({"buildId": build_id})


def _cardview(name: str, synergy: float, num_decks: int, potential_decks: int) -> dict:
    return {"name": name, "synergy": synergy, "num_decks": num_decks, "potential_decks": potential_decks}


def _cardlist(tag: str, header: str, cardviews: list[dict]) -> dict:
    return {"tag": tag, "header": header, "cardviews": cardviews}


def _commander_payload(cardlists: list[dict], taglinks: list[dict] | None = None) -> dict:
    return {
        "pageProps": {
            "data": {
                "container": {"json_dict": {"cardlists": cardlists}},
                "panels": {"taglinks": taglinks or []},
            }
        }
    }


def _global_theme_cardview(name: str, num_decks: int, url: str | None = None) -> dict:
    return {"name": name, "num_decks": num_decks, "url": url or f"/tags/{format_card_name(name)}"}


def _global_themes_payload(cardviews: list[dict]) -> dict:
    return {
        "pageProps": {
            "data": {
                "container": {"json_dict": {"cardlists": [
                    {"tag": "tagsbypopularitysort", "header": "Tags By Popularity Sort", "cardviews": cardviews}
                ]}}
            }
        }
    }


_EMPTY_GLOBAL_TAG_PAGE = _global_themes_payload([])


class _FakeResponse:
    def __init__(self, *, json_data=None, text=None):
        self._json = json_data
        self.text = text or ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class FakeEdhrecSession:
    def __init__(
        self,
        homepage_html: str,
        commander_pages: dict[str, dict],
        global_themes_page: dict | None = None,
        global_typal_page: dict | None = None,
    ):
        self._homepage_html = homepage_html
        self._commander_pages = commander_pages
        self._global_themes_page = global_themes_page or _EMPTY_GLOBAL_TAG_PAGE
        self._global_typal_page = global_typal_page or _EMPTY_GLOBAL_TAG_PAGE
        self.requested_slugs: list[str] = []

    def request(self, method, url, params=None, **kwargs):
        if url == BASE_URL:
            return _FakeResponse(text=self._homepage_html)
        if "tags/themes.json" in url:
            return _FakeResponse(json_data=self._global_themes_page)
        if "tags/typal.json" in url:
            return _FakeResponse(json_data=self._global_typal_page)
        slug = (params or {}).get("commanderName")
        self.requested_slugs.append(slug)
        return _FakeResponse(json_data=self._commander_pages[slug])


# --- slug formatting -------------------------------------------------------

def test_format_card_name_matches_edhrec_slug_rules() -> None:
    assert format_card_name("Krenko, Mob Boss") == "krenko-mob-boss"
    assert format_card_name("Yoshimaru, Ever Faithful") == "yoshimaru-ever-faithful"


def test_commander_slug_single_commander() -> None:
    assert commander_slug(["Krenko, Mob Boss"]) == "krenko-mob-boss"


def test_commander_slug_partner_pair_sorted_alphabetically_regardless_of_input_order() -> None:
    expected = "bruse-tarl-boorish-herder-yoshimaru-ever-faithful"
    assert commander_slug(["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"]) == expected
    assert commander_slug(["Bruse Tarl, Boorish Herder", "Yoshimaru, Ever Faithful"]) == expected


# --- build-id discovery ------------------------------------------------------

def test_get_build_id_parses_next_data_script() -> None:
    session = FakeEdhrecSession(homepage_html=_homepage_html("K2qEK1aTLshG5gVnpr2lK"), commander_pages={})
    assert get_build_id(session) == "K2qEK1aTLshG5gVnpr2lK"


def test_get_build_id_raises_when_missing() -> None:
    session = FakeEdhrecSession(homepage_html=_next_data_html({}), commander_pages={})
    with pytest.raises(RuntimeError):
        get_build_id(session)


# --- commander-page fetch + shape validation ---------------------------------

def test_fetch_commander_data_unwraps_page_props() -> None:
    slug = "krenko-mob-boss"
    payload = _commander_payload([_cardlist("creatures", "Creatures", [_cardview("Goblin Warchief", 0.7, 100, 200)])])
    session = FakeEdhrecSession(homepage_html="", commander_pages={slug: payload})
    data = fetch_commander_data(session, "build1", slug)
    assert "container" in data
    assert "panels" in data


def test_fetch_commander_data_raises_when_data_missing() -> None:
    session = FakeEdhrecSession(homepage_html="", commander_pages={"slug": {"pageProps": {}}})
    with pytest.raises(RuntimeError):
        fetch_commander_data(session, "build1", "slug")


def test_fetch_commander_data_raises_when_container_missing() -> None:
    session = FakeEdhrecSession(homepage_html="", commander_pages={"slug": {"pageProps": {"data": {}}}})
    with pytest.raises(RuntimeError):
        fetch_commander_data(session, "build1", "slug")


# --- card stat / theme extraction --------------------------------------------

def test_extract_card_stats_prefers_editorial_tag_over_type_list_regardless_of_page_order() -> None:
    creatures = _cardlist("creatures", "Creatures", [_cardview("Goblin Warchief", 0.72, 37352, 42669)])
    high_synergy = _cardlist("highsynergycards", "High Synergy Cards", [_cardview("Goblin Warchief", 0.72, 37352, 42669)])

    data_a = {"container": {"json_dict": {"cardlists": [creatures, high_synergy]}}, "panels": {}}
    data_b = {"container": {"json_dict": {"cardlists": [high_synergy, creatures]}}, "panels": {}}

    assert extract_card_stats(data_a)["Goblin Warchief"].category == "High Synergy Cards"
    assert extract_card_stats(data_b)["Goblin Warchief"].category == "High Synergy Cards"


def test_extract_card_stats_falls_back_to_type_list_when_no_editorial_tag_present() -> None:
    cardlists = [_cardlist("creatures", "Creatures", [_cardview("Filler Beast", 0.1, 10, 200)])]
    data = {"container": {"json_dict": {"cardlists": cardlists}}, "panels": {}}
    assert extract_card_stats(data)["Filler Beast"].category == "Creatures"


def test_extract_card_stats_keeps_one_row_per_distinct_card_name() -> None:
    cardlists = [
        _cardlist("topcards", "Top Cards", [_cardview("Goblin Warchief", 0.72, 37352, 42669)]),
        _cardlist("creatures", "Creatures", [
            _cardview("Goblin Warchief", 0.72, 37352, 42669),
            _cardview("Krenko's Command", 0.3, 500, 42669),
        ]),
    ]
    data = {"container": {"json_dict": {"cardlists": cardlists}}, "panels": {}}
    stats = extract_card_stats(data)
    assert set(stats) == {"Goblin Warchief", "Krenko's Command"}
    assert stats["Krenko's Command"].category == "Creatures"


def test_extract_themes_reads_taglinks() -> None:
    data = {"panels": {"taglinks": [{"slug": "goblins", "value": "Goblins", "count": 6368}]}}
    assert extract_themes(data) == [ThemeStat(theme="Goblins", num_decks=6368)]


# --- global theme/typal popularity lists --------------------------------------

def test_fetch_global_tag_page_unwraps_page_props() -> None:
    payload = _global_themes_payload([_global_theme_cardview("Tokens", 196845, "/tags/tokens")])
    session = FakeEdhrecSession(homepage_html="", commander_pages={}, global_themes_page=payload)
    data = fetch_global_tag_page(session, "build1", "theme")
    assert "container" in data


def test_fetch_global_tag_page_uses_typal_route_for_typal_kind() -> None:
    payload = _global_themes_payload([_global_theme_cardview("Goblins", 20000, "/tags/goblins")])
    session = FakeEdhrecSession(homepage_html="", commander_pages={}, global_typal_page=payload)
    data = fetch_global_tag_page(session, "build1", "typal")
    assert "container" in data


def test_fetch_global_tag_page_raises_when_container_missing() -> None:
    session = FakeEdhrecSession(
        homepage_html="", commander_pages={}, global_themes_page={"pageProps": {"data": {}}}
    )
    with pytest.raises(RuntimeError):
        fetch_global_tag_page(session, "build1", "theme")


def test_extract_global_tags_reads_name_url_slug_count_and_tags_kind() -> None:
    data = _global_themes_payload([
        _global_theme_cardview("Tokens", 196845, "/tags/tokens"),
        _global_theme_cardview("+1/+1 Counters", 161727, "/tags/plus-1-plus-1-counters"),
    ])["pageProps"]["data"]
    tags = extract_global_tags(data, kind="theme")
    assert tags == [
        GlobalThemeStat(theme="Tokens", kind="theme", slug="tokens", num_decks=196845),
        GlobalThemeStat(theme="+1/+1 Counters", kind="theme", slug="plus-1-plus-1-counters", num_decks=161727),
    ]


def test_extract_global_tags_tags_typal_entries_as_typal() -> None:
    data = _global_themes_payload([_global_theme_cardview("Goblins", 20000, "/tags/goblins")])["pageProps"]["data"]
    tags = extract_global_tags(data, kind="typal")
    assert tags == [GlobalThemeStat(theme="Goblins", kind="typal", slug="goblins", num_decks=20000)]


def test_extract_global_tags_raises_when_popularity_cardlist_missing() -> None:
    data = {"container": {"json_dict": {"cardlists": [{"tag": "somethingelse", "cardviews": []}]}}}
    with pytest.raises(RuntimeError):
        extract_global_tags(data, kind="theme")


def test_harvest_global_themes_writes_both_kinds_into_one_table(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        themes_payload = _global_themes_payload([
            _global_theme_cardview("Tokens", 196845, "/tags/tokens"),
            _global_theme_cardview("Dandan", 6, "/tags/dandan"),
        ])
        typal_payload = _global_themes_payload([_global_theme_cardview("Goblins", 20000, "/tags/goblins")])
        session = FakeEdhrecSession(
            homepage_html="", commander_pages={},
            global_themes_page=themes_payload, global_typal_page=typal_payload,
        )

        stats = harvest_global_themes(conn, session, "build1")
        assert stats.error is None
        assert stats.themes_written == 2
        assert stats.typal_written == 1
        assert conn.execute("SELECT COUNT(*) FROM edhrec_themes").fetchone()[0] == 3

        theme_row = conn.execute(
            "SELECT kind, slug, num_decks FROM edhrec_themes WHERE theme = 'Tokens'"
        ).fetchone()
        assert theme_row == ("theme", "tokens", 196845)
        typal_row = conn.execute(
            "SELECT kind, slug, num_decks FROM edhrec_themes WHERE theme = 'Goblins'"
        ).fetchone()
        assert typal_row == ("typal", "goblins", 20000)

        # Re-running with shrunk payloads replaces wholesale, not just upserts what's present.
        smaller_themes = _global_themes_payload([_global_theme_cardview("Tokens", 200000, "/tags/tokens")])
        session2 = FakeEdhrecSession(
            homepage_html="", commander_pages={},
            global_themes_page=smaller_themes, global_typal_page=typal_payload,
        )
        stats2 = harvest_global_themes(conn, session2, "build1")
        assert stats2.themes_written == 1
        assert stats2.typal_written == 1
        assert conn.execute("SELECT COUNT(*) FROM edhrec_themes").fetchone()[0] == 2
        assert conn.execute(
            "SELECT num_decks FROM edhrec_themes WHERE theme = 'Tokens'"
        ).fetchone()[0] == 200000


def test_harvest_global_themes_records_error_without_raising(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        class _BrokenSession:
            def request(self, method, url, params=None, **kwargs):
                raise ConnectionError("simulated network failure")

        stats = harvest_global_themes(conn, _BrokenSession(), "build1")
        assert stats.error is not None
        assert "simulated network failure" in stats.error
        assert stats.themes_written == 0
        assert stats.typal_written == 0


# --- harvest_slot end-to-end --------------------------------------------------

@pytest.fixture
def db_with_cards(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO cards (oracle_id, name) VALUES (?, ?)",
            [
                ("krenko-uid", "Krenko, Mob Boss"),
                ("warchief-uid", "Goblin Warchief"),
                ("yoshi-uid", "Yoshimaru, Ever Faithful"),
                ("bruse-uid", "Bruse Tarl, Boorish Herder"),
            ],
        )
        conn.executemany(
            "INSERT INTO card_names (name_normalized, oracle_id) VALUES (?, ?)",
            [
                (normalize_name("Krenko, Mob Boss"), "krenko-uid"),
                (normalize_name("Goblin Warchief"), "warchief-uid"),
                (normalize_name("Yoshimaru, Ever Faithful"), "yoshi-uid"),
                (normalize_name("Bruse Tarl, Boorish Herder"), "bruse-uid"),
            ],
        )
        conn.commit()
    with connect(db_path) as conn:
        yield conn


def test_harvest_slot_writes_card_stats_and_themes_and_logs_unresolved(db_with_cards) -> None:
    conn = db_with_cards
    slug = "krenko-mob-boss"
    payload = _commander_payload(
        cardlists=[
            _cardlist("highsynergycards", "High Synergy Cards", [_cardview("Goblin Warchief", 0.72, 37352, 42669)]),
            _cardlist("creatures", "Creatures", [
                _cardview("Goblin Warchief", 0.72, 37352, 42669),
                _cardview("Unresolvable Goblin", 0.2, 50, 42669),
            ]),
        ],
        taglinks=[{"slug": "goblins", "value": "Goblins", "count": 6368}],
    )
    session = FakeEdhrecSession(homepage_html="", commander_pages={slug: payload})

    stats = harvest_slot(conn, session, "build1", ["Krenko, Mob Boss"])

    assert stats.error is None
    assert stats.cards_written == 1
    assert stats.themes_written == 1
    assert stats.unresolved_names == 1
    assert stats.unresolved_name_samples == ["Unresolvable Goblin"]

    row = conn.execute(
        "SELECT oracle_id, inclusion_rate, synergy_score, num_decks, category "
        "FROM edhrec_card_stats WHERE commander_key = ?",
        ("krenko-uid",),
    ).fetchone()
    assert row[0] == "warchief-uid"
    assert row[1] == pytest.approx(37352 / 42669)
    assert row[2] == pytest.approx(0.72)
    assert row[3] == 37352
    assert row[4] == "High Synergy Cards"

    theme_row = conn.execute(
        "SELECT theme, num_decks FROM edhrec_themes_per_commander WHERE commander_key = ?",
        ("krenko-uid",),
    ).fetchone()
    assert theme_row == ("Goblins", 6368)


def test_harvest_slot_partner_pair_uses_sorted_slug_and_config_ordered_commander_key(db_with_cards) -> None:
    conn = db_with_cards
    slug = "bruse-tarl-boorish-herder-yoshimaru-ever-faithful"
    payload = _commander_payload([_cardlist("creatures", "Creatures", [_cardview("Goblin Warchief", 0.1, 10, 222)])])
    session = FakeEdhrecSession(homepage_html="", commander_pages={slug: payload})

    stats = harvest_slot(conn, session, "build1", ["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"])

    assert stats.error is None
    assert session.requested_slugs == [slug]
    # commander_key follows config order (yoshi then bruse), independent of the slug's own
    # alphabetical ordering (bruse then yoshi) — two separate identifiers, not required to match.
    row = conn.execute("SELECT commander_key FROM edhrec_card_stats").fetchone()
    assert row[0] == "yoshi-uid+bruse-uid"


def test_harvest_slot_reharvest_replaces_rows_not_duplicates(db_with_cards) -> None:
    conn = db_with_cards
    slug = "krenko-mob-boss"
    payload = _commander_payload(
        [_cardlist("creatures", "Creatures", [_cardview("Goblin Warchief", 0.5, 10, 100)])],
        taglinks=[{"slug": "goblins", "value": "Goblins", "count": 10}],
    )
    session = FakeEdhrecSession(homepage_html="", commander_pages={slug: payload})

    harvest_slot(conn, session, "build1", ["Krenko, Mob Boss"])
    harvest_slot(conn, session, "build1", ["Krenko, Mob Boss"])

    assert conn.execute(
        "SELECT COUNT(*) FROM edhrec_card_stats WHERE commander_key = 'krenko-uid'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM edhrec_themes_per_commander WHERE commander_key = 'krenko-uid'"
    ).fetchone()[0] == 1


def test_harvest_slot_records_error_without_raising_on_fetch_failure(db_with_cards) -> None:
    conn = db_with_cards

    class _BrokenSession:
        def request(self, method, url, params=None, **kwargs):
            raise ConnectionError("simulated network failure")

    stats = harvest_slot(conn, _BrokenSession(), "build1", ["Krenko, Mob Boss"])

    assert stats.error is not None
    assert "simulated network failure" in stats.error
    assert stats.cards_written == 0


def test_harvest_slot_records_error_when_commander_itself_unresolved(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        pass
    with connect(db_path) as conn:
        stats = harvest_slot(conn, FakeEdhrecSession("", {}), "build1", ["Nonexistent Commander"])
    assert stats.error is not None
    assert "Nonexistent Commander" in stats.error


def test_run_fetches_build_id_once_refreshes_global_themes_and_writes_ingest_log_per_slot(
    db_with_cards,
) -> None:
    conn = db_with_cards
    krenko_slug = "krenko-mob-boss"
    partner_slug = "bruse-tarl-boorish-herder-yoshimaru-ever-faithful"
    session = FakeEdhrecSession(
        homepage_html=_homepage_html("build1"),
        commander_pages={
            krenko_slug: _commander_payload([_cardlist("creatures", "Creatures", [_cardview("Goblin Warchief", 0.5, 10, 100)])]),
            partner_slug: _commander_payload([_cardlist("creatures", "Creatures", [_cardview("Goblin Warchief", 0.1, 10, 222)])]),
        },
        global_themes_page=_global_themes_payload([_global_theme_cardview("Tokens", 196845, "/tags/tokens")]),
        global_typal_page=_global_themes_payload([_global_theme_cardview("Goblins", 20000, "/tags/goblins")]),
    )
    slots = [["Krenko, Mob Boss"], ["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"]]

    result = run(conn, session, slots=slots, show_progress=False)

    assert result.global_themes.error is None
    assert result.global_themes.themes_written == 1
    assert result.global_themes.typal_written == 1
    assert len(result.slots) == 2
    assert all(stats.error is None for stats in result.slots)
    assert conn.execute("SELECT COUNT(*) FROM ingest_log WHERE source = 'edhrec'").fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM ingest_log WHERE source = 'edhrec_global_themes'"
    ).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM edhrec_themes").fetchone()[0] == 2
