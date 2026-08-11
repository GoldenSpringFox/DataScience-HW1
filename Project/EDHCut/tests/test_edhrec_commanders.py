"""edhrec_commanders ingest tests against a fake session -- no network, no real HTTP cache."""

import json

import pytest

from edhcut.db import connect
from edhcut.ingest.edhrec_commanders import (
    COLOR_IDENTITIES,
    MIN_DECKS_THRESHOLD,
    CommanderListing,
    build_commander_pool,
    extract_commander_listings,
    fetch_identity_page,
    run,
    sample_target,
    split_commander_name,
)
from edhcut.ingest.scryfall import normalize_name


class _FakeResponse:
    def __init__(self, *, text: str = ""):
        self.text = text

    def raise_for_status(self):
        pass


def _cardview(name: str, num_decks: int) -> dict:
    return {"name": name, "num_decks": num_decks}


def _identity_payload(cardviews: list[dict]) -> dict:
    return {"container": {"json_dict": {"cardlists": [{"tag": "past2years", "cardviews": cardviews}]}}}


def _identity_page_html(payload: dict) -> str:
    next_data = {"props": {"pageProps": {"data": payload}}}
    return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(next_data)}</script>'


class FakeIdentitySession:
    """Keyed by colour-identity slug (`/commanders/<slug>`), unlike edhrec.py's fake session
    which keys on a `commanderName` query param -- this module hits a different URL shape."""

    def __init__(self, pages: dict[str, dict], *, broken_slugs: set[str] | None = None):
        self._pages = pages
        self._broken_slugs = broken_slugs or set()
        self.requested_slugs: list[str] = []

    def request(self, method, url, params=None, **kwargs):
        slug = url.rsplit("/", 1)[-1]
        self.requested_slugs.append(slug)
        if slug in self._broken_slugs:
            raise ConnectionError("simulated network failure")
        return _FakeResponse(text=_identity_page_html(self._pages[slug]))


# --- extract_commander_listings ---------------------------------------------------------------

def test_extract_commander_listings_skips_entries_missing_name_or_num_decks() -> None:
    payload = _identity_payload([
        _cardview("Krenko, Mob Boss", 42881),
        {"name": "No Deck Count"},
        {"num_decks": 100},
    ])
    listings = extract_commander_listings(payload, color_identity="r")
    assert listings == [CommanderListing(name="Krenko, Mob Boss", num_decks=42881, rank=None, color_identity="r")]


# --- split_commander_name -----------------------------------------------------------------------

def test_split_commander_name_splits_and_strips_partner_pair() -> None:
    assert split_commander_name("Thrasios, Triton Hero // Tymna the Weaver") == [
        "Thrasios, Triton Hero", "Tymna the Weaver",
    ]


# --- sample_target -----------------------------------------------------------------------------

def test_sample_target_peak_commander_gets_the_max() -> None:
    assert sample_target(1000, 1000) == 25


def test_sample_target_clipped_to_floor_for_small_share() -> None:
    # sqrt(1/1000) * 25 ~= 0.79 -> rounds well under the floor
    assert sample_target(1, 1000) == 5


def test_sample_target_zero_max_decks_returns_floor() -> None:
    assert sample_target(0, 0) == 5


def test_sample_target_sqrt_not_proportional() -> None:
    # A commander at 1/4 the peak's decks gets *more* than 1/4 the peak's allocation --
    # sqrt(0.25) = 0.5, not 0.25 -- the whole point of using sqrt over proportional.
    quarter_share_target = sample_target(250, 1000)
    assert quarter_share_target == round(25 * 0.5)
    assert quarter_share_target > round(25 * 0.25)


# --- build_commander_pool -----------------------------------------------------------------------

def _all_identity_pages(overrides: dict[str, dict]) -> dict[str, dict]:
    """Every configured slug gets an empty page unless overridden -- keeps tests from having to
    stub all 32 identities."""
    pages = {slug: _identity_payload([]) for slug, _ in COLOR_IDENTITIES}
    pages.update(overrides)
    return pages


def test_build_commander_pool_filters_by_threshold_and_ranks_descending() -> None:
    session = FakeIdentitySession(_all_identity_pages({
        "r": _identity_payload([_cardview("Krenko, Mob Boss", 42881), _cardview("Too Small", 100)]),
        "g": _identity_payload([_cardview("Some Green Commander", 5000)]),
    }))

    result = build_commander_pool(session, threshold=MIN_DECKS_THRESHOLD)

    assert not result.failed_identities
    assert [listing.name for listing in result.listings] == ["Krenko, Mob Boss", "Some Green Commander"]
    assert session.requested_slugs == [slug for slug, _ in COLOR_IDENTITIES]


def test_build_commander_pool_raises_on_duplicate_name_across_identities() -> None:
    session = FakeIdentitySession(_all_identity_pages({
        "r": _identity_payload([_cardview("Weird Duplicate", 5000)]),
        "g": _identity_payload([_cardview("Weird Duplicate", 5000)]),
    }))

    with pytest.raises(RuntimeError, match="more than one colour-identity page"):
        build_commander_pool(session, threshold=MIN_DECKS_THRESHOLD)


def test_build_commander_pool_flags_truncation_when_page_cap_hit_all_above_threshold() -> None:
    # 100 rows returned, every single one clears the threshold -- there might have been more.
    full_page = _identity_payload([_cardview(f"Commander {i}", 5000) for i in range(100)])
    session = FakeIdentitySession(_all_identity_pages({"r": full_page}))

    result = build_commander_pool(session, threshold=MIN_DECKS_THRESHOLD)

    assert result.truncated_identities == ["r"]


def test_build_commander_pool_not_flagged_when_page_cap_hit_but_tail_below_threshold() -> None:
    # 100 rows, but the threshold filter drops some -- not a truncation signal.
    cardviews = [_cardview(f"Commander {i}", 5000) for i in range(50)]
    cardviews += [_cardview(f"Small {i}", 100) for i in range(50)]
    full_page = _identity_payload(cardviews)
    session = FakeIdentitySession(_all_identity_pages({"r": full_page}))

    result = build_commander_pool(session, threshold=MIN_DECKS_THRESHOLD)

    assert result.truncated_identities == []


def test_build_commander_pool_records_failed_identity_without_aborting_others() -> None:
    session = FakeIdentitySession(
        _all_identity_pages({"g": _identity_payload([_cardview("Some Green Commander", 5000)])}),
        broken_slugs={"r"},
    )

    result = build_commander_pool(session, threshold=MIN_DECKS_THRESHOLD)

    assert len(result.failed_identities) == 1
    assert result.failed_identities[0][0] == "r"
    assert [listing.name for listing in result.listings] == ["Some Green Commander"]


# --- run() end-to-end -----------------------------------------------------------------------

@pytest.fixture
def db_with_cards(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO cards (oracle_id, name) VALUES (?, ?)",
            [("krenko-uid", "Krenko, Mob Boss"), ("green-uid", "Some Green Commander")],
        )
        conn.executemany(
            "INSERT INTO card_names (name_normalized, oracle_id) VALUES (?, ?)",
            [
                (normalize_name("Krenko, Mob Boss"), "krenko-uid"),
                (normalize_name("Some Green Commander"), "green-uid"),
            ],
        )
        conn.commit()
    with connect(db_path) as conn:
        yield conn


def test_run_writes_rows_ranked_with_sample_targets_and_logs_unresolved(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeIdentitySession(_all_identity_pages({
        "r": _identity_payload([_cardview("Krenko, Mob Boss", 42881)]),
        "g": _identity_payload([_cardview("Some Green Commander", 5000), _cardview("Unresolvable Commander", 3000)]),
    }))

    stats = run(conn, session, threshold=MIN_DECKS_THRESHOLD)

    assert stats.written == 2
    assert stats.unresolved_names == 1
    assert stats.unresolved_name_samples == ["Unresolvable Commander"]

    rows = conn.execute(
        "SELECT commander_oracle_id, partner_oracle_id, edhrec_num_decks, edhrec_rank, sample_target "
        "FROM meta_commanders ORDER BY edhrec_rank"
    ).fetchall()
    assert rows[0][:4] == ("krenko-uid", None, 42881, 1)
    assert rows[0][4] == 25  # the pool's most-played commander gets the max allocation
    assert rows[1][:4] == ("green-uid", None, 5000, 2)
    assert rows[1][4] < 25


def test_run_resolves_partner_pair_listing_via_slot_key(db_with_cards) -> None:
    conn = db_with_cards
    conn.execute("INSERT INTO cards (oracle_id, name) VALUES ('partner-a-uid', 'Partner Commander A')")
    conn.execute("INSERT INTO cards (oracle_id, name) VALUES ('partner-b-uid', 'Partner Commander B')")
    conn.executemany(
        "INSERT INTO card_names (name_normalized, oracle_id) VALUES (?, ?)",
        [
            (normalize_name("Partner Commander A"), "partner-a-uid"),
            (normalize_name("Partner Commander B"), "partner-b-uid"),
        ],
    )
    conn.commit()
    session = FakeIdentitySession(_all_identity_pages({
        "wu": _identity_payload([_cardview("Partner Commander A // Partner Commander B", 8000)]),
    }))

    stats = run(conn, session, threshold=MIN_DECKS_THRESHOLD)

    assert stats.written == 1
    assert stats.unresolved_names == 0
    row = conn.execute(
        "SELECT slot_key, commander_oracle_id, partner_oracle_id FROM meta_commanders"
    ).fetchone()
    assert row == ("partner-a-uid+partner-b-uid", "partner-a-uid", "partner-b-uid")


def test_run_drops_listing_when_only_one_partner_side_resolves(db_with_cards) -> None:
    conn = db_with_cards
    conn.execute("INSERT INTO cards (oracle_id, name) VALUES ('partner-a-uid', 'Partner Commander A')")
    conn.executemany(
        "INSERT INTO card_names (name_normalized, oracle_id) VALUES (?, ?)",
        [(normalize_name("Partner Commander A"), "partner-a-uid")],
    )
    conn.commit()
    session = FakeIdentitySession(_all_identity_pages({
        "wu": _identity_payload([_cardview("Partner Commander A // Totally Unresolvable Card", 8000)]),
    }))

    stats = run(conn, session, threshold=MIN_DECKS_THRESHOLD)

    assert stats.written == 0
    assert stats.unresolved_names == 1
    assert conn.execute("SELECT COUNT(*) FROM meta_commanders").fetchone()[0] == 0


def test_run_reharvest_replaces_rows_not_duplicates(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeIdentitySession(_all_identity_pages({
        "r": _identity_payload([_cardview("Krenko, Mob Boss", 42881)]),
    }))

    run(conn, session, threshold=MIN_DECKS_THRESHOLD)
    run(conn, session, threshold=MIN_DECKS_THRESHOLD)

    assert conn.execute("SELECT COUNT(*) FROM meta_commanders").fetchone()[0] == 1


def test_run_raises_and_writes_nothing_when_an_identity_fails_to_fetch(db_with_cards) -> None:
    conn = db_with_cards
    session = FakeIdentitySession(
        _all_identity_pages({"g": _identity_payload([_cardview("Some Green Commander", 5000)])}),
        broken_slugs={"r"},
    )

    with pytest.raises(RuntimeError, match="Failed to fetch"):
        run(conn, session, threshold=MIN_DECKS_THRESHOLD)

    assert conn.execute("SELECT COUNT(*) FROM meta_commanders").fetchone()[0] == 0
