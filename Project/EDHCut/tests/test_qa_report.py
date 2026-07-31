"""Tests for qa_report.py's pure logic — no network, no gzip fakery (same test-scope
convention as test_tagger_bulk.py: test the parsing/reconciliation logic directly, don't mock
the network/gzip machinery around it)."""

from edhcut.db import connect
from edhcut.ingest.qa_report import (
    _kept_deck_unresolved_counts,
    _markdown_table,
    _resplit_comma_joined_names,
    deck_size_violations,
    parse_fixture_decklist,
    precon_similarity_histogram,
)


# --- comma-joined name resplitting -------------------------------------------

def test_resplit_merges_a_name_that_contains_its_own_comma() -> None:
    known = {"Orcrist, Goblin-cleaver", "Sol Ring"}
    assert _resplit_comma_joined_names(["Orcrist", "Goblin-cleaver"], known) == ["Orcrist, Goblin-cleaver"]


def test_resplit_leaves_ordinary_names_alone() -> None:
    known = {"Sol Ring", "Mana Crypt"}
    assert _resplit_comma_joined_names(["Sol Ring", "Mana Crypt"], known) == ["Sol Ring", "Mana Crypt"]


def test_resplit_handles_mixed_ordinary_and_comma_names() -> None:
    known = {"Sol Ring", "Thorin, Mountain-king", "Karakas"}
    fragments = ["Sol Ring", "Thorin", "Mountain-king", "Karakas"]
    assert _resplit_comma_joined_names(fragments, known) == ["Sol Ring", "Thorin, Mountain-king", "Karakas"]


def test_resplit_falls_back_to_original_fragment_when_no_merge_matches() -> None:
    # Neither "Made Up Card" nor any merge with it is in `known` — best-effort, kept as-is.
    known = {"Sol Ring"}
    assert _resplit_comma_joined_names(["Made Up Card"], known) == ["Made Up Card"]


# --- deck-size reconciliation against the audit log --------------------------

def test_kept_deck_unresolved_counts_reads_illegal_count_for_kept_lines(tmp_path) -> None:
    log = tmp_path / "log.txt"
    log.write_text(
        "Krenko, Mob Boss, problems: illegal_cards [2] (Mana Crypt, Jeweled Lotus), "
        "https://archidekt.com/decks/1/\n"
        "Krenko, Mob Boss, problems: none, https://archidekt.com/decks/2/\n",
        encoding="utf-8",
    )
    counts = _kept_deck_unresolved_counts(log)
    assert counts == {
        "https://archidekt.com/decks/1/": 2,
        "https://archidekt.com/decks/2/": 0,
    }


def test_kept_deck_unresolved_counts_ignores_rejected_candidate_lines(tmp_path) -> None:
    log = tmp_path / "log.txt"
    log.write_text(
        "Krenko, Mob Boss, problems: wrong_size, illegal_cards [1] (Foo), "
        "https://archidekt.com/decks/1/\n"
        "Krenko, Mob Boss, problems: stale, https://archidekt.com/decks/2/\n"
        "Krenko, Mob Boss, problems: mismatch, https://archidekt.com/decks/3/\n",
        encoding="utf-8",
    )
    assert _kept_deck_unresolved_counts(log) == {}


def test_kept_deck_unresolved_counts_missing_file_returns_empty(tmp_path) -> None:
    assert _kept_deck_unresolved_counts(tmp_path / "does_not_exist.txt") == {}


def test_deck_size_violations_accounts_for_unresolved_cards_not_stored(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    log = tmp_path / "log.txt"
    log.write_text(
        "Krenko, Mob Boss, problems: illegal_cards [1] (Mana Crypt), "
        "https://archidekt.com/decks/42/\n",
        encoding="utf-8",
    )
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cards (oracle_id, name) VALUES (?, ?)", ("krenko-uid", "Krenko, Mob Boss")
        )
        conn.execute(
            "INSERT INTO cards (oracle_id, name) VALUES (?, ?)", ("sol-ring-uid", "Sol Ring")
        )
        conn.execute(
            "INSERT INTO decks (deck_id, source, source_id, url, commander_oracle_id, "
            "partner_oracle_id, slot_key) VALUES (42, 'archidekt', '42', "
            "'https://archidekt.com/decks/42/', 'krenko-uid', NULL, 'krenko-uid')"
        )
        # 98 stored cards exactly matches the reconciled expectation (99 - 1 unresolved) —
        # should NOT be flagged, even though it's short of the naive 99 target.
        conn.executemany(
            "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, ?)",
            [(42, "sol-ring-uid", 98)],
        )
        conn.commit()

        assert deck_size_violations(conn, log) == []


def test_deck_size_violations_flags_a_real_mismatch(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    log = tmp_path / "log.txt"
    log.write_text("Krenko, Mob Boss, problems: none, https://archidekt.com/decks/7/\n", encoding="utf-8")
    with connect(db_path) as conn:
        conn.execute("INSERT INTO cards (oracle_id, name) VALUES (?, ?)", ("krenko-uid", "Krenko, Mob Boss"))
        conn.execute("INSERT INTO cards (oracle_id, name) VALUES (?, ?)", ("sol-ring-uid", "Sol Ring"))
        conn.execute(
            "INSERT INTO decks (deck_id, source, source_id, url, commander_oracle_id, "
            "partner_oracle_id, slot_key) VALUES (7, 'archidekt', '7', "
            "'https://archidekt.com/decks/7/', 'krenko-uid', NULL, 'krenko-uid')"
        )
        conn.execute("INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (7, 'sol-ring-uid', 90)")
        conn.commit()

        violations = deck_size_violations(conn, log)
        assert len(violations) == 1
        assert violations[0].expected == 99
        assert violations[0].actual == 90


# --- fixture decklist parsing -------------------------------------------------

def test_parse_fixture_decklist_strips_commander_prefix_and_quantities(tmp_path) -> None:
    path = tmp_path / "deck.txt"
    path.write_text(
        "Commander: 1 Kyler, Sigardian Emissary\n"
        "\n"
        "1 Sol Ring\n"
        "4 Forest\n"
        "# a comment\n"
        "\n"
        "1x Command Tower\n",
        encoding="utf-8",
    )
    assert parse_fixture_decklist(path) == [
        "Kyler, Sigardian Emissary", "Sol Ring", "Forest", "Command Tower",
    ]


def test_parse_fixture_decklist_handles_cmdr_marker(tmp_path) -> None:
    path = tmp_path / "deck.txt"
    path.write_text("1 Yoshimaru, Ever Faithful *CMDR*\n1 Sol Ring\n", encoding="utf-8")
    names = parse_fixture_decklist(path)
    assert "Sol Ring" in names
    assert any("Yoshimaru" in n for n in names)


# --- precon-similarity histogram bucketing -----------------------------------

def test_precon_similarity_histogram_buckets_correctly(tmp_path) -> None:
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        conn.execute("INSERT INTO cards (oracle_id, name) VALUES ('kyler-uid', 'Kyler, Sigardian Emissary')")
        rows = [
            (i, "archidekt", str(i), f"https://x/{i}", "kyler-uid", None, "kyler-uid", sim)
            for i, sim in enumerate([0.05, 0.15, 0.15, 0.95, 1.0])
        ]
        conn.executemany(
            "INSERT INTO decks (deck_id, source, source_id, url, commander_oracle_id, "
            "partner_oracle_id, slot_key, precon_similarity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()

        hist = dict(precon_similarity_histogram(conn, "kyler-uid"))
        assert hist["0.0-0.1"] == 1
        assert hist["0.1-0.2"] == 2
        assert hist["0.9-1.0"] == 2  # both 0.95 and the exact 1.0 edge land in the last bucket
        assert sum(hist.values()) == 5


# --- markdown table helper ----------------------------------------------------

def test_markdown_table_formats_headers_and_rows() -> None:
    table = _markdown_table(["A", "B"], [[1, "x"], [2, "y"]])
    lines = table.splitlines()
    assert lines[0] == "| A | B |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | x |"
    assert lines[3] == "| 2 | y |"
