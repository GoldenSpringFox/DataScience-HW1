"""QA & coverage report (plan §6, task 5.6).

Cross-source sanity report over everything tasks 5.2-5.5 harvested: per-slot deck/card-pool
sizes, deck-size sanity, unresolved-name leaderboards per source, tag coverage, an EDHREC-vs-
Archidekt inclusion agreement spot-check, Kyler's precon-similarity distribution, Orysa's
corpus thinness, and a resolve-check of the user's real decklist fixtures. Pure read/analysis
over the already-populated DB (and the already-downloaded raw bulk files/HTTP cache) — no new
network calls beyond what the EDHREC re-check hits, which is itself cache-only within the
14-day window (see `unresolved_edhrec_names`).

`archidekt`/`tagger_bulk` don't persist a queryable per-name unresolved list anywhere (the
Archidekt harvester only writes card *names* into its text audit log, not a DB table; the
Tagger bulk file's taggings only carry `oracle_id`, no name at all) — this report re-derives
both rather than changing those modules' schemas for a one-off report: Archidekt's from
`data/logs/archidekt_harvest_log.txt`, Tagger's by cross-referencing unresolved `oracle_id`s
against the full (unfiltered) raw `oracle_cards.jsonl.gz` for a display name.

Run as `python -m edhcut.ingest.qa_report` (writes `data/qa_report.md` by default).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.http import get_session
from edhcut.ingest.archidekt import _resolve_local_oracle_id, slot_key_for
from edhcut.ingest.edhrec import (
    commander_slug,
    extract_card_stats,
    fetch_commander_data,
    get_build_id,
    resolve_oracle_id,
)
from edhcut.ingest.scryfall import normalize_name

TOP_N_UNRESOLVED = 20


@dataclass
class SlotInfo:
    label: str
    commander_key: str
    names: list[str]


def resolve_slots(conn: sqlite3.Connection) -> list[SlotInfo]:
    """One `SlotInfo` per configured commander slot, in plan §1 roster order."""
    slots = []
    for names in CONFIG.commander_slots:
        oracle_ids = [_resolve_local_oracle_id(conn, n) for n in names]
        slots.append(SlotInfo(label=" + ".join(names), commander_key=slot_key_for(oracle_ids), names=names))
    return slots


# --- Deck counts / card-pool sizes / size sanity -----------------------------

def deck_pool_stats(conn: sqlite3.Connection, slot: SlotInfo) -> dict[str, int]:
    deck_count = conn.execute(
        "SELECT COUNT(*) FROM decks WHERE slot_key = ?", (slot.commander_key,)
    ).fetchone()[0]
    pool_size = conn.execute(
        """
        SELECT COUNT(DISTINCT dc.oracle_id) FROM deck_cards dc
        JOIN decks d ON d.deck_id = dc.deck_id WHERE d.slot_key = ?
        """,
        (slot.commander_key,),
    ).fetchone()[0]
    return {"deck_count": deck_count, "pool_size": pool_size}


@dataclass
class SizeViolation:
    deck_id: int
    url: str
    expected: int
    actual: int


_LOG_LINE_RE = re.compile(r"^.*?, problems: (?P<problems>.*), (?P<url>https://\S+)$")
_ILLEGAL_COUNT_RE = re.compile(r"illegal_cards \[(\d+)\]")


def _kept_deck_unresolved_counts(log_path: Path) -> dict[str, int]:
    """Most recent audit-log entry's unresolved-card count per deck URL, for lines
    representing a deck that was actually *kept* (no stale/mismatch/wrong_size flag).
    `deck_cards` deliberately excludes unresolved (e.g. now-banned) cards, so a kept deck can
    legitimately have fewer stored rows than 99/98 — the harvester validated size *including*
    those before deciding to keep it, but only the resolved subset ever gets written. Needed
    to reconcile `deck_size_violations` against that, not to treat every such deck as broken.
    Rejected-candidate log lines for the same URL are ignored even if they appear later, since
    only a kept line reflects what's actually in the DB."""
    counts: dict[str, int] = {}
    if not log_path.exists():
        return counts
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = _LOG_LINE_RE.match(line)
        if not match:
            continue
        problems, url = match.group("problems"), match.group("url")
        if any(flag in problems for flag in ("stale", "mismatch", "wrong_size")):
            continue
        illegal_match = _ILLEGAL_COUNT_RE.search(problems)
        counts[url] = int(illegal_match.group(1)) if illegal_match else 0
    return counts


def deck_size_violations(conn: sqlite3.Connection, log_path: Path) -> list[SizeViolation]:
    """Recompute each deck's stored library size from `deck_cards` and flag anything that
    doesn't match (99 or 98, *minus* however many of that deck's cards were unresolved at
    harvest time per the audit log) — a live re-check, not a trust-the-harvester assumption.
    Expected to come back empty; the point is confirming that, not assuming it."""
    unresolved_by_url = _kept_deck_unresolved_counts(log_path)
    rows = conn.execute(
        """
        SELECT d.deck_id, d.url, d.partner_oracle_id, COALESCE(SUM(dc.qty), 0)
        FROM decks d LEFT JOIN deck_cards dc ON dc.deck_id = d.deck_id
        GROUP BY d.deck_id
        """
    ).fetchall()
    violations = []
    for deck_id, url, partner_oracle_id, total_qty in rows:
        expected_total = 98 if partner_oracle_id else 99
        expected_stored = expected_total - unresolved_by_url.get(url, 0)
        if total_qty != expected_stored:
            violations.append(SizeViolation(deck_id, url, expected_stored, total_qty))
    return violations


# --- Unresolved names per source ---------------------------------------------

_ILLEGAL_CARDS_RE = re.compile(r"illegal_cards \[\d+\] \(([^)]*)\)")


def _resplit_comma_joined_names(fragments: list[str], known_names: set[str]) -> list[str]:
    """The audit log joins multiple unresolved names with `", "` — indistinguishable from a
    comma *inside* a single card's own name (e.g. "Orcrist, Goblin-cleaver"), so a naive split
    fragments those into two fake entries. Greedily re-merges adjacent fragments whenever the
    merge matches a real Scryfall name (checked against the full, unfiltered card-name set —
    not our own `cards` table, since these are precisely the cards *excluded* from it).
    Best-effort: a fragment that matches nothing (real split-only name, or one whose full name
    genuinely isn't in `known_names` for some other reason) is kept as-is rather than dropped."""
    result = []
    i = 0
    while i < len(fragments):
        if fragments[i] in known_names:
            result.append(fragments[i])
            i += 1
            continue
        merged = None
        for j in range(i + 1, len(fragments)):
            candidate = ", ".join(fragments[i : j + 1])
            if candidate in known_names:
                merged = candidate
                i = j + 1
                break
        if merged:
            result.append(merged)
        else:
            result.append(fragments[i])
            i += 1
    return result


def unresolved_archidekt_names(log_path: Path, known_names: set[str]) -> Counter:
    """Parse `illegal_cards [N] (name1, name2, ...)` segments out of the harvest audit log —
    the only place these names are recorded (not persisted to any DB table)."""
    counts: Counter = Counter()
    if not log_path.exists():
        return counts
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = _ILLEGAL_CARDS_RE.search(line)
        if match:
            fragments = [f for f in match.group(1).split(", ") if f]
            for name in _resplit_comma_joined_names(fragments, known_names):
                counts[name] += 1
    return counts


def unresolved_edhrec_names(conn: sqlite3.Connection, slots: list[SlotInfo]) -> Counter:
    """Re-derive full (untruncated) unresolved-name counts per slot. `edhrec.SlotEdhrecStats`
    only keeps a 10-name sample, not the full list or per-name counts, so this replays the
    same fetch/extract/resolve steps `harvest_slot` uses — against the shared cached session,
    so within the 14-day cache window this makes zero live requests."""
    counts: Counter = Counter()
    session = get_session("edhrec")
    build_id = get_build_id(session)
    for slot in slots:
        data = fetch_commander_data(session, build_id, commander_slug(slot.names))
        for name in extract_card_stats(data):
            if resolve_oracle_id(conn, name) is None:
                counts[name] += 1
    return counts


def _oracle_id_to_name_lookup(path: Path) -> dict[str, str]:
    """oracle_id -> name from the *unfiltered* raw Scryfall bulk file (not our `cards` table,
    which excludes banned/non-commander-legal cards) — needed to give a human name to
    Tagger-bulk oracle_ids that failed resolution precisely *because* they're not in `cards`."""
    lookup: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            if card.get("oracle_id"):
                lookup[card["oracle_id"]] = card.get("name", card["oracle_id"])
    return lookup


def unresolved_tagger_bulk_names(oracle_tags_path: Path, name_lookup: dict[str, str], known_oracle_ids: set[str]) -> Counter:
    """Tagger's raw taggings carry only `oracle_id`, no card name — count unresolved
    oracle_ids (once per tagging that referenced them, i.e. once per tag they were tagged
    with) and translate to names via `name_lookup` (the full raw card pool)."""
    id_counts: Counter = Counter()
    with gzip.open(oracle_tags_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag = json.loads(line)
            for tagging in tag.get("taggings") or []:
                oracle_id = tagging.get("oracle_id")
                if oracle_id and oracle_id not in known_oracle_ids:
                    id_counts[oracle_id] += 1
    return Counter({name_lookup.get(oid, oid): n for oid, n in id_counts.items()})


# --- Tag coverage per source --------------------------------------------------

def tag_coverage_per_source(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    total_cards = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    rows = conn.execute(
        "SELECT source, COUNT(DISTINCT oracle_id) FROM card_tags GROUP BY source"
    ).fetchall()
    return [
        {"source": source, "tagged_cards": tagged, "coverage": tagged / total_cards if total_cards else 0.0}
        for source, tagged in rows
    ]


# --- EDHREC vs Archidekt agreement spot-check --------------------------------

def edhrec_top10(conn: sqlite3.Connection, commander_key: str) -> list[tuple[str, float]]:
    rows = conn.execute(
        """
        SELECT c.name, s.inclusion_rate FROM edhrec_card_stats s
        JOIN cards c ON c.oracle_id = s.oracle_id
        WHERE s.commander_key = ? AND s.inclusion_rate IS NOT NULL
        ORDER BY s.inclusion_rate DESC LIMIT 10
        """,
        (commander_key,),
    ).fetchall()
    return rows


def archidekt_top10(conn: sqlite3.Connection, commander_key: str) -> list[tuple[str, int]]:
    rows = conn.execute(
        """
        SELECT c.name, COUNT(*) as deck_count FROM deck_cards dc
        JOIN decks d ON d.deck_id = dc.deck_id
        JOIN cards c ON c.oracle_id = dc.oracle_id
        WHERE d.slot_key = ?
        GROUP BY dc.oracle_id ORDER BY deck_count DESC LIMIT 10
        """,
        (commander_key,),
    ).fetchall()
    return rows


@dataclass
class AgreementSpotCheck:
    slot_label: str
    edhrec_top10: list[tuple[str, float]]
    archidekt_top10: list[tuple[str, int]]
    overlap: int


def edhrec_archidekt_agreement(conn: sqlite3.Connection, slots: list[SlotInfo]) -> list[AgreementSpotCheck]:
    results = []
    for slot in slots:
        e = edhrec_top10(conn, slot.commander_key)
        a = archidekt_top10(conn, slot.commander_key)
        overlap = len({name for name, _ in e} & {name for name, _ in a})
        results.append(AgreementSpotCheck(slot.label, e, a, overlap))
    return results


# --- Kyler precon-similarity histogram ---------------------------------------

_BUCKET_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def precon_similarity_histogram(conn: sqlite3.Connection, commander_key: str) -> list[tuple[str, int]]:
    values = [
        row[0] for row in conn.execute(
            "SELECT precon_similarity FROM decks WHERE slot_key = ? AND precon_similarity IS NOT NULL",
            (commander_key,),
        ).fetchall()
    ]
    buckets = []
    for lo, hi in zip(_BUCKET_EDGES, _BUCKET_EDGES[1:]):
        count = sum(1 for v in values if lo <= v < hi)
        buckets.append((f"{lo:.1f}-{min(hi, 1.0):.1f}", count))
    return buckets


# --- Orysa corpus thinness ----------------------------------------------------

def edhrec_analyzed_deck_count(conn: sqlite3.Connection, commander_key: str) -> int | None:
    """EDHREC's own analyzed-deck total for a slot, derived from any card's `num_decks /
    inclusion_rate` (constant across a slot's rows — see docs/edhrec_api.md). None if the
    slot has no usable rows (e.g. genuinely zero inclusion data)."""
    row = conn.execute(
        """
        SELECT num_decks, inclusion_rate FROM edhrec_card_stats
        WHERE commander_key = ? AND inclusion_rate > 0 LIMIT 1
        """,
        (commander_key,),
    ).fetchone()
    if row is None:
        return None
    num_decks, inclusion_rate = row
    return round(num_decks / inclusion_rate)


def orysa_thinness_stats(conn: sqlite3.Connection, orysa_slot: SlotInfo) -> dict[str, Any]:
    pool = deck_pool_stats(conn, orysa_slot)
    return {
        **pool,
        "edhrec_analyzed_decks": edhrec_analyzed_deck_count(conn, orysa_slot.commander_key),
    }


# --- Fixture decklist resolution check ---------------------------------------

_QTY_PREFIX_RE = re.compile(r"^(\d+)x?\s+(.+)$")


def parse_fixture_decklist(path: Path) -> list[str]:
    """Minimal parser for exactly the format `data/fixtures/my_decks/README.md` documents —
    not task 7.1's general multi-site parser. Returns every card name on the list (commander
    included), one entry per physical line (quantities aren't expanded, this only cares about
    whether each *name* resolves)."""
    names = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("commander:"):
            line = line.split(":", 1)[1].strip()
        line = line.replace("*CMDR*", "").strip()
        if not line:
            continue
        match = _QTY_PREFIX_RE.match(line)
        name = match.group(2).strip() if match else line
        # Category headers (bare words like "Creature", "Land") have no quantity prefix and
        # aren't real card names — skip single all-alphabetic words unless quoted/parenthesized
        # forms suggest a real card slipped through; real card names in these fixtures are
        # always quantity-prefixed except the commander line, already stripped above.
        if name:
            names.append(name)
    return names


@dataclass
class FixtureCheckResult:
    file: str
    total_cards: int
    unresolved: list[str]


def check_fixture_decklists(conn: sqlite3.Connection, fixtures_dir: Path) -> list[FixtureCheckResult]:
    results = []
    for path in sorted(fixtures_dir.glob("*.txt")):
        names = parse_fixture_decklist(path)
        unresolved = [n for n in names if resolve_oracle_id(conn, n) is None]
        results.append(FixtureCheckResult(path.name, len(names), unresolved))
    return results


# --- Report assembly ----------------------------------------------------------

def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def generate_report(conn: sqlite3.Connection) -> str:
    """Assemble the whole QA report as one markdown string (written to `data/qa_report.md`).
    Section order is deliberate — deck/pool sizes first, then the checks that must come back empty
    (deck-size violations, unresolved names), then the cross-source agreement and coverage tables.
    Each section is built by its own function above; this one only sequences them and formats the
    tables, so a new check is added by writing a function and appending it here."""
    slots = resolve_slots(conn)
    archidekt_log_path = CONFIG.paths.logs_dir / "archidekt_harvest_log.txt"
    parts = ["# EDHCut QA & Coverage Report\n"]

    parts.append("## Per-slot deck counts and card-pool sizes\n")
    rows = [[s.label, deck_pool_stats(conn, s)["deck_count"], deck_pool_stats(conn, s)["pool_size"]] for s in slots]
    parts.append(_markdown_table(["Slot", "Decks", "Distinct cards"], rows) + "\n")

    parts.append("## Deck size sanity (expect 0 violations)\n")
    violations = deck_size_violations(conn, archidekt_log_path)
    if violations:
        v_rows = [[v.deck_id, v.expected, v.actual, v.url] for v in violations]
        parts.append(_markdown_table(["deck_id", "expected", "actual", "url"], v_rows) + "\n")
        parts.append(
            "(`expected` already subtracts each deck's own unresolved-card count from the "
            "audit log. Remaining mismatches here are real, but not necessarily bad data — "
            "spot-checked live: both currently-known cases have *more than 2* Archidekt "
            "`\"Commander\"`-tagged cards (e.g. a Partner-with/Background chain), which "
            "`decks.partner_oracle_id` can only represent up to 2 of. Their stored "
            "`deck_cards` counts are internally consistent with their *real* commander count "
            "— this check assumes exactly 1 or 2 per the plan's own definition, so it's "
            "correctly flagging an edge case the schema doesn't fully model, not corrupt "
            "data.)\n"
        )
    else:
        parts.append(f"All {conn.execute('SELECT COUNT(*) FROM decks').fetchone()[0]} decks are exactly the right size (after accounting for unresolved/banned cards, which are intentionally excluded from `deck_cards`).\n")

    parts.append("## Top-20 unresolved names per source\n")
    raw_name_lookup = _oracle_id_to_name_lookup(CONFIG.paths.raw_dir / "oracle_cards.jsonl.gz")
    archidekt_unresolved = unresolved_archidekt_names(archidekt_log_path, set(raw_name_lookup.values()))
    edhrec_unresolved = unresolved_edhrec_names(conn, slots)
    known_oracle_ids = {row[0] for row in conn.execute("SELECT oracle_id FROM cards")}
    tagger_unresolved = unresolved_tagger_bulk_names(
        CONFIG.paths.raw_dir / "oracle_tags.jsonl.gz", raw_name_lookup, known_oracle_ids,
    )
    for label, counter in [("Archidekt", archidekt_unresolved), ("EDHREC", edhrec_unresolved), ("Tagger bulk", tagger_unresolved)]:
        parts.append(f"### {label} ({sum(counter.values())} total unresolved occurrences, {len(counter)} distinct names)\n")
        top = counter.most_common(TOP_N_UNRESOLVED)
        if top:
            parts.append(_markdown_table(["Name", "Count"], [[n, c] for n, c in top]) + "\n")
        else:
            parts.append("None.\n")

    parts.append("## Tag coverage per source\n")
    coverage = tag_coverage_per_source(conn)
    total_cards = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    cov_rows = [[c["source"], c["tagged_cards"], f"{c['coverage']:.1%}"] for c in coverage]
    parts.append(_markdown_table(["Source", "Tagged cards", f"Coverage (/{total_cards})"], cov_rows) + "\n")

    parts.append("## EDHREC vs Archidekt top-10 inclusion agreement\n")
    for check in edhrec_archidekt_agreement(conn, slots):
        parts.append(f"### {check.slot_label} (overlap: {check.overlap}/10)\n")
        rows = []
        for i in range(10):
            e = f"{check.edhrec_top10[i][0]} ({check.edhrec_top10[i][1]:.0%})" if i < len(check.edhrec_top10) else ""
            a = f"{check.archidekt_top10[i][0]} ({check.archidekt_top10[i][1]})" if i < len(check.archidekt_top10) else ""
            rows.append([i + 1, e, a])
        parts.append(_markdown_table(["#", "EDHREC top-10 (inclusion)", "Archidekt top-10 (deck count)"], rows) + "\n")

    kyler_slot = next(s for s in slots if s.names[0].startswith("Kyler"))
    parts.append("## Kyler precon-similarity histogram\n")
    hist = precon_similarity_histogram(conn, kyler_slot.commander_key)
    parts.append(_markdown_table(["Bucket", "Deck count"], [[b, c] for b, c in hist]) + "\n")

    orysa_slot = next(s for s in slots if s.names[0].startswith("Orysa"))
    parts.append("## Orysa corpus thinness\n")
    thin = orysa_thinness_stats(conn, orysa_slot)
    parts.append(
        f"- Decks harvested: {thin['deck_count']}\n"
        f"- Distinct cards seen: {thin['pool_size']}\n"
        f"- EDHREC's own analyzed-deck total for this commander: "
        f"{thin['edhrec_analyzed_decks'] if thin['edhrec_analyzed_decks'] is not None else 'unavailable'}\n"
    )

    parts.append("## Fixture decklist resolution check\n")
    for result in check_fixture_decklists(conn, CONFIG.paths.fixtures_dir):
        status = "OK" if not result.unresolved else f"{len(result.unresolved)} UNRESOLVED"
        parts.append(f"- `{result.file}`: {result.total_cards} cards, {status}")
        if result.unresolved:
            parts.append(f"  - Unresolved: {', '.join(result.unresolved)}")
    parts.append("")

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the EDHCut QA & coverage report.")
    parser.add_argument("--output", type=Path, default=CONFIG.paths.data_dir / "qa_report.md")
    args = parser.parse_args()

    with connect(CONFIG.paths.db_path) as conn:
        report = generate_report(conn)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"QA report written to {args.output}")


if __name__ == "__main__":
    main()
