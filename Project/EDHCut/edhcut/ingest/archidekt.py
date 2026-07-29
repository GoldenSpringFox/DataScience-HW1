"""Archidekt deck harvester (plan §2.1/§6 task 5.3-B).

See `docs/archidekt_api.md` for the full API investigation this is built on: deck-by-id
(`GET /api/decks/{id}/`) is public; the "clean" search API is broken for external callers, so
search goes through the public SSR page's embedded `__NEXT_DATA__` payload instead; Archidekt
reuses Scryfall's `oracle_id` verbatim as `card.card.oracleCard.uid`, so no fuzzy name
matching is needed here (unlike task 7.1's free-text decklist parser). Note: cards that are
banned/restricted in Commander are intentionally absent from our own `cards` table (task 5.2
filters to `legalities.commander == "legal"`), so decks containing them will show those cards
as "unresolved" here — that's the filter working as intended, not a resolution failure.

Filters applied before a deck is kept: must be Commander format, updated within
`config.deck_staleness_cutoff_days` (skipped using search-listing data, before even fetching
the deck), the searched commander(s) must be confirmed via the `"Commander"` category once
fetched, and the deck must have *exactly* the right physical card count — 99 library cards
for a single commander or 98 for a partner pair (100 total including commander(s)). Anything
failing that last check is silently excluded, not flagged — an incomplete/WIP list, not an
anomaly worth review.

Run as `python -m edhcut.ingest.archidekt` (all configured commander slots) or
`python -m edhcut.ingest.archidekt --slot N` (a single 0-indexed slot from
`config.commander_slots`). Idempotent: `decks` upserts by `(source, source_id)`;
`deck_cards` is fully replaced per deck on each (re-)harvest so cards a user later cut from
their list don't linger (see devlog for why this differs from task 5.2's card upsert choice).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

from tqdm import tqdm

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.http import RateLimitedSession, get_session, request_with_retry
from edhcut.ingest.scryfall import normalize_name

SEARCH_URL = "https://archidekt.com/search/decks"
DECK_URL_TEMPLATE = "https://archidekt.com/api/decks/{deck_id}/"
COMMANDER_FORMAT = 3

# Flag a kept deck if more than this fraction of its cards failed oracle_id resolution
# (mostly expected to be banned/restricted cards, but a high ratio can mean something else
# is going on, e.g. an un-set/joke deck).
UNRESOLVED_RATIO_FLAG_THRESHOLD = 0.15

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)


def _parse_next_data(html: str) -> dict[str, Any]:
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise RuntimeError("__NEXT_DATA__ script tag not found in Archidekt search page")
    return json.loads(match.group(1))


def _parse_archidekt_datetime(value: str) -> datetime:
    # Archidekt timestamps look like "2026-07-29T13:01:35.185237Z" — fromisoformat handles
    # the trailing "Z" directly on Python >=3.11.
    return datetime.fromisoformat(value)


def search_deck_listings(
    session: RateLimitedSession,
    primary_commander: str,
    other_partner: str | None = None,
    *,
    order_by: str = "-viewCount",
) -> Iterator[dict[str, Any]]:
    """Yield candidate deck *listing* dicts, paging through Archidekt's search page ourselves.

    Listings carry enough metadata (`id`, `updatedAt`, `viewCount`, ...) to pre-filter (e.g.
    for staleness) before spending a request on the full deck detail.

    `other_partner`, if given, adds `cardName=` to pre-filter to decks containing that card
    somewhere (not necessarily as a declared commander — caller must still verify via the
    deck's own `"Commander"` category once fetched; see docs/archidekt_api.md).
    """
    page = 1
    while True:
        params: dict[str, Any] = {
            "commanderName": primary_commander,
            "deckFormat": COMMANDER_FORMAT,
            "orderBy": order_by,
            "page": page,
        }
        if other_partner:
            params["cardName"] = other_partner

        response = request_with_retry(session, "GET", SEARCH_URL, params=params)
        deck_results = _parse_next_data(response.text)["props"]["pageProps"]["deckResults"]
        results = deck_results["results"]
        if not results:
            return
        yield from results
        if not deck_results.get("next"):
            return
        page += 1


def fetch_deck(session: RateLimitedSession, deck_id: int) -> dict[str, Any]:
    response = request_with_retry(session, "GET", DECK_URL_TEMPLATE.format(deck_id=deck_id))
    return response.json()


def is_token(card: dict[str, Any]) -> bool:
    return card["card"]["oracleCard"].get("layout") == "token"


# Archidekt doesn't expose a trustworthy computed "how many cards are really in this deck"
# number anywhere we can reach: the search-listing `size` field can be stale relative to the
# deck's current state (confirmed live — a listing said 101 for a deck whose own page
# currently shows "Size: 100"), the deck-detail API has no such field at all, and the page's
# own render is computed client-side from this same category data by JS we don't have access
# to. So board membership is derived from categories instead — but only a card's *first*
# category counts. Later categories in the list are just personal-organization tags layered
# on top and don't affect whether the card counts toward the deck (confirmed against 2 real
# decks that were wrongly shrunk by an earlier version of this rule that checked *all* of a
# card's categories: e.g. `["Goblin", "Maybeboard"]` — real deck inclusion, "Maybeboard" is
# just a second tag — vs. `["Maybeboard", "Ramp"]` — genuinely benched, "Maybeboard" is
# first).
#
# Within "first category", one name is hardcoded rather than flag-driven: the built-in
# `"Sideboard"` category is *always* excluded regardless of its own `includedInDeck` flag.
# Confirmed live on multiple decks where the API reports `includedInDeck: true` for
# "Sideboard" (seemingly always true — same as "Commander"), yet cards whose first category
# is "Sideboard" never show up in the deck's own displayed size or stats. "Maybeboard" is
# NOT special-cased the same way — it's an ordinary user-manageable category like any other,
# and its own `includedInDeck` flag is authoritative: one real deck had "Maybeboard" set to
# `includedInDeck: true` and its first-tagged cards genuinely counted toward the 100 (this
# overturned an earlier version of this rule that hardcoded "maybeboard" out too — that
# version happened to work on the decks tested at the time, but broke on this one).
_HARDCODED_EXCLUDED_FIRST_CATEGORY = "Sideboard"


def included_cards(deck: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Cards actually in the deck's list.

    A card belongs if it has no categories, or its *first* category is one of the deck's
    `includedInDeck: true` categories — except the built-in "Sideboard" category, which is
    always excluded regardless of its own flag (see `_HARDCODED_EXCLUDED_FIRST_CATEGORY`).
    Token-tracking entries are dropped outright regardless of category (`is_token`).
    """
    included_categories = {
        category["name"]
        for category in deck.get("categories", [])
        if category.get("includedInDeck", False)
    }
    for card in deck.get("cards", []):
        if is_token(card):
            continue
        categories = card.get("categories") or []
        if not categories:
            yield card
            continue
        first_category = categories[0]
        if first_category == _HARDCODED_EXCLUDED_FIRST_CATEGORY:
            continue
        if first_category in included_categories:
            yield card


def commander_cards(deck: dict[str, Any]) -> list[dict[str, Any]]:
    return [card for card in deck.get("cards", []) if "Commander" in (card.get("categories") or [])]


def card_oracle_id(card: dict[str, Any]) -> str:
    return card["card"]["oracleCard"]["uid"]


def deck_url(deck_id: int) -> str:
    return f"https://archidekt.com/decks/{deck_id}/"


MAX_ILLEGAL_NAMES_IN_LOG = 5


def deck_log_line(
    slot_label: str,
    url: str,
    *,
    stale: bool = False,
    mismatch: bool = False,
    wrong_size: bool = False,
    illegal_count: int = 0,
    illegal_names: list[str] | None = None,
) -> str:
    """One line of the `<commander>, problems: <...>, <url>` audit log (one per candidate
    deck, not just kept ones — the point is letting a human spot-check every decision, not
    just the ones already deemed noteworthy)."""
    problems = []
    if stale:
        problems.append("stale")
    if mismatch:
        problems.append("mismatch")
    if wrong_size:
        problems.append("wrong_size")
    if illegal_count:
        shown = ", ".join((illegal_names or [])[:MAX_ILLEGAL_NAMES_IN_LOG])
        problems.append(f"illegal_cards [{illegal_count}] ({shown})")
    return f"{slot_label}, problems: {', '.join(problems) if problems else 'none'}, {url}"


def _write_log(log_file: TextIO | None, line: str) -> None:
    if log_file is None:
        return
    log_file.write(line + "\n")
    log_file.flush()  # so `tail -f` works during a long-running harvest


def _resolve_local_oracle_id(conn: sqlite3.Connection, card_name: str) -> str:
    row = conn.execute(
        "SELECT oracle_id FROM card_names WHERE name_normalized = ?",
        (normalize_name(card_name),),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Commander {card_name!r} not found in card_names — has task 5.2 "
            "(edhcut.ingest.scryfall) run against this database?"
        )
    return row[0]


@dataclass
class FlaggedDeck:
    url: str
    reason: str


@dataclass
class SlotHarvestStats:
    slot_label: str
    candidates_checked: int = 0
    stale_skipped: int = 0
    commander_mismatch_rejected: int = 0
    invalid_size_rejected: int = 0
    decks_kept: int = 0
    cards_written: int = 0
    unresolved_oracle_ids: int = 0
    flagged: list[FlaggedDeck] = field(default_factory=list)
    error: str | None = None


def _upsert_deck(
    conn: sqlite3.Connection,
    deck: dict[str, Any],
    commander_oracle_id: str,
    partner_oracle_id: str | None,
) -> int:
    source = "archidekt"
    source_id = str(deck["id"])
    url = deck_url(deck["id"])
    fetched_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO decks (
            source, source_id, url, commander_oracle_id, partner_oracle_id,
            fetched_at, views, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, source_id) DO UPDATE SET
            url = excluded.url,
            commander_oracle_id = excluded.commander_oracle_id,
            partner_oracle_id = excluded.partner_oracle_id,
            fetched_at = excluded.fetched_at,
            views = excluded.views,
            updated_at = excluded.updated_at
        """,
        (
            source,
            source_id,
            url,
            commander_oracle_id,
            partner_oracle_id,
            fetched_at,
            deck.get("viewCount"),
            deck.get("updatedAt"),
        ),
    )
    row = conn.execute(
        "SELECT deck_id FROM decks WHERE source = ? AND source_id = ?", (source, source_id)
    ).fetchone()
    return row[0]


@dataclass
class DeckCardsResult:
    quantities: dict[str, int]  # oracle_id -> qty, for resolved cards actually in the library
    unresolved_count: int  # distinct oracle_ids that failed resolution
    unresolved_quantity: int  # total physical card count of unresolved cards
    # Names of unresolved cards, in encounter order (almost always banned/restricted-in-
    # Commander cards — see module docstring) — used to make the per-deck log actionable
    # without needing to click through to the deck.
    unresolved_names: list[str] = field(default_factory=list)

    @property
    def resolved_quantity(self) -> int:
        return sum(self.quantities.values())

    @property
    def total_physical_cards(self) -> int:
        return self.resolved_quantity + self.unresolved_quantity


def _compute_deck_cards(
    deck: dict[str, Any],
    *,
    exclude_oracle_ids: set[str],
    known_oracle_ids: set[str],
) -> DeckCardsResult:
    """Categorize this deck's library cards (excluding commander(s)/maybeboard) — no DB I/O.

    Kept separate from writing so the caller can decide whether the deck is even valid
    (exact 100-card size check) before touching the database at all.
    """
    quantities: dict[str, int] = {}
    unresolved_count = 0
    unresolved_quantity = 0
    unresolved_names: list[str] = []
    for card in included_cards(deck):
        oracle_id = card_oracle_id(card)
        if oracle_id in exclude_oracle_ids:
            continue
        quantity = card.get("quantity", 1)
        if oracle_id not in known_oracle_ids:
            unresolved_count += 1
            unresolved_quantity += quantity
            unresolved_names.append(card["card"]["oracleCard"]["name"])
            continue
        quantities[oracle_id] = quantities.get(oracle_id, 0) + quantity

    return DeckCardsResult(
        quantities=quantities,
        unresolved_count=unresolved_count,
        unresolved_quantity=unresolved_quantity,
        unresolved_names=unresolved_names,
    )


def _write_deck_cards(conn: sqlite3.Connection, deck_id: int, quantities: dict[str, int]) -> int:
    """Fully replace this deck's `deck_cards` rows. Returns rows written."""
    conn.execute("DELETE FROM deck_cards WHERE deck_id = ?", (deck_id,))
    rows = [(deck_id, oracle_id, qty) for oracle_id, qty in quantities.items()]
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, ?)", rows
    )
    conn.commit()
    return len(rows)


def harvest_slot(
    conn: sqlite3.Connection,
    session: RateLimitedSession,
    commander_names: list[str],
    known_oracle_ids: set[str],
    *,
    decks_per_commander: int,
    staleness_cutoff_days: int,
    show_progress: bool = True,
    log_file: TextIO | None = None,
) -> SlotHarvestStats:
    primary, *rest = commander_names
    other_partner = rest[0] if rest else None
    is_partner_slot = other_partner is not None
    stats = SlotHarvestStats(slot_label=" + ".join(commander_names))

    expected_oracle_ids = {name: _resolve_local_oracle_id(conn, name) for name in commander_names}
    required_ids = set(expected_oracle_ids.values())
    commander_oracle_id = expected_oracle_ids[primary]
    partner_oracle_id = expected_oracle_ids[other_partner] if other_partner else None
    expected_library_size = 100 - len(commander_names)

    stale_cutoff = datetime.now(timezone.utc) - timedelta(days=staleness_cutoff_days)

    pbar = tqdm(
        total=decks_per_commander,
        desc=stats.slot_label,
        unit="deck",
        disable=not show_progress,
    )
    try:
        for listing in search_deck_listings(session, primary, other_partner):
            if stats.decks_kept >= decks_per_commander:
                break
            stats.candidates_checked += 1
            pbar.set_postfix(
                checked=stats.candidates_checked,
                stale=stats.stale_skipped,
                mismatch=stats.commander_mismatch_rejected,
                bad_size=stats.invalid_size_rejected,
                unresolved=stats.unresolved_oracle_ids,
                refresh=False,
            )

            updated_at = listing.get("updatedAt")
            if updated_at and _parse_archidekt_datetime(updated_at) < stale_cutoff:
                stats.stale_skipped += 1
                _write_log(log_file, deck_log_line(stats.slot_label, deck_url(listing["id"]), stale=True))
                continue

            deck = fetch_deck(session, listing["id"])
            commander_ids = {card_oracle_id(c) for c in commander_cards(deck)}

            if not required_ids <= commander_ids:
                stats.commander_mismatch_rejected += 1
                _write_log(log_file, deck_log_line(stats.slot_label, deck_url(deck["id"]), mismatch=True))
                # For single-commander slots this is genuinely unexpected (the search already
                # filtered by commanderName) — worth a look. For partner slots, most
                # cardName matches are routine non-partnerships (see docs/archidekt_api.md),
                # so flagging every one would be noise, not signal.
                if not is_partner_slot:
                    stats.flagged.append(FlaggedDeck(
                        url=deck_url(deck["id"]),
                        reason="matched commander search but the searched commander isn't "
                               "\"Commander\"-tagged in the fetched deck",
                    ))
                continue

            # A deck must have *exactly* the right physical card count (99 library cards for
            # a single commander, 98 for a partner pair — 100 total including commander(s))
            # to be considered valid. Anything else is silently excluded, not flagged — a
            # routine filter, not an anomaly worth manual review. See `included_cards()` for
            # the maybeboard/sideboard/token handling this depends on, and quantity-summed
            # (not row-counted) per the same "30 Mountain is 1 row but 30 cards" fix as
            # before.
            result = _compute_deck_cards(
                deck, exclude_oracle_ids=required_ids, known_oracle_ids=known_oracle_ids
            )
            if result.total_physical_cards != expected_library_size:
                stats.invalid_size_rejected += 1
                _write_log(log_file, deck_log_line(
                    stats.slot_label, deck_url(deck["id"]),
                    wrong_size=True,
                    illegal_count=result.unresolved_count,
                    illegal_names=result.unresolved_names,
                ))
                continue

            stats.decks_kept += 1
            deck_pk = _upsert_deck(conn, deck, commander_oracle_id, partner_oracle_id)
            rows_written = _write_deck_cards(conn, deck_pk, result.quantities)
            stats.cards_written += rows_written
            stats.unresolved_oracle_ids += result.unresolved_count
            pbar.update(1)
            _write_log(log_file, deck_log_line(
                stats.slot_label, deck_url(deck["id"]),
                illegal_count=result.unresolved_count,
                illegal_names=result.unresolved_names,
            ))

            if (
                result.unresolved_quantity
                and result.unresolved_quantity / result.total_physical_cards
                > UNRESOLVED_RATIO_FLAG_THRESHOLD
            ):
                stats.flagged.append(FlaggedDeck(
                    url=deck_url(deck["id"]),
                    reason=f"{result.unresolved_quantity}/{result.total_physical_cards} cards "
                           f"unresolved ({result.unresolved_quantity / result.total_physical_cards:.0%}) "
                           "— may be more than just banned/restricted cards, worth a look",
                ))
    except Exception as exc:  # noqa: BLE001 - deliberately broad: this is a top-level harvest loop
        stats.error = (
            f"Harvest of slot {stats.slot_label!r} stopped after {stats.candidates_checked} "
            f"candidates checked ({stats.decks_kept} decks already saved to the database — "
            f"that progress is safe). Underlying error: {type(exc).__name__}: {exc}"
        )
    finally:
        pbar.close()

    return stats


def _write_ingest_log(conn: sqlite3.Connection, stats: SlotHarvestStats) -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, run_at, items, unresolved, notes) VALUES (?, ?, ?, ?, ?)",
        (
            "archidekt",
            datetime.now(timezone.utc).isoformat(),
            stats.decks_kept,
            stats.unresolved_oracle_ids,
            f"slot={stats.slot_label!r} candidates_checked={stats.candidates_checked} "
            f"stale_skipped={stats.stale_skipped} "
            f"commander_mismatch_rejected={stats.commander_mismatch_rejected} "
            f"invalid_size_rejected={stats.invalid_size_rejected} "
            f"decks_kept={stats.decks_kept} cards_written={stats.cards_written} "
            f"flagged={len(stats.flagged)}"
            + (f" error={stats.error!r}" if stats.error else ""),
        ),
    )
    conn.commit()


def run(
    conn: sqlite3.Connection,
    session: RateLimitedSession | None = None,
    *,
    slots: list[list[str]] | None = None,
    show_progress: bool = True,
    log_path: Path | None = None,
) -> list[SlotHarvestStats]:
    session = session or get_session("archidekt")
    slots = CONFIG.commander_slots if slots is None else slots
    known_oracle_ids = {row[0] for row in conn.execute("SELECT oracle_id FROM cards")}

    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")

    try:
        all_stats = []
        for commander_names in slots:
            stats = harvest_slot(
                conn, session, commander_names, known_oracle_ids,
                decks_per_commander=CONFIG.decks_per_commander,
                staleness_cutoff_days=CONFIG.deck_staleness_cutoff_days,
                show_progress=show_progress,
                log_file=log_file,
            )
            _write_ingest_log(conn, stats)
            all_stats.append(stats)
            if stats.error:
                break
        return all_stats
    finally:
        if log_file is not None:
            log_file.close()


def _print_summary(stats: SlotHarvestStats) -> None:
    print(
        f"{stats.slot_label}: kept {stats.decks_kept} decks "
        f"(checked {stats.candidates_checked} candidates, "
        f"{stats.stale_skipped} stale-skipped, "
        f"{stats.commander_mismatch_rejected} commander-mismatch, "
        f"{stats.invalid_size_rejected} wrong-size), "
        f"{stats.cards_written} card rows written, "
        f"{stats.unresolved_oracle_ids} unresolved oracle_ids"
    )
    if stats.flagged:
        print(f"  Flagged for manual review ({len(stats.flagged)}):")
        for flag in stats.flagged:
            print(f"    - {flag.url} — {flag.reason}")
    if stats.error:
        print(f"  STOPPED: {stats.error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest Commander decks from Archidekt for configured commander slots."
    )
    parser.add_argument(
        "--slot", type=int, default=None,
        help="Only harvest this 0-indexed slot from config.commander_slots (default: all slots).",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar (e.g. for non-interactive/log output).",
    )
    parser.add_argument(
        "--log-file", type=Path, default=CONFIG.paths.logs_dir / "archidekt_harvest_log.txt",
        help="Per-candidate audit log: '<commander>, problems: <...>, <url>' for every deck "
             "checked (not just kept ones) — appended to, so it accumulates across runs. "
             "Pass an empty string to disable.",
    )
    args = parser.parse_args()

    slots = CONFIG.commander_slots if args.slot is None else [CONFIG.commander_slots[args.slot]]
    log_path = args.log_file if str(args.log_file) else None

    with connect(CONFIG.paths.db_path) as conn:
        all_stats = run(conn, slots=slots, show_progress=not args.no_progress, log_path=log_path)

    print()
    for stats in all_stats:
        _print_summary(stats)
    if log_path is not None:
        print(f"\nPer-deck audit log: {log_path}")

    if any(stats.error for stats in all_stats):
        print(
            "\nOne or more slots stopped early. Already-harvested decks are saved (upserts "
            "are idempotent) — re-run with --slot N to retry just the affected slot(s)."
        )


if __name__ == "__main__":
    main()
