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
    colors: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield candidate deck *listing* dicts, paging through Archidekt's search page ourselves.

    Listings carry enough metadata (`id`, `updatedAt`, `viewCount`, ...) to pre-filter (e.g.
    for staleness) before spending a request on the full deck detail.

    `other_partner`, if given, adds `cardName=` to pre-filter to decks containing that card
    somewhere (not necessarily as a declared commander — caller must still verify via the
    deck's own `"Commander"` category once fetched; see docs/archidekt_api.md).

    `colors`, if given, adds `colors=` (a WUBRG-ordered string like "WR") to pre-filter to
    decks of *exactly* that color identity — verified live to be an exact match, not a
    superset/subset one (see docs/archidekt_api.md §2.2). Used by the partner-slot fallback.
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
        if colors:
            params["colors"] = colors

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


WUBRG = "WUBRG"


def combined_color_identity(conn: sqlite3.Connection, oracle_ids: list[str]) -> str:
    """Union of these cards' color identities, as a WUBRG-ordered string (e.g. "WR").

    This is the format Archidekt's `colors=` search param takes. Canonical WUBRG ordering is
    used so the same slot always produces the same query string.
    """
    letters: set[str] = set()
    for oracle_id in oracle_ids:
        row = conn.execute(
            "SELECT color_identity FROM cards WHERE oracle_id = ?", (oracle_id,)
        ).fetchone()
        if row and row[0]:
            letters.update(json.loads(row[0]))
    return "".join(letter for letter in WUBRG if letter in letters)


def slot_key_for(oracle_ids: list[str]) -> str:
    """Stable identifier for a configured commander slot — a bare `oracle_id` for a single
    commander, `"primary_id+partner_id"` for a partner pair (matching the `commander_key`
    convention `edhrec_card_stats` already uses). Stored on every harvested deck so
    partner-slot fallback decks (which run a *different* second commander) still group into
    the pair's corpus."""
    return "+".join(oracle_ids)


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
    # Partner slots only: how many of `decks_kept` ran the exact configured pair, before the
    # single-partner colour-matched fallback topped the slot up (None for single-commander
    # slots, and for partner slots the exact pair filled on its own).
    exact_pair_decks: int | None = None
    flagged: list[FlaggedDeck] = field(default_factory=list)
    error: str | None = None

    @property
    def fallback_decks(self) -> int:
        return 0 if self.exact_pair_decks is None else self.decks_kept - self.exact_pair_decks


def _upsert_deck(
    conn: sqlite3.Connection,
    deck: dict[str, Any],
    commander_oracle_id: str,
    partner_oracle_id: str | None,
    slot_key: str,
) -> int:
    source = "archidekt"
    source_id = str(deck["id"])
    url = deck_url(deck["id"])
    fetched_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO decks (
            source, source_id, url, commander_oracle_id, partner_oracle_id, slot_key,
            fetched_at, views, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, source_id) DO UPDATE SET
            url = excluded.url,
            commander_oracle_id = excluded.commander_oracle_id,
            partner_oracle_id = excluded.partner_oracle_id,
            slot_key = excluded.slot_key,
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
            slot_key,
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


def _harvest_pass(
    conn: sqlite3.Connection,
    session: RateLimitedSession,
    listings: Iterator[dict[str, Any]],
    *,
    required_ids: set[str],
    searched_oracle_id: str,
    slot_key: str,
    known_oracle_ids: set[str],
    stale_cutoff: datetime,
    stop_at_total: int,
    stats: SlotHarvestStats,
    seen_source_ids: set[str],
    flag_mismatches: bool,
    log_label: str,
    pbar: Any,
    log_file: TextIO | None,
) -> None:
    """Run one search's worth of candidates, keeping decks until `stats.decks_kept` reaches
    `stop_at_total`. Mutates `stats`/`seen_source_ids` in place; shared by the exact-pair pass
    and the partner-fallback passes (see `harvest_slot`)."""
    for listing in listings:
        if stats.decks_kept >= stop_at_total:
            return

        source_id = str(listing["id"])
        # A deck can legitimately surface in more than one pass — most obviously the exact
        # partner-pair decks, which also match both single-partner fallback searches. Skip
        # before spending a request on the deck detail, and don't let it count twice.
        if source_id in seen_source_ids:
            continue

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
            seen_source_ids.add(source_id)
            _write_log(log_file, deck_log_line(log_label, deck_url(listing["id"]), stale=True))
            continue

        deck = fetch_deck(session, listing["id"])
        seen_source_ids.add(source_id)
        deck_commander_ids = [card_oracle_id(c) for c in commander_cards(deck)]

        if not required_ids <= set(deck_commander_ids):
            stats.commander_mismatch_rejected += 1
            _write_log(log_file, deck_log_line(log_label, deck_url(deck["id"]), mismatch=True))
            # For single-commander slots this is genuinely unexpected (the search already
            # filtered by commanderName) — worth a look. For the exact-pair partner pass,
            # most cardName matches are routine non-partnerships (see docs/archidekt_api.md),
            # so flagging every one would be noise, not signal.
            if flag_mismatches:
                stats.flagged.append(FlaggedDeck(
                    url=deck_url(deck["id"]),
                    reason="matched commander search but the searched commander isn't "
                           "\"Commander\"-tagged in the fetched deck",
                ))
            continue

        # A deck must have *exactly* the right physical card count (100 total including its
        # commander(s), so 99 library cards for a single commander / 98 for a pair) to be
        # considered valid. Derived from the deck's *own* declared commander count rather
        # than the slot's, since a fallback deck runs a different second commander than the
        # slot names. Anything else is silently excluded, not flagged — a routine filter,
        # not an anomaly worth manual review. See `included_cards()` for the
        # maybeboard/sideboard/token handling this depends on, and quantity-summed (not
        # row-counted) per the same "30 Mountain is 1 row but 30 cards" fix as before.
        expected_library_size = 100 - len(deck_commander_ids)
        result = _compute_deck_cards(
            deck, exclude_oracle_ids=set(deck_commander_ids), known_oracle_ids=known_oracle_ids
        )
        if result.total_physical_cards != expected_library_size:
            stats.invalid_size_rejected += 1
            _write_log(log_file, deck_log_line(
                log_label, deck_url(deck["id"]),
                wrong_size=True,
                illegal_count=result.unresolved_count,
                illegal_names=result.unresolved_names,
            ))
            continue

        # The searched commander always lands in `commander_oracle_id`; whatever else the
        # deck declares as a commander goes in `partner_oracle_id`. For the exact-pair pass
        # that reproduces the slot's own pair; for a fallback pass it records the deck's real
        # other commander (e.g. Kediss, not Bruse) — `slot_key` is what ties it to the slot.
        others = [oid for oid in deck_commander_ids if oid != searched_oracle_id]
        stats.decks_kept += 1
        deck_pk = _upsert_deck(
            conn, deck, searched_oracle_id, others[0] if others else None, slot_key
        )
        rows_written = _write_deck_cards(conn, deck_pk, result.quantities)
        stats.cards_written += rows_written
        stats.unresolved_oracle_ids += result.unresolved_count
        pbar.update(1)
        _write_log(log_file, deck_log_line(
            log_label, deck_url(deck["id"]),
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
    """Harvest one configured commander slot up to `decks_per_commander` decks.

    Single-commander slots run a single search. **Partner slots** run up to three passes,
    because Archidekt rarely has enough decks running an exact partner pair (Yoshimaru +
    Bruse Tarl had just 4):

    1. *Exact pair* — `commanderName=<primary>&cardName=<partner>`, keeping only decks where
       both are `"Commander"`-tagged. These are true pair decks.
    2. *Fallback*, only if pass 1 came up short — for each partner separately,
       `commanderName=<that partner>&colors=<the pair's combined color identity>`, keeping
       decks where just that one partner is `"Commander"`-tagged. Because the color filter is
       an exact match, these are decks running that partner alongside some *other* partner
       that lands on the same colors (e.g. Yoshimaru + Kediss instead of + Bruse Tarl) —
       structurally the closest available stand-ins. The remaining quota is split evenly
       between the two partners, and if one runs dry the other picks up the slack.

    Fallback decks are stored with their real commanders; `decks.slot_key` is what groups
    them into this slot's corpus.
    """
    primary, *rest = commander_names
    other_partner = rest[0] if rest else None
    is_partner_slot = other_partner is not None
    stats = SlotHarvestStats(slot_label=" + ".join(commander_names))

    expected_oracle_ids = {name: _resolve_local_oracle_id(conn, name) for name in commander_names}
    slot_oracle_ids = [expected_oracle_ids[name] for name in commander_names]
    slot_key = slot_key_for(slot_oracle_ids)
    required_ids = set(slot_oracle_ids)

    stale_cutoff = datetime.now(timezone.utc) - timedelta(days=staleness_cutoff_days)
    seen_source_ids: set[str] = set()

    pbar = tqdm(
        total=decks_per_commander,
        desc=stats.slot_label,
        unit="deck",
        disable=not show_progress,
    )
    try:
        _harvest_pass(
            conn, session,
            search_deck_listings(session, primary, other_partner),
            required_ids=required_ids,
            searched_oracle_id=expected_oracle_ids[primary],
            slot_key=slot_key,
            known_oracle_ids=known_oracle_ids,
            stale_cutoff=stale_cutoff,
            stop_at_total=decks_per_commander,
            stats=stats,
            seen_source_ids=seen_source_ids,
            flag_mismatches=not is_partner_slot,
            log_label=stats.slot_label,
            pbar=pbar,
            log_file=log_file,
        )

        if is_partner_slot and stats.decks_kept < decks_per_commander:
            stats.exact_pair_decks = stats.decks_kept
            colors = combined_color_identity(conn, slot_oracle_ids)
            # Split what's left evenly, then run each partner again with whatever the pair
            # still owes — so if the first partner's search runs dry, the second one absorbs
            # its shortfall, and the third pass lets the first absorb the second's.
            half = (decks_per_commander - stats.decks_kept) // 2
            targets = [
                (primary, stats.decks_kept + half),
                (other_partner, decks_per_commander),
                (primary, decks_per_commander),
            ]
            for partner_name, stop_at_total in targets:
                if stats.decks_kept >= decks_per_commander:
                    break
                _harvest_pass(
                    conn, session,
                    search_deck_listings(session, partner_name, colors=colors),
                    required_ids={expected_oracle_ids[partner_name]},
                    searched_oracle_id=expected_oracle_ids[partner_name],
                    slot_key=slot_key,
                    known_oracle_ids=known_oracle_ids,
                    stale_cutoff=stale_cutoff,
                    stop_at_total=stop_at_total,
                    stats=stats,
                    seen_source_ids=seen_source_ids,
                    flag_mismatches=False,
                    log_label=f"{stats.slot_label} [fallback: {partner_name} @ {colors}]",
                    pbar=pbar,
                    log_file=log_file,
                )
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
            + (
                f" exact_pair_decks={stats.exact_pair_decks} "
                f"fallback_decks={stats.fallback_decks}"
                if stats.exact_pair_decks is not None else ""
            )
            + (f" error={stats.error!r}" if stats.error else ""),
        ),
    )
    conn.commit()


def run(
    conn: sqlite3.Connection,
    session: RateLimitedSession | None = None,
    *,
    slots: list[list[str]] | None = None,
    decks_per_commander: int | None = None,
    show_progress: bool = True,
    log_path: Path | None = None,
) -> list[SlotHarvestStats]:
    session = session or get_session("archidekt")
    slots = CONFIG.commander_slots if slots is None else slots
    decks_per_commander = CONFIG.decks_per_commander if decks_per_commander is None else decks_per_commander
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
                decks_per_commander=decks_per_commander,
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
    if stats.exact_pair_decks is not None:
        print(
            f"  Partner fallback used: {stats.exact_pair_decks} decks run the exact pair, "
            f"{stats.fallback_decks} run one partner at the pair's color identity"
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
        "--max-decks", type=int, default=None,
        help="Override config.decks_per_commander (default 300) for this run — e.g. a large "
             "value to harvest every available valid deck for the selected slot(s), since "
             "search_deck_listings() already stops naturally once Archidekt's own results run "
             "dry (no artificial cap needed to avoid over-fetching).",
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
        all_stats = run(
            conn, slots=slots, decks_per_commander=args.max_decks,
            show_progress=not args.no_progress, log_path=log_path,
        )

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
