"""Archidekt official precon harvester (plan §5.3 QA — precon-contamination detection).

Populates `precons` (one row per official precon decklist) and `precon_cards` (its full
100-card list, commander(s) included). Source: `archidekt.com/commander-precons`, a curated
index Archidekt itself maintains (owner account "Archidekt_Precons") — same
`__NEXT_DATA__`-embedded-JSON technique as deck search
(`docs/archidekt_api.md` §2.2), but the payload here is a ready-made
`{set_name: [listing, ...]}` dict at `props.pageProps.precons`, not a paginated search result.
Each listing's `id` is a normal Archidekt deck id, fetched the same way as any other deck
(`fetch_deck()`).

Unlike `deck_cards`, `precon_cards` keeps the commander(s) in the card list. A precon's
physical contents don't change based on which of its legendaries someone chooses to run as
commander, and precons often ship other legendary creatures in the same color identity as
genuine alternative-commander options (confirmed live: Kyler, Sigardian Emissary — GW —
ships inside the Midnight Hunt "Coven Counters" precon, whose *declared* commander is
Leinore, Autumn Sovereign, also GW, deck 2209041) — `alternative_commander_oracle_ids`
records those.

Run as `python -m edhcut.ingest.precons`. Idempotent: `precons` upserts by `precon_id`
(Archidekt's own deck id); `precon_cards` is fully replaced per precon on each re-run.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.http import RateLimitedSession, get_session, request_with_retry
from edhcut.ingest.archidekt import (
    WUBRG,
    _parse_next_data,
    card_oracle_id,
    combined_color_identity,
    commander_cards,
    deck_url,
    fetch_deck,
    included_cards,
)

PRECON_INDEX_URL = "https://archidekt.com/commander-precons"

_SET_CODE_RE = re.compile(r"\(([A-Za-z0-9]+)\)\s*$")


def parse_set_code(set_name: str) -> str | None:
    """"Midnight Hunt Commander (MIC)" -> "MIC". None if the label carries no trailing code
    (seen on a couple of older/miscellaneous folders)."""
    match = _SET_CODE_RE.search(set_name)
    return match.group(1) if match else None


def fetch_precon_index(session: RateLimitedSession) -> dict[str, list[dict[str, Any]]]:
    """`{set_name: [listing, ...]}` — every official precon Archidekt's curated page lists,
    across every set. Listings carry `id`/`name` (enough to fetch the full deck) but not the
    commander or card list."""
    response = request_with_retry(session, "GET", PRECON_INDEX_URL)
    return _parse_next_data(response.text)["props"]["pageProps"]["precons"]


def alternative_commander_oracle_ids(
    conn: sqlite3.Connection,
    declared_oracle_ids: list[str],
    precon_oracle_ids: set[str],
) -> list[str]:
    """Other legendary creatures (or "can be your commander" cards — `can_be_commander`
    covers both) in this precon's own card list that share the declared commander(s)' exact
    combined color identity — real alternative-commander candidates a player might swap in
    instead of the box's printed commander(s), not just any legend that happens to be along
    for the ride."""
    target = combined_color_identity(conn, declared_oracle_ids)
    candidates = precon_oracle_ids - set(declared_oracle_ids)
    if not candidates:
        return []
    placeholders = ",".join("?" * len(candidates))
    rows = conn.execute(
        f"SELECT oracle_id, color_identity FROM cards "
        f"WHERE oracle_id IN ({placeholders}) AND can_be_commander = 1",
        tuple(candidates),
    ).fetchall()
    matches = []
    for oracle_id, color_identity_json in rows:
        letters = set(json.loads(color_identity_json)) if color_identity_json else set()
        identity = "".join(letter for letter in WUBRG if letter in letters)
        if identity == target:
            matches.append(oracle_id)
    return sorted(matches)


@dataclass
class PreconHarvestStats:
    precons_checked: int = 0
    precons_kept: int = 0
    cards_written: int = 0
    unresolved_oracle_ids: int = 0
    errors: list[str] = field(default_factory=list)


def _upsert_precon(
    conn: sqlite3.Connection,
    listing: dict[str, Any],
    *,
    set_name: str,
    set_code: str | None,
    commander_oracle_id: str | None,
    partner_oracle_id: str | None,
    alternative_oracle_ids: list[str],
) -> int:
    """Insert or refresh one precon product row, keyed on Archidekt's own `precon_id`. Upsert
    rather than insert so re-running the ingest updates in place — a precon's commander resolution
    can improve once Scryfall data is fresher, and the harvest is meant to be resumable."""
    precon_id = listing["id"]
    conn.execute(
        """
        INSERT INTO precons (
            precon_id, set_name, set_code, deck_name, url,
            commander_oracle_id, partner_oracle_id, alternative_commander_oracle_ids,
            fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(precon_id) DO UPDATE SET
            set_name = excluded.set_name,
            set_code = excluded.set_code,
            deck_name = excluded.deck_name,
            url = excluded.url,
            commander_oracle_id = excluded.commander_oracle_id,
            partner_oracle_id = excluded.partner_oracle_id,
            alternative_commander_oracle_ids = excluded.alternative_commander_oracle_ids,
            fetched_at = excluded.fetched_at
        """,
        (
            precon_id,
            set_name,
            set_code,
            listing["name"],
            deck_url(precon_id),
            commander_oracle_id,
            partner_oracle_id,
            json.dumps(alternative_oracle_ids),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    return precon_id


def _write_precon_cards(
    conn: sqlite3.Connection, precon_id: int, quantities: dict[str, int]
) -> int:
    conn.execute("DELETE FROM precon_cards WHERE precon_id = ?", (precon_id,))
    rows = [(precon_id, oracle_id, qty) for oracle_id, qty in quantities.items()]
    conn.executemany(
        "INSERT INTO precon_cards (precon_id, oracle_id, qty) VALUES (?, ?, ?)", rows
    )
    conn.commit()
    return len(rows)


def harvest_precons(
    conn: sqlite3.Connection,
    session: RateLimitedSession,
    known_oracle_ids: set[str],
    *,
    show_progress: bool = True,
) -> PreconHarvestStats:
    """Fetch every precon Archidekt's curated index lists and upsert it. Unlike
    `harvest_slot()`, there's no candidate filtering here — the index is a small, fixed,
    already-curated list (179 decks as of this writing), not search results to be validated.
    A per-deck fetch failure is recorded in `stats.errors` and skipped rather than aborting
    the whole run (a single broken listing shouldn't cost the other ~178)."""
    stats = PreconHarvestStats()
    index = fetch_precon_index(session)
    total = sum(len(listings) for listings in index.values())

    pbar = tqdm(total=total, desc="precons", unit="deck", disable=not show_progress)
    try:
        for set_name, listings in index.items():
            set_code = parse_set_code(set_name)
            for listing in listings:
                stats.precons_checked += 1
                pbar.set_postfix(
                    checked=stats.precons_checked,
                    kept=stats.precons_kept,
                    errors=len(stats.errors),
                    refresh=False,
                )
                try:
                    deck = fetch_deck(session, listing["id"])
                except Exception as exc:  # noqa: BLE001 - one bad listing shouldn't abort the run
                    stats.errors.append(
                        f"{listing['id']} ({listing['name']!r}): {type(exc).__name__}: {exc}"
                    )
                    pbar.update(1)
                    continue

                declared_ids = [card_oracle_id(c) for c in commander_cards(deck)]

                quantities: dict[str, int] = {}
                unresolved = 0
                for card in included_cards(deck):
                    oracle_id = card_oracle_id(card)
                    quantity = card.get("quantity", 1)
                    if oracle_id not in known_oracle_ids:
                        unresolved += 1
                        continue
                    quantities[oracle_id] = quantities.get(oracle_id, 0) + quantity

                alt_ids = alternative_commander_oracle_ids(
                    conn, declared_ids, set(quantities)
                )
                precon_id = _upsert_precon(
                    conn, listing,
                    set_name=set_name, set_code=set_code,
                    commander_oracle_id=declared_ids[0] if declared_ids else None,
                    partner_oracle_id=declared_ids[1] if len(declared_ids) > 1 else None,
                    alternative_oracle_ids=alt_ids,
                )
                rows_written = _write_precon_cards(conn, precon_id, quantities)

                stats.precons_kept += 1
                stats.cards_written += rows_written
                stats.unresolved_oracle_ids += unresolved
                pbar.update(1)
    finally:
        pbar.close()

    return stats


def _write_ingest_log(conn: sqlite3.Connection, stats: PreconHarvestStats) -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, run_at, items, unresolved, notes) VALUES (?, ?, ?, ?, ?)",
        (
            "archidekt_precons",
            datetime.now(timezone.utc).isoformat(),
            stats.precons_kept,
            stats.unresolved_oracle_ids,
            f"precons_checked={stats.precons_checked} precons_kept={stats.precons_kept} "
            f"cards_written={stats.cards_written} errors={len(stats.errors)}"
            + (f" error_details={stats.errors!r}" if stats.errors else ""),
        ),
    )
    conn.commit()


def run(
    conn: sqlite3.Connection,
    session: RateLimitedSession | None = None,
    *,
    show_progress: bool = True,
) -> PreconHarvestStats:
    session = session or get_session("archidekt")
    known_oracle_ids = {row[0] for row in conn.execute("SELECT oracle_id FROM cards")}
    stats = harvest_precons(conn, session, known_oracle_ids, show_progress=show_progress)
    _write_ingest_log(conn, stats)
    return stats


def main() -> None:
    argparse.ArgumentParser(
        description="Harvest official precon decklists from Archidekt's curated index."
    ).parse_args()

    with connect(CONFIG.paths.db_path) as conn:
        stats = run(conn)

    print(
        f"precons: kept {stats.precons_kept}/{stats.precons_checked}, "
        f"{stats.cards_written} card rows written, "
        f"{stats.unresolved_oracle_ids} unresolved oracle_ids"
    )
    if stats.errors:
        print(f"  {len(stats.errors)} listing(s) failed to fetch:")
        for err in stats.errors:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
