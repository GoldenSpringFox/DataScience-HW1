"""EDHREC metagame commander pool (plan `EDHCut_PLAN.md` §7 task 5.7).

Builds `meta_commanders`: every commander at/above `MIN_DECKS_THRESHOLD` decks across all 32
EDHREC colour-identity pages, plus a `sample_target` deck allocation for the meta-sample
harvest that consumes this table (`edhcut/ingest/archidekt.py`'s meta-sample mode, not yet
built). This module only builds the *list* of commanders and their targets -- it does not
harvest decks itself.

**Partner pairs**: EDHREC lists a partner-pair commander as one row with `name` = "A // B" --
confirmed live to be ~4.3% of the pool (43/999 at the 2026-08-09 snapshot), all genuine
distinct-card pairs (Thrasios/Tymna-style Partner combos, not a single DFC's own two-faced
name). `meta_commanders` mirrors `decks`'s own `slot_key`/`commander_oracle_id`/
`partner_oracle_id` shape (`edhcut.ingest.archidekt.slot_key_for`) rather than a single
`oracle_id` column, so these aren't silently dropped -- `_resolve_commander_listing` tries the
whole name first (a DFC's own name can itself contain `" // "`) and only falls back to
splitting if that fails.

**Route**: `edhrec.com/commanders/<slug>` (plain HTML page, `__NEXT_DATA__` embedded) --
**not** the `/_next/data/<build_id>/commanders/<slug>.json` route `edhrec.py` uses for
individual commander pages, which 404s for these colour-identity listing pages (checked live,
2026-08-09). Same underlying payload shape once unwrapped (`container.json_dict.cardlists`),
just reached differently, so `extract_commander_listings` below mirrors
`edhrec.extract_card_stats`'s cardlist-walking shape.

**Why the union of 32 colour-identity pages instead of paginating `edhrec.com/commanders`
directly**: that page's own listing is capped at 100 server-side (its "load more" button
paginates client-side, not reachable via a single fetch) -- confirmed live. The colour-identity
pages are 32 cheap cached fetches instead. **Verified live before relying on this**: the union
of all 32 pages' commanders at `num_decks >= 2,300` is exactly 999 distinct names with zero
duplicates -- an exact match for the user's own observation that EDHREC's global rank 999
sits at 2,301 decks, confirming this reconstructs the true global top-1000 rather than an
approximation of it. No identity is truncated: the most any single identity contributes above
the threshold is 50 (five-color), well under the observed 100-per-page cap -- `build_commander_pool`
still checks this defensively on every run (`_check_truncation`) since EDHREC's live numbers
will keep moving.

**Boros is `rw`, not `wr`.** The user's own colour-wheel list had one transposition, caught by
checking the URL live: `wr` 404s, `rw` returns Boros commanders. The enemy-pair ordering
elsewhere in the same list (`wb`, `bg`, `gu`, `ur`) is internally consistent with `rw` -- each
pair is "this colour, then the one clockwise past its enemy" -- so `rw` is the one that fits
the pattern, not `wr`. Guild/shard names (`boros`, `azorius`, `golgari`, ...) also resolve as
aliases but aren't used here, for a stable canonical slug per identity.

**Deck allocation**: `sample_target = clip(round(SAMPLE_TARGET_MAX * sqrt(share)), MIN, MAX)`,
`share = num_decks / max(num_decks across the pool)`. Square-root rather than proportional so
the most-played commanders don't consume the whole harvest budget and starve the tail --
proportional allocation would reintroduce exactly the corpus-dominance problem (task 6.3's
finding) this harvest exists to fix. Bounds and the resulting realised totals (≈9,355 decks,
mean 9.4/commander, ≈8.3h at the existing 2.0s Archidekt delay) are recorded in
`EDHCut_PLAN.md` task 5.7 -- this module doesn't itself harvest decks, so those numbers aren't
re-derivable from here alone.

Idempotent: `meta_commanders` is fully replaced on each run (same "delete then insert"
convention as `deck_cards`/`edhrec_card_stats`), since it's wholly re-fetchable and small (a
handful of cached page fetches).
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import sqrt

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.http import RateLimitedSession, get_session, request_with_retry
from edhcut.ingest.archidekt import _parse_next_data, slot_key_for
from edhcut.ingest.edhrec import BASE_URL, resolve_oracle_id

# The user's own colour-wheel ordering, with "wr" corrected to "rw" (see module docstring).
# Colourless uses the `colorless` slug -- bare `c` 404s.
COLOR_IDENTITIES: list[tuple[str, str]] = [
    ("colorless", "Colorless"), ("w", "White"), ("u", "Blue"), ("b", "Black"), ("r", "Red"), ("g", "Green"),
    ("wu", "Azorius"), ("ub", "Dimir"), ("br", "Rakdos"), ("rg", "Gruul"), ("gw", "Selesnya"),
    ("wb", "Orzhov"), ("bg", "Golgari"), ("gu", "Simic"), ("ur", "Izzet"), ("rw", "Boros"),
    ("wub", "Esper"), ("ubr", "Grixis"), ("brg", "Jund"), ("rgw", "Naya"), ("gwu", "Bant"),
    ("wbg", "Abzan"), ("urw", "Jeskai"), ("bgu", "Sultai"), ("rwb", "Mardu"), ("gur", "Temur"),
    ("wubr", "Yore"), ("ubrg", "Glint"), ("brgw", "Dune"), ("rgwu", "Ink"), ("gwub", "Witch"),
    ("wubrg", "Five Color"),
]

MIN_DECKS_THRESHOLD = 2300  # the 999th-ranked commander sits at 2,301 decks -- see module docstring
OBSERVED_PAGE_CAP = 100  # every non-4-colour identity page returns exactly this many rows
SAMPLE_TARGET_MIN = 5
SAMPLE_TARGET_MAX = 25

COMMANDER_LIST_URL_TEMPLATE = BASE_URL + "/commanders/{slug}"


def fetch_identity_page(session: RateLimitedSession, slug: str) -> dict:
    """Fetch one colour-identity commander-listing page and unwrap its cardlist payload."""
    response = request_with_retry(session, "GET", COMMANDER_LIST_URL_TEMPLATE.format(slug=slug))
    next_data = _parse_next_data(response.text)
    page = next_data.get("props", {}).get("pageProps", {}).get("data")
    if page is None or "container" not in page:
        raise RuntimeError(
            f"Unexpected EDHREC response shape for /commanders/{slug} -- no props.pageProps."
            "data.container (page shape may have drifted)"
        )
    return page


@dataclass
class CommanderListing:
    name: str
    num_decks: int
    rank: int | None
    color_identity: str  # the colour-identity slug this listing came from, e.g. "rg"


def extract_commander_listings(page_data: dict, *, color_identity: str) -> list[CommanderListing]:
    cardlists = page_data.get("container", {}).get("json_dict", {}).get("cardlists") or []
    listings = []
    for cardlist in cardlists:
        for cardview in cardlist.get("cardviews") or []:
            name = cardview.get("name")
            num_decks = cardview.get("num_decks")
            if not name or num_decks is None:
                continue
            listings.append(
                CommanderListing(name=name, num_decks=num_decks, rank=cardview.get("rank"), color_identity=color_identity)
            )
    return listings


PARTNER_NAME_SEPARATOR = " // "


def split_commander_name(name: str) -> list[str]:
    """EDHREC lists a partner-pair commander as one row, `name` = "A // B" -- split back into
    individual card names to resolve each side through `card_names` separately. A single
    commander's name never contains this exact separator (real card names don't), so this is a
    plain split, not a heuristic."""
    return [part.strip() for part in name.split(PARTNER_NAME_SEPARATOR)]


def sample_target(num_decks: int, max_num_decks: int) -> int:
    """Square-root-share deck allocation, clipped to `[SAMPLE_TARGET_MIN, SAMPLE_TARGET_MAX]` --
    see module docstring for why sqrt rather than proportional."""
    if max_num_decks <= 0:
        return SAMPLE_TARGET_MIN
    raw = round(SAMPLE_TARGET_MAX * sqrt(num_decks / max_num_decks))
    return max(SAMPLE_TARGET_MIN, min(SAMPLE_TARGET_MAX, raw))


@dataclass
class CommanderPoolResult:
    listings: list[CommanderListing]
    truncated_identities: list[str] = field(default_factory=list)
    failed_identities: list[tuple[str, str]] = field(default_factory=list)  # (slug, error message)


def _check_truncation(slug: str, raw_count: int, kept: list[CommanderListing]) -> bool:
    """A page is *possibly* truncated if it returned the observed page cap's worth of rows and
    every one of them still cleared the threshold -- meaning there might have been more
    qualifying commanders past what this page returned. Not the case for any identity as of
    2026-08-09 (max kept per identity: 50, well under the 100 cap) -- checked on every run
    since EDHREC's live numbers move."""
    return raw_count >= OBSERVED_PAGE_CAP and len(kept) == raw_count


def build_commander_pool(
    session: RateLimitedSession, *, threshold: int = MIN_DECKS_THRESHOLD
) -> CommanderPoolResult:
    """Fetch all 32 colour-identity pages, keep every commander at/above `threshold`. A
    commander appearing on more than one page would indicate a real problem (each commander has
    exactly one colour identity), so that raises rather than silently taking the first."""
    by_name: dict[str, CommanderListing] = {}
    truncated: list[str] = []
    failed: list[tuple[str, str]] = []

    for slug, label in COLOR_IDENTITIES:
        try:
            page = fetch_identity_page(session, slug)
        except Exception as exc:  # noqa: BLE001 - one identity's failure shouldn't abort the rest
            failed.append((slug, f"{type(exc).__name__}: {exc}"))
            continue

        raw_listings = extract_commander_listings(page, color_identity=slug)
        kept = [listing for listing in raw_listings if listing.num_decks >= threshold]
        if _check_truncation(slug, len(raw_listings), kept):
            truncated.append(slug)

        for listing in kept:
            existing = by_name.get(listing.name)
            if existing is not None:
                raise RuntimeError(
                    f"{listing.name!r} appears on more than one colour-identity page "
                    f"({existing.color_identity!r} and {slug!r}) -- a commander should have "
                    "exactly one colour identity, investigate before trusting this pool"
                )
            by_name[listing.name] = listing

    ranked = sorted(by_name.values(), key=lambda listing: -listing.num_decks)
    return CommanderPoolResult(listings=ranked, truncated_identities=truncated, failed_identities=failed)


@dataclass
class MetaCommandersStats:
    written: int = 0
    unresolved_names: int = 0
    unresolved_name_samples: list[str] = field(default_factory=list)
    truncated_identities: list[str] = field(default_factory=list)
    failed_identities: list[tuple[str, str]] = field(default_factory=list)


def _resolve_commander_listing(conn: sqlite3.Connection, name: str) -> tuple[str, str | None] | None:
    """Resolve one EDHREC commander-listing name to `(commander_oracle_id, partner_oracle_id)`.
    Tries the name whole first -- a single card's own name can itself contain the `" // "`
    partner separator (a DFC/split card, e.g. "Valki, God of Lies // Tibalt, Cosmic Impostor"),
    so splitting must never be the first move -- and only falls back to
    `split_commander_name` if that fails. Returns `None` if neither the whole name nor every
    split part resolves (a genuine partner pair only counts if *both* sides resolve; a
    half-resolved pair is not stored at all, since a wrong partner would corrupt the slot's
    identity)."""
    whole = resolve_oracle_id(conn, name)
    if whole is not None:
        return whole, None

    parts = split_commander_name(name)
    if len(parts) != 2:
        return None
    first, second = (resolve_oracle_id(conn, part) for part in parts)
    if first is None or second is None:
        return None
    return first, second


def _write_meta_commanders(conn: sqlite3.Connection, pool: CommanderPoolResult) -> MetaCommandersStats:
    """Fully replace `meta_commanders`. Unresolved names are skipped and logged, not inserted
    with a null oracle_id -- same convention as `edhrec.py`'s `_write_card_stats`."""
    conn.execute("DELETE FROM meta_commanders")
    max_decks = pool.listings[0].num_decks if pool.listings else 0
    fetched_at = datetime.now(timezone.utc).isoformat()

    rows = []
    unresolved_names: list[str] = []
    for rank, listing in enumerate(pool.listings, start=1):
        resolved = _resolve_commander_listing(conn, listing.name)
        if resolved is None:
            unresolved_names.append(listing.name)
            continue
        commander_oracle_id, partner_oracle_id = resolved
        oracle_ids = [commander_oracle_id] + ([partner_oracle_id] if partner_oracle_id else [])
        rows.append((
            slot_key_for(oracle_ids), commander_oracle_id, partner_oracle_id, listing.name,
            listing.color_identity, listing.num_decks, rank,
            sample_target(listing.num_decks, max_decks), fetched_at,
        ))

    conn.executemany(
        "INSERT INTO meta_commanders "
        "(slot_key, commander_oracle_id, partner_oracle_id, name, color_identity, edhrec_num_decks, "
        "edhrec_rank, sample_target, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    return MetaCommandersStats(
        written=len(rows),
        unresolved_names=len(unresolved_names),
        unresolved_name_samples=unresolved_names[:10],
        truncated_identities=pool.truncated_identities,
        failed_identities=pool.failed_identities,
    )


def _write_ingest_log(conn: sqlite3.Connection, stats: MetaCommandersStats) -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, run_at, items, unresolved, notes) VALUES (?, ?, ?, ?, ?)",
        (
            "edhrec_commanders",
            datetime.now(timezone.utc).isoformat(),
            stats.written,
            stats.unresolved_names,
            f"written={stats.written} unresolved_names={stats.unresolved_names}"
            + (f" unresolved_samples={stats.unresolved_name_samples!r}" if stats.unresolved_name_samples else "")
            + (f" truncated_identities={stats.truncated_identities!r}" if stats.truncated_identities else "")
            + (f" failed_identities={stats.failed_identities!r}" if stats.failed_identities else ""),
        ),
    )
    conn.commit()


def run(
    conn: sqlite3.Connection, session: RateLimitedSession | None = None, *, threshold: int = MIN_DECKS_THRESHOLD
) -> MetaCommandersStats:
    session = session or get_session("edhrec")
    pool = build_commander_pool(session, threshold=threshold)
    if pool.failed_identities:
        raise RuntimeError(
            f"Failed to fetch {len(pool.failed_identities)}/{len(COLOR_IDENTITIES)} colour-identity "
            f"page(s), refusing to write a partial pool: {pool.failed_identities}"
        )
    stats = _write_meta_commanders(conn, pool)
    _write_ingest_log(conn, stats)
    return stats


def _print_summary(stats: MetaCommandersStats) -> None:
    print(f"meta_commanders: {stats.written} rows written, {stats.unresolved_names} unresolved names")
    if stats.unresolved_name_samples:
        print(f"  Unresolved (sample): {', '.join(stats.unresolved_name_samples)}")
    if stats.truncated_identities:
        print(f"  WARNING -- possibly truncated identities (hit page cap while still above "
              f"threshold): {stats.truncated_identities}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the metagame commander pool (meta_commanders) from EDHREC's 32 "
        "colour-identity pages."
    )
    parser.add_argument(
        "--threshold", type=int, default=MIN_DECKS_THRESHOLD,
        help=f"Minimum edhrec num_decks to include a commander (default: {MIN_DECKS_THRESHOLD}).",
    )
    args = parser.parse_args()

    with connect(CONFIG.paths.db_path) as conn:
        stats = run(conn, threshold=args.threshold)

    _print_summary(stats)


if __name__ == "__main__":
    main()
