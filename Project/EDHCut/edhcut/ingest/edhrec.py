"""EDHREC commander-page + global-tag harvester (plan §2.3/§6 tasks 5.4-B/5.4-C/5.4-D).

See `docs/edhrec_api.md` for the full API investigation this is built on: build-ID discovery
(homepage `__NEXT_DATA__` -> `.buildId`), the `/_next/data/<build_id>/commanders/<slug>.json`
route, and the undocumented partner-pair slug format (each partner's slug formatted
independently, then joined with a plain hyphen in alphabetical order — not `"+"`/`"&"`).
Deliberately does NOT depend on `pyedhrec` (reimplemented here against our own
cached/rate-limited session per the plan's step-B instruction) and deliberately does NOT copy
its rotating fake-browser User-Agent behavior — `edhcut.http.get_session` already attaches our
own identifying UA.

Populates two distinct kinds of tag data, in two tables — don't confuse them:
- `edhrec_themes` — the **global**, site-wide tag popularity ranking, covering both of
  EDHREC's own tag categories: `edhrec.com/tags/themes` (mechanical/archetype tags, e.g.
  "Aristocrats") and `edhrec.com/tags/typal` (tribal/creature-type tags, e.g. "Goblins") —
  distinguished by the `kind` column (`'theme'`/`'typal'`), same table, since both are the
  same kind of site-wide-comparable popularity number. This is the fixed vocabulary intended
  for the clustering/labeling step (task 6.6) — comparable across tags, not scoped to any one
  commander.
- `edhrec_themes_per_commander` — how many decks *for one specific commander slot* carry each
  tag, from that commander's own page. Not comparable across slots (a slot with a bigger
  overall corpus will show bigger raw counts for the same tag).
All three come from the same underlying EDHREC "cardlist" JSON shape (`container.json_dict.
cardlists`), just different pages (`/tags/themes`, `/tags/typal`, vs. `/commanders/<slug>`)
and a different field name for the payload (`count`/`value` on `panels.taglinks` for the
per-commander page, `num_decks`/`name` on the cardlist's own `cardviews` for the two global
pages) — see `extract_themes` vs. `extract_global_tags`.

Unlike Archidekt (`oracleCard.uid` *is* the Scryfall `oracle_id`), EDHREC cardview entries
carry no oracle_id at all — only EDHREC's own UUID plus `name`/`slug` — so resolution goes
through `card_names` like any free-text source, and unresolved names are expected and logged
rather than treated as errors.

A card commonly appears in more than one EDHREC cardlist on the same commander page (e.g. both
"Creatures" and "High Synergy Cards"), but `edhrec_card_stats` has one row per
`(commander_key, oracle_id)`. Explicit user decision on the resulting precedence: prefer
EDHREC's editorial "notability" tags over the type-based lists, in priority order Game
Changers > High Synergy Cards > Top Cards > New Cards, falling back to whichever type list
(Creatures/Instants/...) the card is in otherwise — this captures EDHREC's own judgment about
*why* a card is notable, which isn't derivable from the `type_line` we already have from
Scryfall. The underlying synergy/inclusion numbers are identical regardless of which list
"wins" — only the stored `category` label differs.

Run as `python -m edhcut.ingest.edhrec` (all configured commander slots, plus a refresh of the
global theme table) or `python -m edhcut.ingest.edhrec --slot N` (a single 0-indexed slot —
the global theme table still refreshes every run, since it's a single cheap cached fetch, not
scoped to any slot). Idempotent: `edhrec_card_stats`/`edhrec_themes_per_commander` are fully
replaced per commander slot on each (re-)run, and `edhrec_themes` is fully replaced wholesale
each run — same "delete then insert" convention as `deck_cards`/`precon_cards`.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from tqdm import tqdm

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.http import RateLimitedSession, get_session, request_with_retry
from edhcut.ingest.archidekt import _parse_next_data, _resolve_local_oracle_id, slot_key_for
from edhcut.ingest.scryfall import normalize_name

BASE_URL = "https://edhrec.com"
COMMANDER_DATA_URL_TEMPLATE = BASE_URL + "/_next/data/{build_id}/commanders/{slug}.json"
# EDHREC splits its own site-wide tag browser into two kinds: "themes" (mechanical/archetype
# tags like Aristocrats) and "typal" (tribal/creature-type tags like Goblins) — same page
# shape, different route, both feed the single `edhrec_themes` table (distinguished by `kind`).
GLOBAL_TAG_URL_TEMPLATES: dict[str, str] = {
    "theme": BASE_URL + "/_next/data/{build_id}/tags/themes.json",
    "typal": BASE_URL + "/_next/data/{build_id}/tags/typal.json",
}
GLOBAL_TAG_CARDLIST_TAG = "tagsbypopularitysort"

# Precedence order for which EDHREC cardlist "wins" a card's stored `category` when it
# appears in more than one — see module docstring for the reasoning. Any cardlist tag not in
# this list (the type-based ones: creatures, instants, sorceries, ...) is treated as a
# fallback, in whatever order the page itself lists them (stable per fetch, arbitrary between
# them — not itself a meaningful ranking).
_EDITORIAL_TAG_PRIORITY = ["gamechangers", "highsynergycards", "topcards", "newcards"]

MAX_UNRESOLVED_NAMES_IN_LOG = 10


def format_card_name(name: str) -> str:
    """EDHREC's own slug rule (`pyedhrec.format_card_name`, confirmed still current in
    5.4-A): lowercase, spaces -> hyphens, strip apostrophes and commas."""
    name = name.lower()
    name = name.replace(" ", "-")
    name = name.replace("'", "")
    name = name.replace(",", "")
    return name


def commander_slug(commander_names: list[str]) -> str:
    """URL slug for a commander page — a single formatted name, or for a partner pair, each
    partner's slug formatted independently and joined with a plain hyphen in alphabetical
    order (see docs/edhrec_api.md §2.2; not documented anywhere, worked out empirically)."""
    return "-".join(sorted(format_card_name(name) for name in commander_names))


def get_build_id(session: RateLimitedSession) -> str:
    """Fetch EDHREC's homepage and pull the current Next.js build id out of its embedded
    `__NEXT_DATA__` payload. Reuses Archidekt's generic `_parse_next_data` — same underlying
    technique, see docs/edhrec_api.md §2.1."""
    response = request_with_retry(session, "GET", BASE_URL)
    data = _parse_next_data(response.text)
    build_id = data.get("buildId")
    if not build_id:
        raise RuntimeError("Could not find buildId in EDHREC homepage __NEXT_DATA__ payload")
    return build_id


def fetch_commander_data(session: RateLimitedSession, build_id: str, slug: str) -> dict[str, Any]:
    """Fetch and unwrap one commander page's data payload. Raises if the response doesn't
    have the expected `pageProps.data` / `container` / `panels` shape — per the plan's
    instruction, page-shape drift should stop and be reported, not be silently worked around."""
    url = COMMANDER_DATA_URL_TEMPLATE.format(build_id=build_id, slug=slug)
    response = request_with_retry(session, "GET", url, params={"commanderName": slug})
    payload = response.json()
    data = payload.get("pageProps", {}).get("data")
    if data is None:
        raise RuntimeError(
            f"Unexpected EDHREC response shape for slug {slug!r} — no pageProps.data "
            "(page shape may have drifted, see docs/edhrec_api.md)"
        )
    if "container" not in data or "panels" not in data:
        raise RuntimeError(
            f"Unexpected EDHREC commander-page shape for slug {slug!r} — missing "
            "'container' or 'panels' (page shape may have drifted, see docs/edhrec_api.md)"
        )
    return data


@dataclass
class CardStat:
    name: str
    synergy: float | None
    num_decks: int | None
    potential_decks: int | None
    category: str


def extract_card_stats(commander_data: dict[str, Any]) -> dict[str, CardStat]:
    """One `CardStat` per distinct card name, keyed by name — the precedence rule from the
    module docstring resolves any card appearing in multiple cardlists to a single entry."""
    cardlists = commander_data.get("container", {}).get("json_dict", {}).get("cardlists") or []
    by_tag = {cardlist.get("tag"): cardlist for cardlist in cardlists}
    ordered_tags = [tag for tag in _EDITORIAL_TAG_PRIORITY if tag in by_tag]
    ordered_tags += [cardlist.get("tag") for cardlist in cardlists if cardlist.get("tag") not in _EDITORIAL_TAG_PRIORITY]

    chosen: dict[str, CardStat] = {}
    for tag in ordered_tags:
        cardlist = by_tag[tag]
        header = cardlist.get("header") or tag
        for cardview in cardlist.get("cardviews") or []:
            name = cardview.get("name")
            if not name or name in chosen:
                continue
            chosen[name] = CardStat(
                name=name,
                synergy=cardview.get("synergy"),
                num_decks=cardview.get("num_decks"),
                potential_decks=cardview.get("potential_decks"),
                category=header,
            )
    return chosen


@dataclass
class ThemeStat:
    theme: str
    num_decks: int


def extract_themes(commander_data: dict[str, Any]) -> list[ThemeStat]:
    taglinks = commander_data.get("panels", {}).get("taglinks") or []
    return [ThemeStat(theme=t["value"], num_decks=t["count"]) for t in taglinks if t.get("value")]


def fetch_global_tag_page(session: RateLimitedSession, build_id: str, kind: str) -> dict[str, Any]:
    """Fetch and unwrap `edhrec.com/tags/themes` (`kind="theme"`) or `edhrec.com/tags/typal`
    (`kind="typal"`) — identical Next.js data-route mechanism and payload shape, just a
    different page. Neither has a `panels` key (only `container`), so this can't reuse
    `fetch_commander_data`'s validation as-is."""
    url = GLOBAL_TAG_URL_TEMPLATES[kind].format(build_id=build_id)
    response = request_with_retry(session, "GET", url)
    payload = response.json()
    data = payload.get("pageProps", {}).get("data")
    if data is None or "container" not in data:
        raise RuntimeError(
            f"Unexpected EDHREC response shape for /tags/{kind} — no pageProps.data.container "
            "(page shape may have drifted, see docs/edhrec_api.md)"
        )
    return data


@dataclass
class GlobalThemeStat:
    theme: str
    kind: str  # "theme" | "typal"
    slug: str
    num_decks: int


def extract_global_tags(tag_page_data: dict[str, Any], *, kind: str) -> list[GlobalThemeStat]:
    """The single `"tagsbypopularitysort"` cardlist on `/tags/themes` or `/tags/typal` — every
    tag of that kind that exists anywhere on EDHREC, already sorted by global deck count
    descending. Each `cardviews` entry's `id`/`sanitized`/`slug` fields describe a
    representative *card* thumbnail EDHREC shows for that tag, not the tag itself — the tag's
    own identity is `name` (display name) and `url` (e.g. `"/tags/tokens"`, parsed here into a
    stable slug)."""
    cardlists = tag_page_data.get("container", {}).get("json_dict", {}).get("cardlists") or []
    cardlist = next((cl for cl in cardlists if cl.get("tag") == GLOBAL_TAG_CARDLIST_TAG), None)
    if cardlist is None:
        raise RuntimeError(
            f"Expected a {GLOBAL_TAG_CARDLIST_TAG!r} cardlist on /tags/{kind}, found none "
            "(page shape may have drifted, see docs/edhrec_api.md)"
        )
    tags = []
    for cardview in cardlist.get("cardviews") or []:
        name = cardview.get("name")
        num_decks = cardview.get("num_decks")
        url = cardview.get("url") or ""
        if not name or num_decks is None:
            continue
        slug = url.rsplit("/", 1)[-1] if url else format_card_name(name)
        tags.append(GlobalThemeStat(theme=name, kind=kind, slug=slug, num_decks=num_decks))
    return tags


def resolve_oracle_id(conn: sqlite3.Connection, card_name: str) -> str | None:
    row = conn.execute(
        "SELECT oracle_id FROM card_names WHERE name_normalized = ?",
        (normalize_name(card_name),),
    ).fetchone()
    return row[0] if row else None


@dataclass
class SlotEdhrecStats:
    slot_label: str
    cards_written: int = 0
    themes_written: int = 0
    unresolved_names: int = 0
    unresolved_name_samples: list[str] = field(default_factory=list)
    error: str | None = None


def _write_card_stats(
    conn: sqlite3.Connection,
    commander_key: str,
    card_stats: dict[str, CardStat],
    resolve: Callable[[str], str | None],
) -> tuple[int, int, list[str]]:
    """Fully replace this slot's `edhrec_card_stats` rows. Returns
    (rows_written, unresolved_count, sample_unresolved_names)."""
    conn.execute("DELETE FROM edhrec_card_stats WHERE commander_key = ?", (commander_key,))
    rows = []
    unresolved_names: list[str] = []
    for stat in card_stats.values():
        oracle_id = resolve(stat.name)
        if oracle_id is None:
            unresolved_names.append(stat.name)
            continue
        inclusion_rate = (
            stat.num_decks / stat.potential_decks
            if stat.num_decks is not None and stat.potential_decks
            else None
        )
        rows.append((commander_key, oracle_id, inclusion_rate, stat.synergy, stat.num_decks, stat.category))
    conn.executemany(
        "INSERT INTO edhrec_card_stats "
        "(commander_key, oracle_id, inclusion_rate, synergy_score, num_decks, category) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows), len(unresolved_names), unresolved_names


def _write_themes(conn: sqlite3.Connection, commander_key: str, themes: list[ThemeStat]) -> int:
    conn.execute("DELETE FROM edhrec_themes_per_commander WHERE commander_key = ?", (commander_key,))
    rows = [(commander_key, theme.theme, theme.num_decks) for theme in themes]
    conn.executemany(
        "INSERT INTO edhrec_themes_per_commander (commander_key, theme, num_decks) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def _write_global_themes(conn: sqlite3.Connection, tags: list[GlobalThemeStat]) -> int:
    """Fully replace the whole `edhrec_themes` table — a single global list covering both
    kinds, not scoped per commander slot, so unlike `_write_themes` there's no `WHERE` clause
    to scope the delete. Callers pass the combined theme + typal list in one call so the table
    is never left with only one kind mid-refresh."""
    conn.execute("DELETE FROM edhrec_themes")
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [(tag.theme, tag.kind, tag.slug, tag.num_decks, fetched_at) for tag in tags]
    conn.executemany(
        "INSERT INTO edhrec_themes (theme, kind, slug, num_decks, fetched_at) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


@dataclass
class GlobalThemesStats:
    themes_written: int = 0
    typal_written: int = 0
    error: str | None = None


def harvest_global_themes(
    conn: sqlite3.Connection, session: RateLimitedSession, build_id: str
) -> GlobalThemesStats:
    """Fetch both `/tags/themes` and `/tags/typal` and write them into `edhrec_themes`
    together (one wholesale replace, not two separate ones — see `_write_global_themes`)."""
    stats = GlobalThemesStats()
    try:
        all_tags: list[GlobalThemeStat] = []
        for kind in ("theme", "typal"):
            data = fetch_global_tag_page(session, build_id, kind)
            all_tags.extend(extract_global_tags(data, kind=kind))
        stats.themes_written = sum(1 for tag in all_tags if tag.kind == "theme")
        stats.typal_written = sum(1 for tag in all_tags if tag.kind == "typal")
        _write_global_themes(conn, all_tags)
    except Exception as exc:  # noqa: BLE001 - shouldn't abort the per-slot harvest that follows
        stats.error = f"{type(exc).__name__}: {exc}"
    return stats


def harvest_slot(
    conn: sqlite3.Connection,
    session: RateLimitedSession,
    build_id: str,
    commander_names: list[str],
) -> SlotEdhrecStats:
    slot_label = " + ".join(commander_names)
    stats = SlotEdhrecStats(slot_label=slot_label)
    try:
        oracle_ids = [_resolve_local_oracle_id(conn, name) for name in commander_names]
        commander_key = slot_key_for(oracle_ids)
        slug = commander_slug(commander_names)

        data = fetch_commander_data(session, build_id, slug)
        card_stats = extract_card_stats(data)
        themes = extract_themes(data)

        written, unresolved, unresolved_names = _write_card_stats(
            conn, commander_key, card_stats, lambda name: resolve_oracle_id(conn, name)
        )
        themes_written = _write_themes(conn, commander_key, themes)

        stats.cards_written = written
        stats.themes_written = themes_written
        stats.unresolved_names = unresolved
        stats.unresolved_name_samples = unresolved_names[:MAX_UNRESOLVED_NAMES_IN_LOG]
    except Exception as exc:  # noqa: BLE001 - one slot's failure shouldn't abort the whole run
        stats.error = f"{type(exc).__name__}: {exc}"
    return stats


def _write_ingest_log(conn: sqlite3.Connection, stats: SlotEdhrecStats) -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, run_at, items, unresolved, notes) VALUES (?, ?, ?, ?, ?)",
        (
            "edhrec",
            datetime.now(timezone.utc).isoformat(),
            stats.cards_written,
            stats.unresolved_names,
            f"slot={stats.slot_label!r} cards_written={stats.cards_written} "
            f"themes_written={stats.themes_written} unresolved_names={stats.unresolved_names}"
            + (f" unresolved_samples={stats.unresolved_name_samples!r}" if stats.unresolved_name_samples else "")
            + (f" error={stats.error!r}" if stats.error else ""),
        ),
    )
    conn.commit()


def _write_global_themes_ingest_log(conn: sqlite3.Connection, stats: GlobalThemesStats) -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, run_at, items, unresolved, notes) VALUES (?, ?, ?, ?, ?)",
        (
            "edhrec_global_themes",
            datetime.now(timezone.utc).isoformat(),
            stats.themes_written + stats.typal_written,
            0,
            f"themes_written={stats.themes_written} typal_written={stats.typal_written}"
            + (f" error={stats.error!r}" if stats.error else ""),
        ),
    )
    conn.commit()


@dataclass
class EdhrecRunResult:
    global_themes: GlobalThemesStats
    slots: list[SlotEdhrecStats]


def run(
    conn: sqlite3.Connection,
    session: RateLimitedSession | None = None,
    *,
    slots: list[list[str]] | None = None,
    show_progress: bool = True,
) -> EdhrecRunResult:
    session = session or get_session("edhrec")
    slots = CONFIG.commander_slots if slots is None else slots
    build_id = get_build_id(session)

    global_stats = harvest_global_themes(conn, session, build_id)
    _write_global_themes_ingest_log(conn, global_stats)

    all_stats = []
    for commander_names in tqdm(slots, desc="edhrec", unit="slot", disable=not show_progress):
        stats = harvest_slot(conn, session, build_id, commander_names)
        _write_ingest_log(conn, stats)
        all_stats.append(stats)
    return EdhrecRunResult(global_themes=global_stats, slots=all_stats)


def _print_global_themes_summary(stats: GlobalThemesStats) -> None:
    if stats.error:
        print(f"Global themes: STOPPED — {stats.error}")
    else:
        print(f"Global themes: {stats.themes_written} theme rows, {stats.typal_written} typal rows")


def _print_summary(stats: SlotEdhrecStats) -> None:
    if stats.error:
        print(f"{stats.slot_label}: STOPPED — {stats.error}")
        return
    print(
        f"{stats.slot_label}: {stats.cards_written} card stat rows, "
        f"{stats.themes_written} theme rows, {stats.unresolved_names} unresolved names"
    )
    if stats.unresolved_name_samples:
        print(f"  Unresolved (sample): {', '.join(stats.unresolved_name_samples)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest EDHREC commander-page card stats and themes for configured slots."
    )
    parser.add_argument(
        "--slot", type=int, default=None,
        help="Only harvest this 0-indexed slot from config.commander_slots (default: all slots).",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar (e.g. for non-interactive/log output).",
    )
    args = parser.parse_args()

    slots = CONFIG.commander_slots if args.slot is None else [CONFIG.commander_slots[args.slot]]

    with connect(CONFIG.paths.db_path) as conn:
        result = run(conn, slots=slots, show_progress=not args.no_progress)

    print()
    _print_global_themes_summary(result.global_themes)
    for stats in result.slots:
        _print_summary(stats)


if __name__ == "__main__":
    main()
