"""Scryfall Tagger bulk tag ingest (plan §3, task 5.5).

Official `oracle_tags` bulk file (`api.scryfall.com/bulk-data`, docs at
scryfall.com/docs/api/tags) — no scraping, fully permitted per plan §5. Live shape check
(2026-07-31) found two things worth correcting against the plan's original summary:
- Tag objects use `label` for the display name, not `name`, and there's no literal `category`
  field — every object in *this* bulk file already has `type == "oracle"` (the separate
  `art_tags` file is where illustration-tagged, non-functional entries live), so "keep only
  functional/oracle categories" is already satisfied by construction; no extra filtering code
  needed.
- Ancestor tags don't need a `child_ids` inversion — each tag object already carries its own
  direct `parent_ids`, so materializing ancestors is a straight walk upward from there.

Populates `card_tags` with `source='tagger_bulk'`: for every tagging, both the tag itself and
all of its transitive ancestor tags (by `slug`, the stable hyphenated identifier — `label` is
free-text and can contain spaces/casing that varies) are recorded against that `oracle_id`.
`card_tags.oracle_id` has an enforced foreign key to `cards` (`PRAGMA foreign_keys = ON` in
`db.connect`), so taggings referencing an oracle_id outside our own `cards` pool (e.g. a card
excluded by task 5.2's commander-legality filter) are skipped and counted, not inserted.

Also populates `tag_aliases`. Discovered manually verifying this task: Scryfall's own search
(`otag:boardwipe`) resolves against a tag's `aliases` array (e.g. the canonical tag `sweeper`
carries `aliases: ["wrath of god", "mass removal", "wipe", "boardwipe"]`), which the bulk file
provides but this ingest wasn't reading. Kept in a separate table rather than inserted as
duplicate `card_tags` rows, so an alias doesn't get double-counted as a distinct tag by
downstream co-occurrence/clustering — see `docs/devlog/5.5-tagger-bulk.md` for the design
discussion. No alias-alias or alias-slug collisions found live (checked directly against the
bulk file), so `tag_aliases` uses a bare `alias_normalized` primary key.

Run as `python -m edhcut.ingest.tagger_bulk`. Idempotent: replaces all `source='tagger_bulk'`
rows in `card_tags`, and all of `tag_aliases`, wholesale each run (the whole bulk file is one
source, not per-item scoped, so there's no natural per-row upsert key finer than that).
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.http import RateLimitedSession, get_session
from edhcut.ingest.scryfall import download_bulk_file, get_bulk_data_info, iter_bulk_lines, normalize_name

BULK_DATA_TYPE = "oracle_tags"
SOURCE = "tagger_bulk"


def load_tag_registry(path: Path) -> dict[str, dict[str, Any]]:
    """`{tag_id: tag_object}` for every tag in the bulk file, keyed by its own `id`."""
    return {tag["id"]: tag for tag in iter_bulk_lines(path) if tag.get("id")}


def ancestor_slugs(tag_id: str, registry: dict[str, dict[str, Any]]) -> set[str]:
    """Every ancestor tag's `slug`, walking `parent_ids` transitively. A tag can have more
    than one direct parent (it's a DAG, not strictly a tree) — visited-set guards against a
    cycle or a diamond re-visiting the same ancestor twice."""
    seen: set[str] = set()
    slugs: set[str] = set()
    stack = list(registry.get(tag_id, {}).get("parent_ids") or [])
    while stack:
        parent_id = stack.pop()
        if parent_id in seen:
            continue
        seen.add(parent_id)
        parent = registry.get(parent_id)
        if parent is None:  # dangling reference — tag removed from Tagger since this parent_id was recorded
            continue
        if parent.get("slug"):
            slugs.add(parent["slug"])
        stack.extend(parent.get("parent_ids") or [])
    return slugs


@dataclass
class ExpandResult:
    rows: set[tuple[str, str]]  # (oracle_id, tag_slug)
    tag_counts: Counter  # tag_slug -> number of distinct cards carrying it (incl. via ancestors)
    total_taggings: int
    unresolved_oracle_ids: int
    tagged_oracle_ids: set[str]


def expand_taggings(registry: dict[str, dict[str, Any]], known_oracle_ids: set[str]) -> ExpandResult:
    """For every tagging in the registry, expand to (oracle_id, tag_slug) rows for the tag
    itself and all of its ancestors, keeping only oracle_ids present in `known_oracle_ids`
    (mirrors `card_tags`'s enforced FK to `cards`). Pure/no DB or network I/O — the part of
    this module worth testing directly."""
    rows: set[tuple[str, str]] = set()
    tag_counts: Counter = Counter()
    total_taggings = 0
    unresolved = 0
    tagged_oracle_ids: set[str] = set()

    for tag in registry.values():
        slug = tag.get("slug")
        if not slug:
            continue
        ancestors = ancestor_slugs(tag["id"], registry)
        for tagging in tag.get("taggings") or []:
            total_taggings += 1
            oracle_id = tagging.get("oracle_id")
            if oracle_id not in known_oracle_ids:
                unresolved += 1
                continue
            tagged_oracle_ids.add(oracle_id)
            for tag_slug in {slug, *ancestors}:
                if (oracle_id, tag_slug) not in rows:
                    tag_counts[tag_slug] += 1
                rows.add((oracle_id, tag_slug))

    return ExpandResult(
        rows=rows,
        tag_counts=tag_counts,
        total_taggings=total_taggings,
        unresolved_oracle_ids=unresolved,
        tagged_oracle_ids=tagged_oracle_ids,
    )


def extract_aliases(registry: dict[str, dict[str, Any]]) -> dict[str, str]:
    """`{alias_normalized: canonical_tag_slug}` for every alias in the bulk file. Pure, no I/O.
    Skips a tag with no `slug` (shouldn't happen — same guard `expand_taggings` uses) and
    silently lets a later tag win if two ever claim the same normalized alias (none do today,
    verified directly against the live bulk file — see module docstring)."""
    aliases: dict[str, str] = {}
    for tag in registry.values():
        slug = tag.get("slug")
        if not slug:
            continue
        for alias in tag.get("aliases") or []:
            key = normalize_name(alias)
            if key:
                aliases[key] = slug
    return aliases


@dataclass
class TaggerBulkStats:
    total_tags: int = 0
    total_taggings: int = 0
    rows_written: int = 0
    unresolved_oracle_ids: int = 0
    tagged_card_count: int = 0
    tag_counts: Counter = None  # slug -> number of distinct cards carrying it (incl. ancestors)
    alias_count: int = 0
    download_bytes: int = 0


def _write_card_tags(conn: sqlite3.Connection, rows: set[tuple[str, str]]) -> int:
    conn.execute("DELETE FROM card_tags WHERE source = ?", (SOURCE,))
    conn.executemany(
        "INSERT INTO card_tags (oracle_id, tag, source) VALUES (?, ?, ?)",
        [(oracle_id, tag, SOURCE) for oracle_id, tag in rows],
    )
    conn.commit()
    return len(rows)


def _write_tag_aliases(conn: sqlite3.Connection, aliases: dict[str, str]) -> int:
    conn.execute("DELETE FROM tag_aliases")
    conn.executemany(
        "INSERT INTO tag_aliases (alias_normalized, tag) VALUES (?, ?)", list(aliases.items())
    )
    conn.commit()
    return len(aliases)


def _write_ingest_log(conn: sqlite3.Connection, stats: TaggerBulkStats) -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, run_at, items, unresolved, notes) VALUES (?, ?, ?, ?, ?)",
        (
            SOURCE,
            datetime.now(timezone.utc).isoformat(),
            stats.rows_written,
            stats.unresolved_oracle_ids,
            f"total_tags={stats.total_tags} total_taggings={stats.total_taggings} "
            f"rows_written={stats.rows_written} tagged_card_count={stats.tagged_card_count} "
            f"unresolved_oracle_ids={stats.unresolved_oracle_ids} alias_count={stats.alias_count}",
        ),
    )
    conn.commit()


def run(conn: sqlite3.Connection, session: RateLimitedSession | None = None) -> TaggerBulkStats:
    session = session or get_session("scryfall")

    bulk_info = get_bulk_data_info(session, BULK_DATA_TYPE)
    dest = CONFIG.paths.raw_dir / "oracle_tags.jsonl.gz"
    download_bytes = download_bulk_file(
        session, bulk_info["jsonl_download_uri"], dest, desc="Downloading oracle_tags"
    )

    registry = load_tag_registry(dest)
    known_oracle_ids = {row[0] for row in conn.execute("SELECT oracle_id FROM cards")}

    expanded = expand_taggings(registry, known_oracle_ids)
    rows_written = _write_card_tags(conn, expanded.rows)

    aliases = extract_aliases(registry)
    alias_count = _write_tag_aliases(conn, aliases)

    stats = TaggerBulkStats(
        total_tags=len(registry),
        total_taggings=expanded.total_taggings,
        rows_written=rows_written,
        unresolved_oracle_ids=expanded.unresolved_oracle_ids,
        tagged_card_count=len(expanded.tagged_oracle_ids),
        tag_counts=expanded.tag_counts,
        alias_count=alias_count,
        download_bytes=download_bytes,
    )
    _write_ingest_log(conn, stats)
    return stats


def main() -> None:
    with connect(CONFIG.paths.db_path) as conn:
        stats = run(conn)
        total_cards = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]

    coverage = stats.tagged_card_count / total_cards if total_cards else 0.0
    print(
        f"Tagger bulk ingest complete: {stats.rows_written:,} card_tags rows "
        f"({stats.total_tags:,} tags, {stats.total_taggings:,} taggings), "
        f"{stats.tagged_card_count:,}/{total_cards:,} cards tagged ({coverage:.1%}), "
        f"{stats.unresolved_oracle_ids} unresolved oracle_ids, "
        f"{stats.alias_count:,} tag aliases, "
        f"{stats.download_bytes / 1_048_576:.1f} MB downloaded."
    )
    print("Top 20 tags by card count:")
    for slug, count in stats.tag_counts.most_common(20):
        print(f"  {slug}: {count}")


if __name__ == "__main__":
    main()
