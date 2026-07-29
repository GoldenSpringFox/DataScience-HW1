"""Scryfall `oracle_cards` bulk ingest + name resolution table (plan §2.2–2.3, task 5.2).

Downloads the official `oracle_cards` bulk file, filters to `legalities.commander == "legal"`,
and populates `cards` + `card_names`. `oracle_id` is the canonical key (plan §2.2) — every
alias in `card_names` resolves back to it.

NOTE: an earlier version also required `"paper" in games` on top of the legality check — this
was reverted (see devlog) after it was caught wrongly excluding 1,048 real, commander-legal
cards. The `oracle_cards` bulk file has exactly one representative printing per `oracle_id`,
and for cards later reprinted MTGO-only (the "Masters Edition" series), that representative
printing can be the digital one even though the card has real paper prints (e.g. Tundra,
Palinchron). `legalities.commander` is already computed against the paper-legal card pool
regardless of which printing got picked, so it's sufficient alone — confirmed genuinely
digital-only cards (Alchemy rebalances) correctly report `commander: "not_legal"`.

Run as `python -m edhcut.ingest.scryfall`. Idempotent: re-running upserts by `oracle_id` /
`name_normalized` rather than duplicating rows (see devlog for the tradeoff this implies).
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.http import RateLimitedSession, get_session

BULK_DATA_INDEX_URL = "https://api.scryfall.com/bulk-data"
BULK_DATA_TYPE = "oracle_cards"

# Scryfall's `æ`/`œ` cards don't decompose under NFKD (they're atomic letters, not
# accented ASCII) — spell them out explicitly so e.g. "Ætherize" resolves under "aetherize".
_CHAR_REPLACEMENTS = {"æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe"}


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace — the `card_names` key."""
    text = name.strip().lower()
    for src, dst in _CHAR_REPLACEMENTS.items():
        text = text.replace(src.lower(), dst)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_bulk_data_info(session: RateLimitedSession) -> dict[str, Any]:
    response = session.get(BULK_DATA_INDEX_URL)
    response.raise_for_status()
    for entry in response.json()["data"]:
        if entry["type"] == BULK_DATA_TYPE:
            return entry
    raise RuntimeError(f"No bulk-data entry of type {BULK_DATA_TYPE!r} found")


def download_bulk_file(session: RateLimitedSession, download_uri: str, dest: Path) -> int:
    """Stream the gzip-compressed JSONL bulk file to `dest`. Returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(download_uri, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading oracle_cards") as bar:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                bar.update(len(chunk))
    return dest.stat().st_size


def iter_bulk_cards(path: Path):
    """Yield one card dict per line from a gzip-compressed JSONL bulk file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _face_field(card: dict, field: str) -> str | None:
    faces = card.get("card_faces")
    if not faces:
        return None
    parts = [f[field] for f in faces if f.get(field)]
    return " // ".join(parts) if parts else None


def _face_names(card: dict) -> list[str]:
    faces = card.get("card_faces")
    if faces and len(faces) > 1:
        return [f["name"] for f in faces if f.get("name")]
    name = card.get("name", "")
    if " // " in name:
        return [part.strip() for part in name.split(" // ")]
    return []


_WUBRG_ORDER = {c: i for i, c in enumerate("WUBRG")}


def _colors(card: dict) -> list[str]:
    colors = card.get("colors")
    if colors is None:
        faces = card.get("card_faces") or []
        merged: set[str] = set()
        for face in faces:
            merged.update(face.get("colors") or [])
        colors = list(merged)
    return sorted(colors, key=lambda c: _WUBRG_ORDER.get(c, 99))


def is_commander_legal(card: dict) -> bool:
    return (card.get("legalities") or {}).get("commander") == "legal"


def can_be_commander(card: dict) -> bool:
    type_line = card.get("type_line") or ""
    if "Legendary" in type_line and "Creature" in type_line:
        return True
    oracle_text = card.get("oracle_text") or _face_field(card, "oracle_text") or ""
    return "can be your commander" in oracle_text.lower()


def build_card_row(card: dict) -> dict[str, Any]:
    prices = card.get("prices") or {}
    usd = prices.get("usd")
    type_line = card.get("type_line") or ""

    return {
        "oracle_id": card["oracle_id"],
        "name": card["name"],
        "mana_cost": card.get("mana_cost") or _face_field(card, "mana_cost"),
        "cmc": card.get("cmc"),
        "type_line": type_line,
        "oracle_text": card.get("oracle_text") or _face_field(card, "oracle_text"),
        "colors": json.dumps(_colors(card)),
        "color_identity": json.dumps(card.get("color_identity") or []),
        "keywords": json.dumps(card.get("keywords") or []),
        "rarity": card.get("rarity"),
        "edhrec_rank": card.get("edhrec_rank"),
        "price_usd": float(usd) if usd else None,
        "game_changer": bool(card.get("game_changer", False)),
        "legal_commander": is_commander_legal(card),
        "can_be_commander": can_be_commander(card),
        "layout": card.get("layout"),
        "produced_mana": json.dumps(card.get("produced_mana") or []),
        "is_land": "Land" in type_line,
    }


def _write_cards(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    conn.executemany(
        """
        INSERT INTO cards (
            oracle_id, name, mana_cost, cmc, type_line, oracle_text, colors,
            color_identity, keywords, rarity, edhrec_rank, price_usd, game_changer,
            legal_commander, can_be_commander, layout, produced_mana, is_land
        ) VALUES (
            :oracle_id, :name, :mana_cost, :cmc, :type_line, :oracle_text, :colors,
            :color_identity, :keywords, :rarity, :edhrec_rank, :price_usd, :game_changer,
            :legal_commander, :can_be_commander, :layout, :produced_mana, :is_land
        )
        ON CONFLICT(oracle_id) DO UPDATE SET
            name=excluded.name, mana_cost=excluded.mana_cost, cmc=excluded.cmc,
            type_line=excluded.type_line, oracle_text=excluded.oracle_text,
            colors=excluded.colors, color_identity=excluded.color_identity,
            keywords=excluded.keywords, rarity=excluded.rarity,
            edhrec_rank=excluded.edhrec_rank, price_usd=excluded.price_usd,
            game_changer=excluded.game_changer, legal_commander=excluded.legal_commander,
            can_be_commander=excluded.can_be_commander, layout=excluded.layout,
            produced_mana=excluded.produced_mana, is_land=excluded.is_land
        """,
        rows,
    )
    conn.commit()


def _write_card_names(conn: sqlite3.Connection, name_to_oracle_id: dict[str, str]) -> None:
    conn.executemany(
        """
        INSERT INTO card_names (name_normalized, oracle_id) VALUES (?, ?)
        ON CONFLICT(name_normalized) DO UPDATE SET oracle_id = excluded.oracle_id
        """,
        list(name_to_oracle_id.items()),
    )
    conn.commit()


def _write_ingest_log(conn: sqlite3.Connection, *, total: int, kept: int, aliases: int, skipped: int) -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, run_at, items, unresolved, notes) VALUES (?, ?, ?, ?, ?)",
        (
            "scryfall",
            datetime.now(timezone.utc).isoformat(),
            kept,
            skipped,
            f"{total} cards in bulk file, {kept} commander-legal cards kept, "
            f"{aliases} name aliases, {skipped} skipped (no oracle_id, e.g. reversible_card layout)",
        ),
    )
    conn.commit()


@dataclass
class IngestStats:
    total_cards: int
    kept_cards: int
    alias_count: int
    skipped_no_oracle_id: int
    download_bytes: int


def run(conn: sqlite3.Connection, session: RateLimitedSession | None = None) -> IngestStats:
    session = session or get_session("scryfall")

    bulk_info = get_bulk_data_info(session)
    dest = CONFIG.paths.raw_dir / "oracle_cards.jsonl.gz"
    download_bytes = download_bulk_file(session, bulk_info["jsonl_download_uri"], dest)

    total_cards = 0
    skipped_no_oracle_id = 0
    card_rows: list[dict[str, Any]] = []
    name_to_oracle_id: dict[str, str] = {}

    for card in iter_bulk_cards(dest):
        total_cards += 1

        oracle_id = card.get("oracle_id")
        if not oracle_id:
            skipped_no_oracle_id += 1
            continue
        if not is_commander_legal(card):
            continue

        card_rows.append(build_card_row(card))
        for alias in {card["name"], *_face_names(card)}:
            key = normalize_name(alias)
            if key:
                name_to_oracle_id[key] = oracle_id

    _write_cards(conn, card_rows)
    _write_card_names(conn, name_to_oracle_id)
    _write_ingest_log(
        conn,
        total=total_cards,
        kept=len(card_rows),
        aliases=len(name_to_oracle_id),
        skipped=skipped_no_oracle_id,
    )

    return IngestStats(
        total_cards=total_cards,
        kept_cards=len(card_rows),
        alias_count=len(name_to_oracle_id),
        skipped_no_oracle_id=skipped_no_oracle_id,
        download_bytes=download_bytes,
    )


def main() -> None:
    with connect(CONFIG.paths.db_path) as conn:
        stats = run(conn)
    print(
        f"Scryfall ingest complete: {stats.kept_cards:,}/{stats.total_cards:,} cards kept, "
        f"{stats.alias_count:,} name aliases, {stats.skipped_no_oracle_id} skipped, "
        f"{stats.download_bytes / 1_048_576:.1f} MB downloaded."
    )


if __name__ == "__main__":
    main()
