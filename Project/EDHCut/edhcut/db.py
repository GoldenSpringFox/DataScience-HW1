"""SQLite schema (plan `EDHCut_PLAN.md` §2.3) + connection helper.

`oracle_id` is the canonical card key everywhere (plan §2.2); every table that references a
card stores it, never a raw name.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    oracle_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    mana_cost TEXT,
    cmc REAL,
    type_line TEXT,
    oracle_text TEXT,
    colors TEXT,            -- JSON array as TEXT
    color_identity TEXT,    -- JSON array as TEXT
    keywords TEXT,           -- JSON array as TEXT
    rarity TEXT,
    edhrec_rank INTEGER,
    price_usd REAL,
    game_changer BOOLEAN,
    legal_commander BOOLEAN,
    can_be_commander BOOLEAN,
    layout TEXT,
    produced_mana TEXT,      -- JSON array as TEXT
    is_land BOOLEAN
);

CREATE TABLE IF NOT EXISTS card_names (
    name_normalized TEXT PRIMARY KEY,
    oracle_id TEXT NOT NULL REFERENCES cards(oracle_id)
);

CREATE TABLE IF NOT EXISTS decks (
    deck_id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    url TEXT,
    commander_oracle_id TEXT,
    partner_oracle_id TEXT,  -- NULL if no partner
    fetched_at TEXT,
    views INTEGER,
    updated_at TEXT,
    precon_similarity REAL,  -- filled by task 5.3 QA: overlap with known precon list
    UNIQUE(source, source_id)
);

CREATE TABLE IF NOT EXISTS deck_cards (
    deck_id INTEGER NOT NULL REFERENCES decks(deck_id),
    oracle_id TEXT NOT NULL REFERENCES cards(oracle_id),
    qty INTEGER NOT NULL,
    PRIMARY KEY (deck_id, oracle_id)
);

CREATE TABLE IF NOT EXISTS card_tags (
    oracle_id TEXT NOT NULL REFERENCES cards(oracle_id),
    tag TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'tagger_bulk' | 'textmine' | 'manual'
    PRIMARY KEY (oracle_id, tag, source)
);

CREATE TABLE IF NOT EXISTS edhrec_card_stats (
    commander_key TEXT NOT NULL,  -- single oracle_id, or "id1+id2" for partners
    oracle_id TEXT NOT NULL,
    inclusion_rate REAL,
    synergy_score REAL,
    num_decks INTEGER,
    category TEXT,
    PRIMARY KEY (commander_key, oracle_id)
);

CREATE TABLE IF NOT EXISTS edhrec_themes (
    commander_key TEXT NOT NULL,
    theme TEXT NOT NULL,
    num_decks INTEGER,
    PRIMARY KEY (commander_key, theme)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    source TEXT NOT NULL,
    run_at TEXT NOT NULL,
    items INTEGER,
    unresolved INTEGER,
    notes TEXT
);
"""

TABLE_NAMES: tuple[str, ...] = (
    "cards",
    "card_names",
    "decks",
    "deck_cards",
    "card_tags",
    "edhrec_card_stats",
    "edhrec_themes",
    "ingest_log",
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: safe to call against an existing DB."""
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open (creating parent dirs + schema as needed) and yield a connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        create_schema(conn)
        yield conn
    finally:
        conn.close()
