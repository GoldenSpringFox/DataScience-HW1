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
    -- Which configured commander slot this deck was harvested *for*, as an oracle_id for a
    -- single-commander slot or "primary_id+partner_id" for a partner slot (same convention
    -- as edhrec_card_stats.commander_key). Usually redundant with the commander columns, but
    -- not for partner-slot fallback decks: when too few decks run the exact pair, task 5.3-B
    -- backfills with decks running just one of the partners at the pair's color identity, so
    -- the commander columns hold that deck's *real* commanders while slot_key still groups it
    -- into the pair's corpus.
    slot_key TEXT,
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

-- Official Archidekt-curated precon decklists (`archidekt.com/commander-precons`), used to
-- compute decks.precon_similarity (task 5.3 QA — how much of a harvested corpus is a
-- barely-modified precon copy). `precon_id` is Archidekt's own deck id for the precon
-- decklist, reused directly rather than a separate autoincrement key.
CREATE TABLE IF NOT EXISTS precons (
    precon_id INTEGER PRIMARY KEY,
    set_name TEXT NOT NULL,   -- e.g. "Midnight Hunt Commander (MIC)", Archidekt's own folder name
    set_code TEXT,            -- e.g. "MIC", parsed from set_name's trailing "(...)"
    deck_name TEXT NOT NULL,
    url TEXT,
    commander_oracle_id TEXT,
    partner_oracle_id TEXT,   -- NULL unless the precon ships two declared commanders
    -- Other legendary creatures (or "can be your commander" cards) in this precon's own
    -- 100-card list that share the declared commander(s)' exact combined color identity —
    -- real alternative-commander candidates a player might swap in instead of the box's
    -- printed commander(s). JSON array of oracle_ids.
    alternative_commander_oracle_ids TEXT,
    fetched_at TEXT
);

-- The precon's full 100-card list, commander(s) included (unlike deck_cards, which excludes
-- them) — a precon's card list is fixed regardless of which of its legendaries someone
-- chooses to run as commander, so the commander(s) belong in the list like any other card.
CREATE TABLE IF NOT EXISTS precon_cards (
    precon_id INTEGER NOT NULL REFERENCES precons(precon_id),
    oracle_id TEXT NOT NULL REFERENCES cards(oracle_id),
    qty INTEGER NOT NULL,
    PRIMARY KEY (precon_id, oracle_id)
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
    "precons",
    "precon_cards",
    "card_tags",
    "edhrec_card_stats",
    "edhrec_themes",
    "ingest_log",
)


# Columns added to a table *after* it first shipped. `CREATE TABLE IF NOT EXISTS` is a no-op
# against an already-created table, so a database populated before the column existed would
# never gain it — these are ALTERed in explicitly. Keep in sync with the CREATE TABLE
# statements above (a fresh database gets them from there and skips the ALTER).
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "decks": {"slot_key": "TEXT"},
}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, declaration in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def create_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: safe to call against an existing DB."""
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
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
