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
    is_land BOOLEAN,
    image_uri TEXT           -- Scryfall CDN "normal"-size image (front face for multi-face cards)
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

-- Per-card "did real deckbuilders keep this, or cut it?" evidence, scoped per precon AND per
-- the commander context evaluating it (`commander_key`, same bare-oracle_id/"id1+id2"
-- convention as `edhrec_card_stats`) — the same precon's card list can be evaluated
-- differently depending on which commander a deck actually runs (a card can be great to keep
-- under one commander and dead weight under another, e.g. a precon's own declared commander
-- showing up as a mere alternative in someone else's build). Long/tall format so adding more
-- commanders never means adding columns: `similar_deck_count` is how many decks running this
-- commander_key were judged close enough to this precon (see edhcut.ingest.precon_retention's
-- qty-aware card-difference threshold) to count as evidence at all; `kept_count` is how many
-- of those still ran this specific card. Deliberately stores raw counts, not a %cut column —
-- a precon/commander pairing with only a handful of similar decks shouldn't produce a
-- confident-looking percentage (same reasoning as precon_similarity.jaccard_similarity
-- returning None rather than 0.0 on no data).
-- `weighted_cut`/`weighted_kept` are a more nuanced companion signal, summed across *every*
-- deck matching this (precon, commander_key) pair regardless of `similar_deck_count`'s hard
-- threshold: each deck contributes a confidence weight derived from how many cards it changed
-- overall (edhcut.ingest.precon_retention.cut_confidence) -- a deck that barely touched the
-- precon gives strong "cut" evidence for the few cards it did remove and near-zero "kept"
-- evidence for the untouched rest (nothing to conclude from cards nobody bothered to look
-- at); a heavily rebuilt deck gives the opposite -- weak "cut" evidence per swap (everything
-- was in flux) but strong "kept" evidence for whatever specific cards survived a near-total
-- rebuild. Floats, not bounded to [0,1] -- they're sums of per-deck weights across however
-- many decks contributed.
CREATE TABLE IF NOT EXISTS precon_card_retention (
    precon_id INTEGER NOT NULL REFERENCES precons(precon_id),
    commander_key TEXT NOT NULL,
    oracle_id TEXT NOT NULL REFERENCES cards(oracle_id),
    similar_deck_count INTEGER NOT NULL,
    kept_count INTEGER NOT NULL,
    weighted_cut REAL NOT NULL DEFAULT 0,
    weighted_kept REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (precon_id, commander_key, oracle_id)
);

CREATE TABLE IF NOT EXISTS card_tags (
    oracle_id TEXT NOT NULL REFERENCES cards(oracle_id),
    tag TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'tagger_bulk' | 'textmine' | 'manual'
    PRIMARY KEY (oracle_id, tag, source)
);

-- Scryfall Tagger tag aliases (e.g. "boardwipe" -> "sweeper", matching what Scryfall's own
-- `otag:` search resolves) -- kept separate from `card_tags` rather than inserted as
-- duplicate rows there, so co-occurrence/clustering (task 6.x) doesn't double-count the same
-- real tag under two different strings. `alias_normalized` uses the same normalization as
-- `card_names` (`scryfall.normalize_name`). No collisions found live between aliases or
-- against real tag slugs (checked directly against the bulk file), so a bare-alias primary
-- key is safe -- if that ever stops holding, this needs revisiting.
CREATE TABLE IF NOT EXISTS tag_aliases (
    alias_normalized TEXT PRIMARY KEY,
    tag TEXT NOT NULL
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

-- EDHREC's site-wide tag popularity ranking — one row per tag that exists anywhere on EDHREC,
-- with its global deck count, independent of any one commander. Used as the fixed
-- theme/typal vocabulary for the clustering/labeling step (task 6.6), not derived from our
-- own harvested corpus. Not to be confused with `edhrec_themes_per_commander`, which is
-- scoped to a single commander slot. EDHREC splits its own tag browser into two kinds of tag
-- (`edhrec.com/tags/themes` — mechanical/archetype tags like "Aristocrats"; `/tags/typal` —
-- tribal/creature-type tags like "Goblins") — both land here, distinguished by `kind`.
-- Composite `(kind, theme)` primary key rather than `theme` alone: confirmed empirically (5.4-C
-- vs 5.4-D) that the two lists don't currently collide on a name, but nothing guarantees that
-- stays true as EDHREC adds tags, so don't rely on cross-kind uniqueness.
CREATE TABLE IF NOT EXISTS edhrec_themes (
    theme TEXT NOT NULL,
    kind TEXT NOT NULL,      -- 'theme' | 'typal'
    slug TEXT NOT NULL,      -- from the tag's own "/tags/<slug>" url, stable across renames
    num_decks INTEGER,       -- global popularity: decks site-wide tagged with this theme
    fetched_at TEXT,
    PRIMARY KEY (kind, theme)
);

-- Per-commander-slot theme tag counts (task 5.4-B) — how many decks *for this specific
-- commander* carry each tag, from that commander's own EDHREC page. `num_decks` here is
-- local to the slot, not comparable across different commanders' corpora sizes.
CREATE TABLE IF NOT EXISTS edhrec_themes_per_commander (
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
    "precon_card_retention",
    "card_tags",
    "tag_aliases",
    "edhrec_card_stats",
    "edhrec_themes",
    "edhrec_themes_per_commander",
    "ingest_log",
)


# Columns added to a table *after* it first shipped. `CREATE TABLE IF NOT EXISTS` is a no-op
# against an already-created table, so a database populated before the column existed would
# never gain it — these are ALTERed in explicitly. Keep in sync with the CREATE TABLE
# statements above (a fresh database gets them from there and skips the ALTER).
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "decks": {"slot_key": "TEXT"},
    "precon_card_retention": {
        "weighted_cut": "REAL NOT NULL DEFAULT 0",
        "weighted_kept": "REAL NOT NULL DEFAULT 0",
    },
    "cards": {"image_uri": "TEXT"},
}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, declaration in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _migrate_legacy_edhrec_themes_table(conn: sqlite3.Connection) -> None:
    """One-time fixups for `edhrec_themes` across its two schema changes. Must run *before*
    `SCHEMA` is executed, so the fresh `CREATE TABLE IF NOT EXISTS edhrec_themes` isn't a
    no-op against an old shape.

    1. (task 5.4-C) DBs created before `edhrec_themes` was repurposed for the global theme
       popularity ranking (it used to be the per-commander-slot table, now called
       `edhrec_themes_per_commander`) — detected by the old table's `commander_key` column,
       which the global-themes schema doesn't have. Renamed rather than dropped, so the
       already-harvested per-commander rows aren't lost.
    2. (task 5.4-D) DBs created with 5.4-C's schema (`theme` alone as primary key) before
       `kind` ('theme' vs. 'typal') was added — detected by the missing `kind` column.
       SQLite can't ALTER a primary key in place, and unlike per-commander data this table is
       wholly derived/re-fetchable (a handful of cached page fetches, not hours of harvesting)
       — so this one's just dropped and left for the fresh `CREATE TABLE` to rebuild correctly
       on next `python -m edhcut.ingest.edhrec` run, rather than hand-writing a copy-migration
       for disposable data."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(edhrec_themes)")}
    if "commander_key" in columns:
        conn.execute("ALTER TABLE edhrec_themes RENAME TO edhrec_themes_per_commander")
    elif columns and "kind" not in columns:
        conn.execute("DROP TABLE edhrec_themes")


def create_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: safe to call against an existing DB."""
    _migrate_legacy_edhrec_themes_table(conn)
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
