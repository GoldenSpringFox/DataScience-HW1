"""Card play-rate metric (ad hoc, per explicit user request 2026-08-08 -- not a numbered plan
task): for each card, `play_rate = decks_running_it / decks_that_could_run_it`.

The denominator is the real subtlety: a card's color identity restricts which decks may run it
at all (Commander's own deckbuilding rule -- a card is legal only if its color identity is a
subset of the deck's commander(s)' combined color identity), so "decks that could run it" is
*not* every harvested deck, and comparing raw `deck_cards` counts across cards of different
colors would conflate "nobody wants this" with "almost nobody could legally play this in the
first place" (e.g. a 5-color card is eligible in far fewer decks than a colorless one, even at
identical popularity within its eligible pool).

`color_identity_deck_counts.parquet` is the fixed 32-row universe this denominator sums over:
one row per one of the 32 possible color identities (the powerset of WUBRG, `""` colorless
through `"WUBRG"` five-color), each with how many harvested decks have *exactly* that identity
(the union of the deck's commander(s)' own `cards.color_identity` -- the actual legality-
determining value, not raw card count or commander popularity). All 32 rows are always present,
zero-filled where no harvested deck happens to have that identity, so a card's eligible-deck
count is a sum over a known-complete universe rather than silently missing whatever combination
happens not to appear yet in a still-growing corpus.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.ingest.scryfall import normalize_name

KB_DEV_DIR = CONFIG.paths.kb_dir / "dev"

_WUBRG = "WUBRG"

# Every one of the 32 possible color identities (powerset of WUBRG), each as a canonical
# WUBRG-ordered string -- "" (colorless) through "WUBRG" (five-color) -- the fixed universe
# `color_identity_deck_counts` always has one row per, and `eligible_deck_count` sums subsets
# over.
ALL_COLOR_IDENTITIES: list[str] = [
    "".join(combo)
    for r in range(len(_WUBRG) + 1)
    for combo in itertools.combinations(_WUBRG, r)
]


def canonical_identity(colors: Iterable[str]) -> str:
    """Colors in any order/case -> canonical WUBRG-ordered string, e.g. `{"G", "U"}` -> `"UG"`."""
    colors = {c.upper() for c in colors}
    return "".join(c for c in _WUBRG if c in colors)


def build_deck_color_identities(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per deck: `deck_id`, `color_identity` -- the union of the deck's commander(s)'
    own `cards.color_identity` (a partner deck's identity is both commanders' union, same rule
    Commander deckbuilding itself uses for legality). Every `decks` row has a non-NULL
    `commander_oracle_id` (5.3-B's own harvest guarantee); `partner_oracle_id` is NULL for a
    single-commander deck."""
    rows = conn.execute(
        """
        SELECT d.deck_id, c1.color_identity, c2.color_identity
        FROM decks d
        JOIN cards c1 ON c1.oracle_id = d.commander_oracle_id
        LEFT JOIN cards c2 ON c2.oracle_id = d.partner_oracle_id
        """
    ).fetchall()

    records = []
    for deck_id, ci1_json, ci2_json in rows:
        colors = set(json.loads(ci1_json or "[]"))
        if ci2_json is not None:
            colors |= set(json.loads(ci2_json or "[]"))
        records.append({"deck_id": deck_id, "color_identity": canonical_identity(colors)})
    return pd.DataFrame(records, columns=["deck_id", "color_identity"])


def build_color_identity_deck_counts(conn: sqlite3.Connection) -> pd.DataFrame:
    """The fixed 32-row `color_identity` -> `deck_count` table (see module docstring) --
    zero-filled for any of the 32 possible identities no harvested deck currently has."""
    deck_identities = build_deck_color_identities(conn)
    counts = deck_identities["color_identity"].value_counts()
    return pd.DataFrame(
        {
            "color_identity": ALL_COLOR_IDENTITIES,
            "deck_count": [int(counts.get(ci, 0)) for ci in ALL_COLOR_IDENTITIES],
        }
    )


def eligible_deck_count(color_identity_counts: pd.DataFrame, card_color_identity: str) -> int:
    """How many decks could legally run a card with this color identity: the sum of
    `deck_count` over every row in the 32-row table whose own identity is a *superset* of the
    card's (a card is includable iff its color identity is a subset of the deck's)."""
    card_colors = set(card_color_identity)
    is_superset = color_identity_counts["color_identity"].map(lambda ci: card_colors <= set(ci))
    return int(color_identity_counts.loc[is_superset, "deck_count"].sum())


def build_card_play_rates(
    conn: sqlite3.Connection, color_identity_counts: pd.DataFrame | None = None
) -> pd.DataFrame:
    """`oracle_id`, `name`, `color_identity`, `deck_count` (decks actually running the card, from
    `deck_cards` -- commanders themselves are never in `deck_cards`, so a commander's own row
    here reflects only decks where it appears as a *non*-commander include, if any),
    `eligible_deck_count`, `play_rate` (`deck_count / eligible_deck_count`). Scoped to
    `cards.legal_commander = 1`, the same universe `embeddings.py`'s `tfidf` space uses, rather
    than only cards already known to appear in >=3 decks -- play rate is well-defined (and
    informative) even for a card in zero decks. `play_rate` is `NA` wherever
    `eligible_deck_count` is 0 -- a color identity with no harvested deck of its own yet -- a
    real "undefined," not a misleading 0/0 collapsed to zero."""
    if color_identity_counts is None:
        color_identity_counts = build_color_identity_deck_counts(conn)

    played = pd.DataFrame(
        conn.execute("SELECT oracle_id, COUNT(DISTINCT deck_id) FROM deck_cards GROUP BY oracle_id").fetchall(),
        columns=["oracle_id", "deck_count"],
    )

    cards = pd.DataFrame(
        conn.execute(
            "SELECT oracle_id, name, color_identity FROM cards WHERE legal_commander = 1"
        ).fetchall(),
        columns=["oracle_id", "name", "color_identity_raw"],
    )
    cards["color_identity"] = cards["color_identity_raw"].map(
        lambda raw: canonical_identity(json.loads(raw or "[]"))
    )
    cards = cards.drop(columns=["color_identity_raw"])

    merged = cards.merge(played, on="oracle_id", how="left")
    merged["deck_count"] = merged["deck_count"].fillna(0).astype(int)

    eligible_by_identity = {
        ci: eligible_deck_count(color_identity_counts, ci) for ci in merged["color_identity"].unique()
    }
    merged["eligible_deck_count"] = merged["color_identity"].map(eligible_by_identity)
    merged["play_rate"] = merged["deck_count"] / merged["eligible_deck_count"].replace(0, pd.NA)

    return merged[["oracle_id", "name", "color_identity", "deck_count", "eligible_deck_count", "play_rate"]]


def build_and_save(conn: sqlite3.Connection, *, out_dir: Path = KB_DEV_DIR) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    color_identity_counts = build_color_identity_deck_counts(conn)
    color_identity_counts.to_parquet(out_dir / "color_identity_deck_counts.parquet", index=False)

    play_rates = build_card_play_rates(conn, color_identity_counts)
    play_rates.to_parquet(out_dir / "card_play_rates.parquet", index=False)

    return {
        "n_color_identities": len(color_identity_counts),
        "n_decks": int(color_identity_counts["deck_count"].sum()),
        "n_cards": len(play_rates),
    }


def load_color_identity_deck_counts(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "color_identity_deck_counts.parquet")


def load_card_play_rates(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "card_play_rates.parquet")


def _resolve_card_name(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute(
        "SELECT oracle_id FROM card_names WHERE name_normalized = ?", (normalize_name(name),)
    ).fetchone()
    if row is None:
        raise SystemExit(f"Card {name!r} not found in card_names.")
    return row[0]


def _cmd_build(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        stats = build_and_save(conn)
    print(f"Built play-rate tables under {KB_DEV_DIR}:")
    print(json.dumps(stats, indent=2))


def _cmd_show(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        oracle_id = _resolve_card_name(conn, args.card)
    table = load_card_play_rates()
    row = table.loc[table["oracle_id"] == oracle_id]
    if row.empty:
        print(f"{args.card!r} not found in card_play_rates (not commander-legal?).")
        return
    row = row.iloc[0]
    rate = row["play_rate"]
    rate_str = f"{rate:.2%}" if pd.notna(rate) else "n/a"
    print(
        f"{row['name']} [{row['color_identity'] or 'colorless'}]: "
        f"{row['deck_count']:,} / {row['eligible_deck_count']:,} eligible decks "
        f"= {rate_str} play rate"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build & save the play-rate tables to data/kb/dev/")
    build_p.set_defaults(func=_cmd_build)

    show_p = sub.add_parser("show", help="Show one card's play rate")
    show_p.add_argument("card", help="Card name")
    show_p.set_defaults(func=_cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
