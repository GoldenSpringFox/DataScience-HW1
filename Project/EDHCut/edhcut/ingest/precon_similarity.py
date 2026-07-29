"""Backfill `decks.precon_similarity` (plan §5.3 QA — precon-contamination detection).

Pure DB computation, no network access — runs against `decks`/`deck_cards` (task 5.3-B) and
`precons`/`precon_cards` (`edhcut.ingest.precons`), both of which must already be populated.

**Matching a deck to a precon**: not by commander *slot* — by each deck's own actual
commander(s), which can differ from the slot's nominal commander(s) for partner-fallback
decks (task 5.3-B's `slot_key` mechanism). A deck matches a precon if either of its
commanders is that precon's declared commander/partner, *or* one of its listed
`alternative_commander_oracle_ids` — the latter is exactly the Kyler case the plan names:
Kyler, Sigardian Emissary never shipped as a precon's own printed commander, only as an
alternative inside "Coven Counters" (declared commander Leinore, Autumn Sovereign). A deck
can match more than one precon (e.g. a card reprinted as an alternative across multiple
sets); `precon_similarity` takes the *best* (highest-Jaccard) match, since the question is
"is this deck basically a copy of some precon," not "of this one specific precon."

**Similarity**: Jaccard over each side's *library* card set — plain set of distinct
`oracle_id`s, so a deck's `deck_cards` (already commander-excluded, per task 5.3-B) is
compared as-is; a precon's `precon_cards` has the *matching deck's own* commander(s)
subtracted out too (not just the precon's own declared commander), so a card like Krenko
that shows up as merely an *alternative* in a precon still gets excluded from both sides
symmetrically, rather than only being absent from the deck side and inflating the mismatch.
Quantities are ignored (e.g. 15 vs. 20 Mountains counts as a full match on "Mountain") —
this is a card*-selection* similarity, not a physical-copy one.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field

from edhcut.config import CONFIG
from edhcut.db import connect


def matching_precon_ids(conn: sqlite3.Connection, commander_oracle_ids: list[str]) -> list[int]:
    """Precons where any of these oracle_ids is the declared commander, the declared
    partner, or a listed alternative commander."""
    if not commander_oracle_ids:
        return []
    placeholders = ",".join("?" * len(commander_oracle_ids))
    rows = conn.execute(
        f"""
        SELECT precon_id, alternative_commander_oracle_ids FROM precons
        WHERE commander_oracle_id IN ({placeholders})
           OR partner_oracle_id IN ({placeholders})
        """,
        tuple(commander_oracle_ids) * 2,
    ).fetchall()
    matched = {precon_id for precon_id, _ in rows}

    # alternative_commander_oracle_ids is a JSON array per row — no useful index to push
    # this into SQL, so scan it in Python. The precons table is small (~180 rows), so this
    # is a full-table scan either way, just done client-side.
    wanted = set(commander_oracle_ids)
    for precon_id, alt_json in conn.execute(
        "SELECT precon_id, alternative_commander_oracle_ids FROM precons"
    ):
        if precon_id in matched:
            continue
        alt_ids = json.loads(alt_json) if alt_json else []
        if wanted & set(alt_ids):
            matched.add(precon_id)

    return sorted(matched)


def _deck_card_oracle_ids(conn: sqlite3.Connection, deck_id: int) -> set[str]:
    return {row[0] for row in conn.execute(
        "SELECT oracle_id FROM deck_cards WHERE deck_id = ?", (deck_id,)
    )}


def _precon_card_oracle_ids(
    conn: sqlite3.Connection, precon_id: int, *, exclude: set[str]
) -> set[str]:
    return {
        row[0] for row in conn.execute(
            "SELECT oracle_id FROM precon_cards WHERE precon_id = ?", (precon_id,)
        )
        if row[0] not in exclude
    }


def jaccard_similarity(a: set[str], b: set[str]) -> float | None:
    """None (not 0.0) when both sides are empty — "no overlap" and "nothing to compare"
    are different states, and the latter shouldn't silently look like a confirmed
    zero-similarity result."""
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def compute_deck_precon_similarity(
    conn: sqlite3.Connection, deck_id: int, commander_oracle_ids: list[str]
) -> float | None:
    """Best (highest) Jaccard similarity between this deck's library and any precon its
    commander(s) match, or None if no precon matches at all."""
    precon_ids = matching_precon_ids(conn, commander_oracle_ids)
    if not precon_ids:
        return None

    deck_cards = _deck_card_oracle_ids(conn, deck_id)
    own_commanders = set(commander_oracle_ids)
    best: float | None = None
    for precon_id in precon_ids:
        precon_cards = _precon_card_oracle_ids(conn, precon_id, exclude=own_commanders)
        similarity = jaccard_similarity(deck_cards, precon_cards)
        if similarity is not None and (best is None or similarity > best):
            best = similarity
    return best


@dataclass
class PreconSimilarityStats:
    decks_checked: int = 0
    decks_matched: int = 0  # matched at least one precon (similarity computed, non-NULL)
    similarities: list[float] = field(default_factory=list)


def backfill_precon_similarity(
    conn: sqlite3.Connection, *, slot_key: str | None = None
) -> PreconSimilarityStats:
    """Recompute `decks.precon_similarity` for every deck (or just one `slot_key`'s decks).
    Unmatched decks are set back to NULL — re-running after a precons re-harvest shouldn't
    leave a stale similarity behind if a match no longer applies."""
    stats = PreconSimilarityStats()
    query = "SELECT deck_id, commander_oracle_id, partner_oracle_id FROM decks"
    params: tuple = ()
    if slot_key is not None:
        query += " WHERE slot_key = ?"
        params = (slot_key,)

    rows = conn.execute(query, params).fetchall()
    for deck_id, commander_oracle_id, partner_oracle_id in rows:
        stats.decks_checked += 1
        commander_ids = [oid for oid in (commander_oracle_id, partner_oracle_id) if oid]
        similarity = compute_deck_precon_similarity(conn, deck_id, commander_ids)
        conn.execute(
            "UPDATE decks SET precon_similarity = ? WHERE deck_id = ?", (similarity, deck_id)
        )
        if similarity is not None:
            stats.decks_matched += 1
            stats.similarities.append(similarity)
    conn.commit()
    return stats


def _print_summary(stats: PreconSimilarityStats, *, label: str) -> None:
    print(
        f"{label}: {stats.decks_matched}/{stats.decks_checked} decks matched a precon "
        f"(commander shipped in one, as declared or alternative)"
    )
    if not stats.similarities:
        return
    values = sorted(stats.similarities)
    n = len(values)
    quantile = lambda q: values[min(n - 1, int(q * n))]  # noqa: E731 - tiny, local-only
    print(
        f"  similarity distribution: min={values[0]:.2f} p25={quantile(0.25):.2f} "
        f"median={quantile(0.5):.2f} p75={quantile(0.75):.2f} max={values[-1]:.2f}"
    )
    near_precon = sum(1 for v in values if v > 0.9)
    print(f"  {near_precon}/{n} decks are >90% overlap with their matched precon")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill decks.precon_similarity against the precons/precon_cards tables."
    )
    parser.add_argument(
        "--slot-key", default=None,
        help="Only recompute decks for this slot_key (default: every deck in the database).",
    )
    args = parser.parse_args()

    with connect(CONFIG.paths.db_path) as conn:
        stats = backfill_precon_similarity(conn, slot_key=args.slot_key)

    _print_summary(stats, label=args.slot_key or "all decks")


if __name__ == "__main__":
    main()
