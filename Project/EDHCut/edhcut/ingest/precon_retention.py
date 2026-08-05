"""Backfill `precon_card_retention` — per-card evidence of whether real deckbuilders kept or
cut each precon card, scoped by precon AND by the commander context evaluating it. Follow-up
to task 6.1: a whole-*deck* novelty weight (`edhcut.analysis.cooccurrence.deck_weight`) turned
out too blunt for precon contamination, since it penalizes a card a deckbuilder consciously
kept just as much as a card only left in by inertia. This module builds the per-*card* signal
needed to fix that (see `docs/devlog/6.1-cooccurrence.md` and the design discussion that led
here) — `cooccurrence.py` itself isn't touched yet; that's a deliberate follow-up once this
table exists.

Pure DB computation, no network access — runs against `decks`/`deck_cards` (5.3-B) and
`precons`/`precon_cards` (`edhcut.ingest.precons`), same prerequisites as
`edhcut.ingest.precon_similarity`.

**"Similar" deck test — quantity-aware, not a percentage.** A raw shared-*distinct*-card count
(or a percentage of one) breaks because precon list sizes vary a lot in *distinct* oracle_ids —
basic lands are stored as one row with a `qty`, so a precon shipping 20 Forests as a single row
"shares" only 1 distinct card no matter how many copies actually overlap. Checked live before
picking a metric: `precon_cards` distinct-oracle_id counts range 48-100 (avg 82.5) even though
real physical card totals (`SUM(qty)`) are consistently ~99-100 — Kyler's own matched precon
("Coven Counters") has only 78 *distinct* cards despite being a full 100-card box, entirely
from stacked basics. A fixed distinct-card threshold would have been unreachable for the exact
commander this feature exists for. Fixed by comparing quantity-aware multisets instead:
`card_difference_count` is "how many of the deck's own physical cards aren't matched by an
available copy in the (commander-excluded) precon list" — 0 for an exact copy, growing as more
cards get swapped. A deck counts as "similar to precon P" when that difference is at most
`DIFF_THRESHOLD` (20, i.e. roughly "at most 20 of ~99 library cards changed").

**Commander scoping.** The same precon's card list means something different depending on
which commander a deck actually runs — a card can be worth keeping under one commander and
dead weight under another (sharpest example: a precon's own *declared* commander showing up as
a plain library card under whichever commander the deck's *actual* build uses instead, e.g.
Kyler, Sigardian Emissary as an alternative commander in "Coven Counters", whose declared
commander is Leinore, Autumn Sovereign — very different subthemes). So retention is grouped by
`(precon_id, commander_key)`, `commander_key` being the deck's own *actual* commander(s) — same
bare-oracle_id / "id1+id2" convention `edhrec_card_stats` uses — not the precon's declared
commander, and not `decks.slot_key` (which can differ from a deck's real commanders for
partner-fallback decks, task 5.3-B).

**Best match only.** If a deck's commander(s) match more than one precon (only happens via the
alternative-commander path today, e.g. Krenko), it contributes retention evidence to just the
single *best* (lowest-difference) match, not every precon it loosely resembles — avoids
double-counting the same deck's cut/stay signal across precons it isn't really closest to
(same "best match wins" principle `precon_similarity.compute_deck_precon_similarity` already
uses, just measured by card-difference here instead of Jaccard).

**`weighted_cut`/`weighted_kept` — a continuous companion to the threshold-gated counts above.**
`similar_deck_count`/`kept_count` treat "close enough" as a hard yes/no, which throws away
signal from every deck on the wrong side of `DIFF_THRESHOLD` and treats a diff-1 deck the same
as a diff-20 one. The actual intuition: a deck that barely touched the precon gives *strong*
evidence about the few cards it did cut (there was no reason to touch anything else, so that
cut was deliberate) but *weak* evidence about the untouched rest (inertia, not a conscious
keep); a heavily rebuilt deck is the mirror image — cutting any one card among fifty swaps
isn't very informative on its own, but a card that survived a near-total rebuild really was
chosen. So every deck matching a precon (regardless of `DIFF_THRESHOLD`) contributes to *both*
columns, weighted by `cut_confidence(diff)` (see its own docstring for the exact curve and how
it was fit) — no hard cutoff, just proportioned confidence.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.ingest.archidekt import slot_key_for
from edhcut.ingest.precon_similarity import matching_precon_ids

DIFF_THRESHOLD = 20

RETENTION_WEIGHT_MIDPOINT = 20.0
RETENTION_WEIGHT_SCALE = 10.0


def cut_confidence(diff: int) -> float:
    """How much weight a deck that changed `diff` cards from its matched precon contributes to
    a *cut* card's evidence; `1 - cut_confidence(diff)` is what it contributes to a *kept*
    card's evidence instead. Logistic curve, least-squares fit (in logit space) against the
    target shape requested for this feature:

        diff= 1 -> target ~1.00, actual 0.870  (barely touched: any cut was deliberate)
        diff=10 -> target  0.80, actual 0.731
        diff=20 -> target  0.50, actual 0.500  (midpoint, exact by construction)
        diff=50 -> target  0.05, actual 0.047  (heavily rebuilt: any survivor was chosen)

    `RETENTION_WEIGHT_MIDPOINT` (20, roughly "1/5 of a ~99-card library changed") is the diff
    at which cut/kept evidence is weighted equally; `RETENTION_WEIGHT_SCALE` (10) controls how
    sharply confidence shifts around that midpoint. Monotonic, no hard cutoff — every matching
    deck contributes *something* to both columns, just proportioned by how much it changed."""
    return 1.0 / (1.0 + math.exp((diff - RETENTION_WEIGHT_MIDPOINT) / RETENTION_WEIGHT_SCALE))


def multiset_overlap(a: dict[str, int], b: dict[str, int]) -> int:
    """Total quantity-aware overlap between two card multisets."""
    return sum(min(qty, b.get(oracle_id, 0)) for oracle_id, qty in a.items())


def card_difference_count(deck_cards: dict[str, int], precon_cards: dict[str, int]) -> int:
    """How many of the deck's own physical cards aren't matched by an available copy in the
    precon's list — 0 for an exact copy, growing as the deck diverges. Measured against the
    deck's own total (not a symmetric two-sided difference) so it stays meaningful even if a
    precon's own recorded list is short a few cards (e.g. from unresolved names)."""
    deck_total = sum(deck_cards.values())
    return deck_total - multiset_overlap(deck_cards, precon_cards)


def _deck_card_quantities(conn: sqlite3.Connection, deck_id: int) -> dict[str, int]:
    return dict(
        conn.execute("SELECT oracle_id, qty FROM deck_cards WHERE deck_id = ?", (deck_id,)).fetchall()
    )


def _precon_card_quantities(
    conn: sqlite3.Connection, precon_id: int, *, exclude: set[str]
) -> dict[str, int]:
    return {
        oracle_id: qty
        for oracle_id, qty in conn.execute(
            "SELECT oracle_id, qty FROM precon_cards WHERE precon_id = ?", (precon_id,)
        )
        if oracle_id not in exclude
    }


def best_matching_precon(
    conn: sqlite3.Connection, deck_cards: dict[str, int], commander_oracle_ids: list[str]
) -> tuple[int, int] | None:
    """`(precon_id, difference)` of the deck's closest (lowest-difference) matching precon, or
    None if no precon matches this deck's commander(s) at all (declared, partner, or
    alternative — see `matching_precon_ids`)."""
    precon_ids = matching_precon_ids(conn, commander_oracle_ids)
    if not precon_ids:
        return None
    own_commanders = set(commander_oracle_ids)
    best: tuple[int, int] | None = None
    for precon_id in precon_ids:
        precon_cards = _precon_card_quantities(conn, precon_id, exclude=own_commanders)
        diff = card_difference_count(deck_cards, precon_cards)
        if best is None or diff < best[1]:
            best = (precon_id, diff)
    return best


class _MatchedDeck(NamedTuple):
    deck_id: int
    diff: int
    card_ids: frozenset[str]


@dataclass
class RetentionStats:
    decks_checked: int = 0
    decks_matched_any: int = 0  # matched some precon at all (feeds weighted_cut/weighted_kept)
    decks_similar: int = 0  # subset within diff_threshold (feeds similar_deck_count/kept_count)
    groups: int = 0  # distinct (precon_id, commander_key) pairs with any evidence
    rows_written: int = 0


def backfill_precon_card_retention(
    conn: sqlite3.Connection, *, diff_threshold: int = DIFF_THRESHOLD
) -> RetentionStats:
    """Recomputes `precon_card_retention` from scratch (full delete + rewrite — this table is
    wholly derived from `decks`/`deck_cards`/`precon_cards`, cheap to rebuild, no reason to
    diff against the old contents)."""
    stats = RetentionStats()

    matches: dict[tuple[int, str], list[_MatchedDeck]] = defaultdict(list)
    for deck_id, commander_oracle_id, partner_oracle_id in conn.execute(
        "SELECT deck_id, commander_oracle_id, partner_oracle_id FROM decks"
    ):
        stats.decks_checked += 1
        commander_ids = [oid for oid in (commander_oracle_id, partner_oracle_id) if oid]
        if not commander_ids:
            continue
        deck_cards = _deck_card_quantities(conn, deck_id)
        match = best_matching_precon(conn, deck_cards, commander_ids)
        if match is None:
            continue
        precon_id, diff = match
        stats.decks_matched_any += 1
        if diff <= diff_threshold:
            stats.decks_similar += 1
        commander_key = slot_key_for(commander_ids)
        matches[(precon_id, commander_key)].append(
            _MatchedDeck(deck_id, diff, frozenset(deck_cards.keys()))
        )

    rows: list[tuple[int, str, str, int, int, float, float]] = []
    for (precon_id, commander_key), entries in matches.items():
        commander_ids = set(commander_key.split("+"))
        tracked_cards = {
            oracle_id
            for (oracle_id,) in conn.execute(
                "SELECT oracle_id FROM precon_cards WHERE precon_id = ?", (precon_id,)
            )
            if oracle_id not in commander_ids
        }

        similar_entries = [e for e in entries if e.diff <= diff_threshold]
        kept_counts: dict[str, int] = defaultdict(int)
        for entry in similar_entries:
            for card_id in tracked_cards & entry.card_ids:
                kept_counts[card_id] += 1

        weighted_cut: dict[str, float] = defaultdict(float)
        weighted_kept: dict[str, float] = defaultdict(float)
        for entry in entries:
            cut_weight = cut_confidence(entry.diff)
            kept_weight = 1.0 - cut_weight
            for card_id in tracked_cards:
                if card_id in entry.card_ids:
                    weighted_kept[card_id] += kept_weight
                else:
                    weighted_cut[card_id] += cut_weight

        for card_id in tracked_cards:
            rows.append((
                precon_id, commander_key, card_id,
                len(similar_entries), kept_counts.get(card_id, 0),
                weighted_cut.get(card_id, 0.0), weighted_kept.get(card_id, 0.0),
            ))
        stats.groups += 1

    conn.execute("DELETE FROM precon_card_retention")
    conn.executemany(
        "INSERT INTO precon_card_retention "
        "(precon_id, commander_key, oracle_id, similar_deck_count, kept_count, "
        "weighted_cut, weighted_kept) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    stats.rows_written = len(rows)
    return stats


def _print_summary(stats: RetentionStats) -> None:
    print(
        f"{stats.decks_matched_any}/{stats.decks_checked} decks matched some precon "
        f"(feeds weighted_cut/weighted_kept); {stats.decks_similar} of those are within "
        f"{DIFF_THRESHOLD} cards (feeds similar_deck_count/kept_count); {stats.groups} "
        f"(precon, commander) group(s), {stats.rows_written} precon_card_retention rows written."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill precon_card_retention: per-card keep/cut evidence from decks close to a precon."
    )
    parser.add_argument("--diff-threshold", type=int, default=DIFF_THRESHOLD)
    args = parser.parse_args()

    with connect(CONFIG.paths.db_path) as conn:
        stats = backfill_precon_card_retention(conn, diff_threshold=args.diff_threshold)

    _print_summary(stats)


if __name__ == "__main__":
    main()
