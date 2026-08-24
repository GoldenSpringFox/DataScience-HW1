# Card play-rate metric
Date: 2026-08-08 | Status: done | Ad hoc per user request (not a numbered plan task)

`play_rate = (decks running the card) / (decks that could legally run it)`. The denominator is
the point: comparing raw `deck_cards` counts across colors conflates "nobody wants this" with
"almost nobody could legally play it". `edhcut/analysis/playrate.py`, 14 tests.

## Decisions
- `color_identity_deck_counts.parquet` is a fixed 32-row table (the full WUBRG powerset),
  zero-filled, so the eligible-deck sum is always over a complete universe rather than
  whichever identities happen to appear.
- Eligibility is subset-based: a card counts against every deck identity that is a *superset*
  of its own. A partner deck's identity is the union of both commanders', matching the real
  deckbuilding rule.
- Scoped to `legal_commander = 1` (the `embeddings.py` universe), not `cooccurrence.py`'s
  ≥3-deck pool — play rate is well-defined for a card in zero decks.
- `play_rate` is `NA`, not 0, where `eligible_deck_count` is 0 — undefined, not measured-zero.

## Outcome
Done. Built against 3,921 decks / 31,623 cards; Sol Ring 84.70%, Cultivate 44.04%.

**Scope caveat:** only 7 of 32 color identities have any decks, because the harvest is scoped to
a fixed 5-commander roster. Cultivate and Avacyn's Pilgrim share an `eligible_deck_count` of
1,585 because every G deck in this corpus is also W. The metric is accurate *on this corpus* but
is not a metagame-wide figure — the same limitation every other analysis module carries.
