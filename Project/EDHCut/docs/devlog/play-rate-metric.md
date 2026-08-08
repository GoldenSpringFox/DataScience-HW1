# Card play-rate metric
Date: 2026-08-08 | Status: done | Ad hoc per explicit user request (not a numbered plan task)

## Goal
`play_rate = (decks running the card) / (decks that could legally run it)`. The denominator
matters: a card's color identity restricts which decks may run it at all, so comparing raw
`deck_cards` counts across cards of different colors conflates "nobody wants this" with "almost
nobody could legally play this" (a 5-color card is eligible in far fewer decks than a colorless
one, even at identical popularity within its own eligible pool).

## Implementation
`edhcut/analysis/playrate.py`, 14 new tests (`tests/test_playrate.py`, 264 total in the package).

- **`color_identity_deck_counts.parquet`**: a fixed 32-row table (one row per possible color
  identity — the powerset of WUBRG, `""` colorless through `"WUBRG"` five-color), each with how
  many harvested decks have *exactly* that identity — the union of the deck's commander(s)' own
  `cards.color_identity` (a partner deck's identity is both commanders' union, the same rule
  Commander deckbuilding itself uses for legality). Zero-filled for any identity no harvested
  deck currently has, so the eligible-deck sum below is always over a known-complete 32-entry
  universe, not just whatever combinations happen to appear.
- **`card_play_rates.parquet`**: `oracle_id`, `name`, `color_identity`, `deck_count` (from
  `deck_cards`), `eligible_deck_count` (sum of `color_identity_deck_counts.deck_count` over
  every row whose identity is a *superset* of the card's — a card is includable iff its color
  identity is a subset of the deck's), `play_rate`. Scoped to `cards.legal_commander = 1` (same
  universe `embeddings.py`'s `tfidf` space uses), not just the >=3-deck pool `cooccurrence.py`
  uses — play rate is well-defined even for a card in zero decks. `play_rate` is `NA` (not 0)
  wherever `eligible_deck_count` is 0, a real "undefined" rather than a misleading 0/0.
- CLI: `python -m edhcut.analysis.playrate build`, `... show "<card name>"`.

## Results & an important scope caveat
Built against the live corpus: 3,921 decks, 31,623 commander-legal cards.
`Sol Ring [colorless]: 3,321 / 3,921 = 84.70%`, `Cultivate [G]: 698 / 1,585 = 44.04%`,
`Avacyn's Pilgrim [WG]: 957 / 1,585 = 60.38%` (its color identity is genuinely GW, not mono-W —
mana cost `{G}` counts too, even though its own ability only produces `{W}`).

**Only 7 of the 32 possible color identities have any decks at all** — R (2,006), WG (1,583),
WR (300), U (28), WRG (2), WU (1), BR (1) — because this project's harvest is deliberately
scoped to a fixed 5-commander-slot roster (`edhcut/config.py`'s `COMMANDER_SLOTS`), not a random
sample of the whole Commander metagame. Concretely: Cultivate's and Avacyn's Pilgrim's
`eligible_deck_count` both landed on the same 1,585 above not by coincidence but because every
G-identity deck in this corpus happens to also be W (the WG + WRG rows) — there's no
standalone-`G` slot harvested. **This metric is accurate on the harvested corpus but reports
"play rate among this project's 5 commander slots," not a metagame-wide figure** — the same
scope limitation every other analysis module (`cooccurrence.py`, `embeddings.py`) already
carries, not a new one introduced here.
