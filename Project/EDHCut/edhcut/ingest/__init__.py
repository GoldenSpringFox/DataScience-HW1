"""Ingest layer: one module per external source, each writing into `data/edhcut.db` and logging
the run to the `ingest_log` table. All network access goes through `edhcut.http`'s cached,
rate-limited session (per-source limits in `edhcut.config`).

* `scryfall` — every commander-legal card from the daily bulk file, plus name resolution.
* `tagger_bulk` — crowdsourced mechanic tags (Scryfall Tagger), the closest thing to
  ground truth about what a card *does*.
* `archidekt` — the decklists themselves, the largest and slowest harvest.
* `edhrec_commanders` / `edhrec` — the commander metagame ranking, and per-commander card
  inclusion rates and themes.
* `precons` — the official preconstructed products, and `precon_similarity` /
  `precon_retention` — how much of a harvested deck is an unmodified precon, and which precon
  cards real deckbuilders actually keep. Both feed the co-occurrence down-weighting.
* `qa_report` — the post-harvest coverage/sanity report (`data/qa_report.md`).

Run any of them with `python -m edhcut.ingest.<name>`; see the repo README §4 for the order.
"""
