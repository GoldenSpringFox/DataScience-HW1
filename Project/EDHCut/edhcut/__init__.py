"""EDHCut — a data-driven analysis of Magic: The Gathering Commander decklists.

Two layers, run in order:

* `edhcut.ingest` — one module per data source. Fills `data/edhcut.db` (SQLite, schema in
  `edhcut.db`) with cards, decklists, precon products, mechanic tags and metagame stats.
* `edhcut.analysis` — everything derived from that database: pairwise association metrics,
  card embeddings, soft/hard community detection, functional roles, play rates, deck weights.
  Outputs land in `data/kb/dev/` as parquet/npz artifacts the notebooks and report figures read.

Supporting modules: `config` (paths, commander roster, per-source rate limits), `db` (schema +
connection), `http` (cached, rate-limited session), `images` (card art, for the notebooks).

Almost every module is also runnable as a CLI — `python -m edhcut.<module> <subcommand>`; see
each module's `main()`. Start at the repo README for what to read first.
"""
