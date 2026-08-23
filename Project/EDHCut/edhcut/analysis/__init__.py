"""Analysis layer: everything derived from `data/edhcut.db`. Artifacts are written to
`data/kb/dev/` (parquet/npz) and read back by the notebooks and the report figure scripts.

In dependency order, with the report question each one backs:

* `playrate` — how often a card is played, out of the decks that could *legally* play it
  (colour identity restricts the denominator). Used everywhere as the popularity axis.
* `deck_weights` — inverse-probability weights correcting the harvest's commander imbalance,
  so a heavily oversampled commander cannot dominate a global statistic.
* `cooccurrence` — **Q1.** The pairwise metrics (raw co-occurrence, smoothed PMI, lift, t-score,
  colour-conditioned t-score) plus the shared `card_index` row numbering that every matrix in
  this package reuses. The largest and most load-bearing module here.
* `card_categories` — the land/nonland filter shared by association and similarity lookups.
* `communities` — **Q2.** Graph construction (`build_graph`, `sparsify_top_k_union`,
  `basic_land_mask`) and a Leiden hard partition over it. Both community notebooks import the
  graph helpers from here; the Louvain partition the report compares against is run in
  `notebooks/louvain_communities.ipynb` itself, on this same graph.
* `symnmf_packages` — **Q2, the main result.** Soft, overlapping packages via symmetric NMF over
  the colour-conditioned t-score matrix, so one card can belong to several packages at once.
  `notebooks/symNMF_communities.ipynb` calls `fit_symnmf` directly.
* `embeddings` — **Q3.** Card vectors from oracle text (TF-IDF + truncated SVD, plus structured
  cost/type features) and from decks (Word2Vec over decklists-as-sentences).
* `roles` — **Q3.** One of 18 functional roles per card (primary + optional secondary), from
  mechanic tags with an oracle-text heuristic layer as backup.

Superseded, kept because `notebooks/archive/communities_exploration.ipynb` documents the route
taken to the final method:
`nmf_packages` (plain NMF over the deck × card matrix — replaced by `symnmf_packages`, which can
use the colour correction), `symnmf_hierarchy` (recursive refinement of the packages into a tree)
and `theme_labels` (EDHREC/Tagger labels used to score clustering granularity and colour bias).
"""
