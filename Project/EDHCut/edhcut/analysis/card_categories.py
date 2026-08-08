"""Shared land/nonland category filter for similarity- and association-based card lookups
(`cooccurrence.py`'s `top_associated`, `embeddings.py`'s `nearest_neighbors`).

A Commander deck's land count is nearly fixed (~35-38 slots) regardless of which specific
nonland cards it runs, so a land's statistical relationship to any one nonland card mostly
reflects "this color needs mana sources," not a genuine substitution relationship -- and this
leaks into every method we have no matter how it's computed: PMI/t-score (task 6.1) see a land's
enormous, largely card-independent marginal deck-frequency; TF-IDF (task 6.2) sees a land's
ability text (`{T}: Add {W}.`) that's often *literally identical* to a creature mana-dork's.
None of PMI, lift, t-score, Word2Vec, or TF-IDF has any built-in notion that land-vs-nonland is
a near-inviolable boundary for a *substitution* recommendation -- that's Magic domain knowledge,
not a pattern any of these statistics derive on their own. Fixed here once, applied identically
everywhere a "what's a good replacement for this card" query is answered, rather than
re-litigated (or silently left unfixed) per module.

Default: exclude cross-category results entirely. `include_cross_category=True` opts back in,
for the rare case a cross-category comparison is actually wanted -- that's a deliberate escape
hatch, not the common path.
"""

from __future__ import annotations

import numpy as np


def same_category_mask(is_land: np.ndarray, query_is_land: bool, *, include_cross_category: bool) -> np.ndarray:
    """Boolean mask selecting candidates in the same land/nonland category as the query card.
    `include_cross_category=True` returns an all-true mask (the filter becomes a no-op)."""
    if include_cross_category:
        return np.ones(len(is_land), dtype=bool)
    return is_land.astype(bool) == bool(query_is_land)
