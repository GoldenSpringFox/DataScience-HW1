"""Co-occurrence counts + smoothed PMI/lift association scores (plan `EDHCut_PLAN.md` §2.4,
§7, task 6.1). First module of phase 2 (`edhcut/analysis/`): turns the raw `deck_cards` table
harvested in phase 1 into the sparse per-card-pair association matrices later phase-2/3 steps
(embeddings, communities, the cut recommender) build on.

**Card pool**: every card appearing in at least `MIN_DECK_COUNT` distinct decks, indexed by
`oracle_id` ascending (deterministic, reproducible across runs — not by deck count, which
would tie-break arbitrarily). This single index is shared by the global matrix and every
per-slot matrix, so a row number means the same card everywhere (plan §2.4's
`card_index.parquet` contract) — a thin slot's matrix is just mostly-empty rows, not a
differently-numbered universe.

**Precon-contamination down-weighting, per-card** (plan §7's "for slots with precon
contamination (Kyler), weight each deck by novelty" ask; revised from an earlier whole-*deck*
scheme — see below): a whole-deck novelty weight scaled *every* card in a near-precon deck
identically, which penalized a card the deckbuilder consciously chose to keep exactly as much
as a card only left in by inertia — no way to tell those apart at the deck level. Fixed by
moving the down-weight to individual cards instead. For a deck that matches a precon (via
`edhcut.ingest.precon_retention.best_matching_precon` — the same quantity-aware matching that
built `precon_card_retention`, reused here rather than re-derived, so "which precon a deck
matched" can't disagree between that table and this one):

- A card the player added themselves — not part of that precon's own printed list — is never
  down-weighted (weight 1.0 always). It can't be inertia; it wasn't there to begin with.
- A card that *did* ship with the precon is weighted by `precon_card_weight()`, a function of
  how often real deckbuilders keep it (`precon_card_retention.weighted_cut` /
  `(weighted_cut + weighted_kept)`, aggregated across every deck that matched that precon under
  that commander — see that table's own module docstring, `edhcut.ingest.precon_retention`, for
  how *that* aggregation itself now also accounts for deck-precon trust): full weight (1.0) at a
  cut rate of `PRECON_CUT_RATE_FULL_WEIGHT` (0.05) or below — i.e. kept 95%+ of the time, which
  reads as a considered inclusion, not inertia — linear decay to 0 weight at a 100% cut rate (a
  card that's *always* cut but still happens to be present in this one deck is the purest
  inertia signal available).
- That per-card weight is itself scaled by `deck_precon_trust()` (`edhcut.ingest.precon_retention`
  — imported from there, not defined here, since that module needs it too for its own
  `weighted_cut`/`weighted_kept` and `cooccurrence.py` already imports `best_matching_precon`
  from it; the reverse would be circular) — *this specific deck's* own overlap with the precon's
  card list, separate from the card's own aggregate cut rate. A card with a high aggregate cut
  rate sitting in a deck that only shares a handful of cards with the precon isn't good evidence
  of inertia — a deck that different was plausibly built from the ground up and just happens to
  include that card on its own merits, not because it was inherited. See
  `edhcut.ingest.precon_retention.deck_precon_trust`'s own docstring for the exact
  `diff`/`deck_total`-based curve (0 trust at 20 or fewer quantity-aware shared cards, 1.0 trust
  at 10 or fewer cards different from the precon).
- A pair's joint weight is the *product* of its two cards' individual weights — if either card
  is likely inertia, the pair's evidence is discounted accordingly; this also means a deck
  where every card carries weight 1.0 (no precon match, low precon-overlap trust, or every
  precon card cleared the 95% keep-rate bar) behaves exactly as if unweighted, the same as the
  old scheme's inert case.
- `total_weight` (`N`) is now just the deck count — under the old scheme a heavily-downweighted
  deck shrank the effective sample size itself; now only its own precon-inertia cards do, via
  their `weighted_marginal` contribution, so `N` stays a stable "how many decks" figure.

This creates a real, direct dependency on `precon_card_retention` being already built and
reasonably fresh — a deck matching a precon with no retention row yet for a given card (stale
data, not yet backfilled) falls back to full weight for that card rather than guessing.

**Smoothing & noise control** (plan §7's "small-corpus PMI noise" risk, sharpest for the
29-deck Orysa slot): PMI uses add-`SMOOTHING_K` Laplace-style smoothing on both the joint and
marginal *weighted* counts before taking the ratio; a pair actually co-occurring in fewer than
`MIN_PAIR_COUNT` distinct decks (raw, unweighted — "seen <3 times" per the plan, literally)
is masked to zero in both PMI and lift regardless of the smoothed value, so 1-2 coincidental
shared decks never manufacture an association. Lift is the same ratio without the add-k term
(a plainer "how many times more often than chance" reading, meant for eyeballing rather than
downstream math) but masked by the same raw-count floor.

**t-score** (added later, ad hoc — not part of the original plan): `compute_tscore` answers a
different question than PMI/lift — not "how much more than chance" (a *relative* effect size,
which a rare pair can win by pure coincidence, exactly the failure mode the PMI discount above
exists to fix) but "how confident am I this isn't noise" (a frequency-weighted question, where
a pair needs *both* real co-occurrence volume and a real excess over chance to score highly).
Same `min_pair_count` floor as PMI/lift, but no smoothing constant and no discount factor —
the `sqrt(joint)` denominator provides equivalent low-count damping on its own. Checked live
before adding: the whole global matrix's top-15 pairs by t-score are all real, broad
multi-card archetype packages (e.g. the goblin-tribal cluster `Goblin Warchief`/
`Skirk Prospector`/`Goblin Matron`/`Impact Tremors`, ~250-270 shared decks each, not a
single fragile 2-card combo), while pure-PMI ranking (ignoring frequency) surfaces mostly
6-10-deck coincidences with 100x+ lift — noise PMI's own discount factor doesn't fully catch
since it only *shrinks* low-count pairs rather than weighing them against genuinely frequent
ones on the same scale.

**Color-conditioned t-score** (added 2026-08-12, alongside plain `compute_tscore` — not a
replacement): `compute_tscore`'s pooled null model treats every deck as equally able to run
every card, which Commander's own color-identity legality rule makes false — two unrelated
green staples co-occur *above* the pooled baseline purely because green decks are a shared
eligible pool, not because of synergy. `compute_color_conditioned_tscore` conditions the null
model on eligibility instead (mirroring `playrate.py`'s own `eligible_deck_count`), and has a
useful self-check built in: whenever either card is colorless, the correction reduces
algebraically to exactly the pooled value (confirmed live to 0.0% change), so it only ever fires
where there's a real color-restriction gap, never as a blanket dampener. Validated live against
14 real card pairs before shipping (genuine same-color/cross-color synergy pairs moved 0% to
-8%; unrelated same-color "generic staple" pairs collapsed 48-92%) — see that function's own
docstring for the exact numbers and the formula.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from edhcut.analysis.card_categories import same_category_mask
from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.ingest.archidekt import _resolve_local_oracle_id, slot_key_for
from edhcut.ingest.precon_retention import (
    PRECON_DIFF_FULL_TRUST_CEILING,
    PRECON_OVERLAP_FLOOR,
    best_matching_precon,
    deck_precon_trust,
)
from edhcut.analysis.playrate import build_color_identity_deck_counts, canonical_identity, eligible_deck_count
from edhcut.ingest.scryfall import normalize_name

MIN_DECK_COUNT = 3
PRECON_CUT_RATE_FULL_WEIGHT = 0.05
SMOOTHING_K = 1.0
MIN_PAIR_COUNT = 3

# Matrix-scale guardrail (plan §7 task 5.7, added once the metagame harvest was expected to grow
# the card pool from ~4.8k to ~15-22k cards): every matrix this module itself produces stays
# sparse throughout -- compute_pmi/compute_lift/compute_tscore iterate CooccurrenceResult's
# pair_weighted dict (O(nnz)), never a dense n x n array, and `top_associated`'s only `.todense()`
# call is on a single sparse *row* (O(n), not O(n^2)). `build_and_save` itself does not call
# `assert_dense_matrix_safe` -- confirmed live post-harvest (18,979 cards, 5.9% pair density) that
# the real PMI/t-score matrices sit at ~63 MiB in memory, ~40x under the naive n^2 estimate, so
# there is no dense allocation here to guard against. This ceiling exists purely as a tripwire for
# something *else* (a future addition, a notebook, an ad hoc script) accidentally materializing a
# dense full-pool matrix and silently thrashing swap instead of failing with a clear error -- see
# `scripts/plot_matrix_overview.py`'s heatmap for the one real call site that needs it.
MAX_DENSE_MATRIX_BYTES = 2 * 1024**3

KB_DEV_DIR = CONFIG.paths.kb_dir / "dev"

Pair = tuple[int, int]


def assert_dense_matrix_safe(n: int, *, dtype_bytes: int = 8, max_bytes: int = MAX_DENSE_MATRIX_BYTES) -> None:
    """Raise `MemoryError` if a dense `n x n` array of `dtype_bytes`-byte elements would exceed
    `max_bytes`. Call this before any code path that might materialize one (`.toarray()`/
    `.todense()` on a matrix scoped to the *whole* card pool, not a small selected submatrix) --
    see the `MAX_DENSE_MATRIX_BYTES` comment above for why this exists despite nothing in this
    module needing a dense matrix today."""
    needed = n * n * dtype_bytes
    if needed > max_bytes:
        raise MemoryError(
            f"Refusing to materialize a dense {n:,}x{n:,} matrix "
            f"({needed / 1024**3:.2f} GiB at {dtype_bytes} bytes/element, over the "
            f"{max_bytes / 1024**3:.2f} GiB limit). This pipeline is sparse-only by design -- "
            "if you're seeing this, something is about to call .toarray()/.todense() on a "
            "full-pool matrix. Either avoid the dense conversion, restrict it to a smaller "
            "submatrix, or raise MIN_DECK_COUNT to shrink the card pool."
        )


def precon_card_weight(cut_rate: float) -> float:
    """Novelty weight for one *card* that shipped with a matched precon, as a function of how
    often real deckbuilders cut it. See module docstring for the scheme and its rationale.
    Full weight (1.0) at or below `PRECON_CUT_RATE_FULL_WEIGHT`, linear decay to 0 at a 100%
    cut rate. Cards that didn't ship with the matched precon never call this — they're always
    weight 1.0, unconditionally."""
    if cut_rate <= PRECON_CUT_RATE_FULL_WEIGHT:
        return 1.0
    if cut_rate >= 1.0:
        return 0.0
    span = 1.0 - PRECON_CUT_RATE_FULL_WEIGHT
    return 1.0 - (cut_rate - PRECON_CUT_RATE_FULL_WEIGHT) / span


def build_card_index(conn: sqlite3.Connection, *, min_decks: int = MIN_DECK_COUNT) -> pd.DataFrame:
    """Cards in >= `min_decks` distinct decks: `oracle_id`, `name`, `deck_count` (raw, global —
    the pool-membership test, independent of the novelty weighting applied to actual counts),
    `is_land` (for `top_associated`'s land/nonland category filter, `card_categories.py`),
    `row` (0-based, `oracle_id`-sorted — the index every matrix in this module shares)."""
    rows = conn.execute(
        """
        SELECT dc.oracle_id, c.name, c.is_land, COUNT(DISTINCT dc.deck_id) AS deck_count
        FROM deck_cards dc
        JOIN cards c ON c.oracle_id = dc.oracle_id
        GROUP BY dc.oracle_id
        HAVING COUNT(DISTINCT dc.deck_id) >= ?
        ORDER BY dc.oracle_id
        """,
        (min_decks,),
    ).fetchall()
    df = pd.DataFrame(rows, columns=["oracle_id", "name", "is_land", "deck_count"])
    df["row"] = np.arange(len(df), dtype=np.int64)
    return df


def _symmetric_sparse(pairs: dict[Pair, float], n: int, dtype) -> sparse.csr_matrix:
    """A dict of `{(i, j): value}` for i < j into a symmetric, zero-diagonal sparse matrix."""
    if not pairs:
        return sparse.csr_matrix((n, n), dtype=dtype)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (i, j), value in pairs.items():
        rows += [i, j]
        cols += [j, i]
        data += [value, value]
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=dtype).tocsr()


@dataclass
class CooccurrenceResult:
    """Everything needed to build the weighted co-occurrence matrix and derive PMI/lift for one
    scope (global, or one commander slot). `pair_raw`/`pair_weighted` are kept as plain dicts
    (not sparse matrices) because PMI/lift computation needs both the raw count (for the
    min-pair-count mask) and the weighted count (for the probability estimate) per pair, keyed
    identically — reconstructing that from two independently-converted sparse matrices would
    just be extra work for the same information."""

    n_cards: int
    pair_raw: dict[Pair, int] = field(default_factory=dict)
    pair_weighted: dict[Pair, float] = field(default_factory=dict)
    raw_marginal: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    weighted_marginal: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    total_weight: float = 0.0
    deck_count: int = 0

    def weighted_matrix(self) -> sparse.csr_matrix:
        return _symmetric_sparse(self.pair_weighted, self.n_cards, dtype=np.float64)

    def raw_matrix(self) -> sparse.csr_matrix:
        return _symmetric_sparse(self.pair_raw, self.n_cards, dtype=np.int64)

    @property
    def nnz_pairs(self) -> int:
        return len(self.pair_weighted)


def _precon_card_weight_overrides(
    conn: sqlite3.Connection,
    deck_qty: dict[int, dict[str, int]],
    deck_commanders: dict[int, list[str]],
) -> dict[int, dict[str, float]]:
    """For every deck that matches a precon (via `best_matching_precon`, same matching
    `precon_card_retention` itself was built with), weight each of *that precon's own* cards by
    `precon_card_weight(cut_rate)`, scaled by `deck_precon_trust()` for how much of the precon
    this specific deck actually still shares. Cards the player added themselves aren't part of
    the returned per-deck dict at all, so callers should default missing entries to 1.0 -- see
    module docstring."""
    precon_cards_cache: dict[int, set[str]] = {}
    retention_cache: dict[tuple[int, str], dict[str, float]] = {}  # (precon_id, commander_key) -> {oracle_id: cut_rate}
    overrides: dict[int, dict[str, float]] = {}

    for deck_id, commander_ids in deck_commanders.items():
        if not commander_ids:
            continue
        match = best_matching_precon(conn, deck_qty[deck_id], commander_ids)
        if match is None:
            continue
        precon_id, diff = match
        commander_key = slot_key_for(commander_ids)

        if precon_id not in precon_cards_cache:
            own_commanders = set(commander_ids)
            precon_cards_cache[precon_id] = {
                oracle_id
                for (oracle_id,) in conn.execute(
                    "SELECT oracle_id FROM precon_cards WHERE precon_id = ?", (precon_id,)
                )
                if oracle_id not in own_commanders
            }
        tracked = precon_cards_cache[precon_id]

        cache_key = (precon_id, commander_key)
        if cache_key not in retention_cache:
            cut_rates: dict[str, float] = {}
            for oracle_id, weighted_cut, weighted_kept in conn.execute(
                "SELECT oracle_id, weighted_cut, weighted_kept FROM precon_card_retention "
                "WHERE precon_id = ? AND commander_key = ?",
                cache_key,
            ):
                total = weighted_cut + weighted_kept
                if total > 0:
                    cut_rates[oracle_id] = weighted_cut / total
            retention_cache[cache_key] = cut_rates
        cut_rates = retention_cache[cache_key]

        deck_total = sum(deck_qty[deck_id].values())
        trust = deck_precon_trust(diff, deck_total)

        deck_overrides = {
            oracle_id: 1.0 - trust * (1.0 - precon_card_weight(cut_rates[oracle_id]))
            for oracle_id in deck_qty[deck_id]
            if oracle_id in tracked and oracle_id in cut_rates
        }
        if deck_overrides:
            overrides[deck_id] = deck_overrides

    return overrides


def build_cooccurrence(
    conn: sqlite3.Connection,
    card_index: pd.DataFrame,
    *,
    slot_key: str | None = None,
    deck_slot_weights: dict[str, float] | None = None,
) -> CooccurrenceResult:
    """Weighted (novelty-adjusted) + raw co-occurrence counts over `card_index`'s row universe,
    scoped to one commander slot's decks (`slot_key`) or every deck (`slot_key=None`, global).

    `deck_slot_weights` (`slot_key -> weight`, e.g. `edhcut.analysis.deck_weights.
    compute_deck_weights`) applies an additional *deck*-level multiplier on top of the existing
    *card*-level precon-novelty weight -- every card's contribution to a deck's marginal/pair
    counts is scaled by that deck's own slot weight, and `total_weight` (the null model's `N`)
    becomes the sum of those weights rather than a plain deck count, so the two stay consistent.
    A slot missing from the mapping defaults to 1.0 (unresolved -> neutral, same convention
    `deck_weight()` itself uses). `None` (the default) disables this entirely -- unweighted,
    unchanged from before this parameter existed."""
    oracle_to_row = dict(zip(card_index["oracle_id"], card_index["row"]))
    row_to_oracle = {row: oracle_id for oracle_id, row in oracle_to_row.items()}
    n = len(card_index)

    query = (
        "SELECT dc.deck_id, dc.oracle_id, dc.qty, d.commander_oracle_id, d.partner_oracle_id, d.slot_key "
        "FROM deck_cards dc JOIN decks d ON d.deck_id = dc.deck_id"
    )
    params: tuple = ()
    if slot_key is not None:
        query += " WHERE d.slot_key = ?"
        params = (slot_key,)

    by_deck: dict[int, set[int]] = {}
    deck_qty: dict[int, dict[str, int]] = {}
    deck_commanders: dict[int, list[str]] = {}
    deck_slot_key: dict[int, str] = {}
    for deck_id, oracle_id, qty, commander_oracle_id, partner_oracle_id, deck_slot in conn.execute(query, params):
        deck_qty.setdefault(deck_id, {})[oracle_id] = qty
        if deck_id not in deck_commanders:
            deck_commanders[deck_id] = [oid for oid in (commander_oracle_id, partner_oracle_id) if oid]
        deck_slot_key[deck_id] = deck_slot
        row = oracle_to_row.get(oracle_id)
        if row is None:
            continue  # below the pool's min-deck-count threshold, not part of this index
        by_deck.setdefault(deck_id, set()).add(row)

    weight_overrides = _precon_card_weight_overrides(conn, deck_qty, deck_commanders)

    pair_raw: Counter = Counter()
    pair_weighted: Counter = Counter()
    raw_marginal = np.zeros(n, dtype=np.int64)
    weighted_marginal = np.zeros(n, dtype=np.float64)
    total_weight = 0.0

    for deck_id, rows in by_deck.items():
        deck_w = deck_slot_weights.get(deck_slot_key[deck_id], 1.0) if deck_slot_weights is not None else 1.0
        total_weight += deck_w
        overrides = weight_overrides.get(deck_id, {})
        card_weight = {r: overrides.get(row_to_oracle[r], 1.0) for r in rows}
        for r in rows:
            raw_marginal[r] += 1
            weighted_marginal[r] += card_weight[r] * deck_w
        for i, j in combinations(sorted(rows), 2):
            pair_raw[(i, j)] += 1
            pair_weighted[(i, j)] += card_weight[i] * card_weight[j] * deck_w

    return CooccurrenceResult(
        n_cards=n,
        pair_raw=dict(pair_raw),
        pair_weighted=dict(pair_weighted),
        raw_marginal=raw_marginal,
        weighted_marginal=weighted_marginal,
        total_weight=total_weight,
        deck_count=len(by_deck),
    )


def compute_pmi(
    result: CooccurrenceResult,
    *,
    k: float = SMOOTHING_K,
    min_pair_count: int = MIN_PAIR_COUNT,
    discount: bool = True,
) -> sparse.csr_matrix:
    """Smoothed PMI (see module docstring for the formula's rationale):
    `log((joint + k) * N / ((marginal_i + k) * (marginal_j + k)))`, `N` = total deck weight in
    this scope. Masked to absent (not just zero — the pair is simply not a key) below
    `min_pair_count` *raw* co-occurrences.

    **Low-count discounting**: the `min_pair_count` floor alone turned out not to be enough —
    checked live against the real corpus (`docs/devlog/6.1-cooccurrence.md`) and found e.g.
    Purphoros's raw top-10 by PMI was ten unrelated cards (Prismatic Vista, Price of Glory,
    Seize the Day, ...) all tied at *exactly* the same value. Cause: whenever a rare card j
    (small `marginal_j`) happens to co-occur with i in *every single* deck it appears in
    (`joint == marginal_j` — easy by chance at marginal_j of 3-30 in a >1000-deck corpus), the
    smoothed-PMI formula reduces to `log(N / (marginal_i + k))`, a constant independent of
    `marginal_j` — so every such coincidentally-100%-co-occurring rare card ties for the
    ceiling score, drowning out cards with genuinely large, non-coincidental joint counts. This
    is exactly the "small-corpus PMI noise" plan §7 flags, just triggered by one side's small
    count rather than the whole corpus being small. Fixed with the standard Pantel/Lin
    discounting factor: `(joint / (joint + 1)) * (min(marginal_i, marginal_j) / (min(...) + 1))`
    — both terms -> 1 as counts grow, so it barely touches well-supported pairs, but shrinks a
    low-count pair's PMI toward 0 regardless of how extreme the raw ratio is, so genuine
    high-joint-count associations rank above one-off coincidences. Applied to PMI (the score
    later phase-2 steps consume) but not to `compute_lift` below, which is deliberately left as
    the plain, undiscounted ratio for eyeballing."""
    marg = result.weighted_marginal
    n_total = result.total_weight
    pmi_pairs: dict[Pair, float] = {}
    for pair, joint in result.pair_weighted.items():
        raw_joint = result.pair_raw[pair]
        if raw_joint < min_pair_count:
            continue
        i, j = pair
        value = (joint + k) * n_total / ((marg[i] + k) * (marg[j] + k))
        if value <= 0:
            continue
        pmi_value = float(np.log(value))
        if discount:
            min_marg = min(marg[i], marg[j])
            discount_factor = (raw_joint / (raw_joint + 1)) * (min_marg / (min_marg + 1))
            pmi_value *= discount_factor
        pmi_pairs[pair] = pmi_value
    return _symmetric_sparse(pmi_pairs, result.n_cards, dtype=np.float64)


def compute_lift(
    result: CooccurrenceResult, *, min_pair_count: int = MIN_PAIR_COUNT
) -> sparse.csr_matrix:
    """Plain (unsmoothed) lift: `joint * N / (marginal_i * marginal_j)`. Same raw-count mask as
    PMI. An eyeballing metric ("how many times more often than chance"), not itself consumed
    downstream — PMI is the association score later phase-2 steps use."""
    marg = result.weighted_marginal
    n_total = result.total_weight
    lift_pairs: dict[Pair, float] = {}
    for pair, joint in result.pair_weighted.items():
        if result.pair_raw[pair] < min_pair_count:
            continue
        i, j = pair
        denom = marg[i] * marg[j]
        if denom > 0:
            lift_pairs[pair] = joint * n_total / denom
    return _symmetric_sparse(lift_pairs, result.n_cards, dtype=np.float64)


def compute_tscore(
    result: CooccurrenceResult, *, min_pair_count: int = MIN_PAIR_COUNT
) -> sparse.csr_matrix:
    """t-score (see module docstring for why it was added alongside PMI/lift):
    `(joint - expected) / sqrt(joint)`, `expected = marginal_i * marginal_j / N`. Same raw-count
    mask as PMI/lift. No smoothing constant, no discount factor -- `sqrt(joint)` alone keeps a
    pair resting on a handful of coincidental shared decks from outscoring a pair with real
    volume behind it, which is what PMI's separate discount factor was built to approximate."""
    marg = result.weighted_marginal
    n_total = result.total_weight
    tscore_pairs: dict[Pair, float] = {}
    for pair, joint in result.pair_weighted.items():
        if result.pair_raw[pair] < min_pair_count or joint <= 0:
            continue
        i, j = pair
        expected = marg[i] * marg[j] / n_total
        tscore_pairs[pair] = (joint - expected) / math.sqrt(joint)
    return _symmetric_sparse(tscore_pairs, result.n_cards, dtype=np.float64)


def build_card_color_identities(conn: sqlite3.Connection, card_index: pd.DataFrame) -> list[str]:
    """`card_index`'s own row-ordered list of canonical WUBRG color identity strings (one per
    card) -- the input `compute_color_conditioned_tscore` needs to look up each card's own
    eligible-deck-count."""
    oracle_ids = tuple(card_index["oracle_id"])
    placeholders = ",".join("?" * len(oracle_ids))
    ci_by_oracle_id = dict(
        conn.execute(f"SELECT oracle_id, color_identity FROM cards WHERE oracle_id IN ({placeholders})", oracle_ids)
    )
    return [
        canonical_identity(json.loads(ci_by_oracle_id.get(oracle_id) or "[]"))
        for oracle_id in card_index["oracle_id"]
    ]


def compute_color_conditioned_tscore(
    result: CooccurrenceResult,
    card_identities: list[str],
    color_identity_deck_counts: pd.DataFrame,
    *,
    min_pair_count: int = MIN_PAIR_COUNT,
) -> sparse.csr_matrix:
    """t-score with the null model conditioned on Commander's color-identity legality rule, not
    just pooled across every deck (see module docstring's pooled `compute_tscore` for the
    baseline this corrects).

    **The problem this fixes**: `compute_tscore`'s `expected = marginal_i * marginal_j / N`
    treats every deck as equally able to run every card. False under Commander -- a green card
    can only appear in green-legal decks. Two unrelated green staples then co-occur *above* the
    pooled baseline purely because green decks are a shared eligible pool, not because of
    synergy -- confirmed live (2026-08-12) against real card pairs before this was written:
    `Swords to Plowshares + Felidar Retreat` (both popular, unrelated white staples) scored a
    pooled t-score of 8.18 -- looked like a real association -- but dropped 92% to 0.63 (noise)
    once conditioned; `Wood Elves + Managorger Hydra` (unrelated green ramp/aggro) fell from 1.87
    to 0.91. Genuine synergy pairs barely moved (`Goblin Warchief + Skirk Prospector`: 35.16 ->
    29.41, -16%; several Human/Zombie/Dragon typal and combo pairs across colors: 0% to -8%).

    **The formula**, mirroring `playrate.py`'s own `eligible_deck_count` logic:
    ```
    N_i   = decks eligible to run card i (color-identity superset of card i's own identity)
    N_ij  = decks eligible to run BOTH i and j (superset of the UNION of their identities)
    p_i   = marginal_i / N_i   -- this *is* playrate.py's own play_rate metric
    expected' = N_ij * p_i * p_j
    t_color = (joint - expected') / sqrt(joint)
    ```
    **Self-validating property, not just an empirical finding**: whenever one card is colorless
    (no identity restriction, `N_i` = every deck), the formula reduces algebraically to exactly
    `compute_tscore`'s own pooled expected value -- confirmed live to 0.0% change on every
    colorless-involving pair tested (`Sol Ring + Arcane Signet`, `Ashnod's Altar + Zulaport
    Cutthroat`, `Solemn Simulacrum + Eternal Witness`). The correction only ever fires when it
    has a real color-restriction gap to correct, never as a blanket dampener.

    `color_identity_deck_counts` should come from `playrate.build_color_identity_deck_counts`,
    scoped with the *same* `slot_key` (or none, for global) that `result` itself was built with
    -- a per-slot analysis's eligible counts must be scoped to that slot's own decks, not every
    harvested deck, or `N_ij` would count decks outside the analysis's own population."""
    marg = result.weighted_marginal
    n_cards = result.n_cards

    eligible_cache: dict[str, int] = {}

    def eligible(ci: str) -> int:
        if ci not in eligible_cache:
            eligible_cache[ci] = eligible_deck_count(color_identity_deck_counts, ci)
        return eligible_cache[ci]

    card_eligible = np.array([eligible(ci) for ci in card_identities], dtype=np.float64)
    card_play_rate = np.divide(
        marg, card_eligible, out=np.zeros_like(marg), where=card_eligible > 0
    )

    pair_n_ij_cache: dict[frozenset[str], int] = {}
    tscore_pairs: dict[Pair, float] = {}
    for pair, joint in result.pair_weighted.items():
        if result.pair_raw[pair] < min_pair_count or joint <= 0:
            continue
        i, j = pair
        ci_i, ci_j = card_identities[i], card_identities[j]
        cache_key = frozenset((ci_i, ci_j))
        if cache_key not in pair_n_ij_cache:
            pair_n_ij_cache[cache_key] = eligible(canonical_identity(set(ci_i) | set(ci_j)))
        n_ij = pair_n_ij_cache[cache_key]
        expected = n_ij * card_play_rate[i] * card_play_rate[j]
        tscore_pairs[pair] = (joint - expected) / math.sqrt(joint)
    return _symmetric_sparse(tscore_pairs, n_cards, dtype=np.float64)


def _slot_label(commander_names: Iterable[str]) -> str:
    """Filesystem-friendly slug for a slot, e.g. ["Krenko, Mob Boss"] -> "krenko",
    ["Yoshimaru, Ever Faithful", "Bruse Tarl, Boorish Herder"] -> "yoshimaru_bruse_tarl".
    Human-readable filenames beat raw oracle_id-based `slot_key`s for anyone browsing
    `data/kb/dev/` by hand."""
    parts = []
    for name in commander_names:
        before_comma = name.split(",")[0].strip().lower()
        parts.append(re.sub(r"[^a-z0-9]+", "_", before_comma).strip("_"))
    return "_".join(parts)


@dataclass
class SlotSpec:
    """One commander slot resolved for matrix-building: its filesystem slug (`label`, e.g.
    "yoshimaru_bruse_tarl"), the configured card names, and the `slot_key` (the `oracle_id`-based
    key the `decks` table stores) that selects its decks."""

    label: str
    commander_names: list[str]
    slot_key: str


def configured_slots(conn: sqlite3.Connection) -> list[SlotSpec]:
    """Resolve `CONFIG.commander_slots` (card names) to their `slot_key`s (oracle_ids) against
    this database, in plan §1 roster order."""
    specs = []
    for names in CONFIG.commander_slots:
        oracle_ids = [_resolve_local_oracle_id(conn, name) for name in names]
        specs.append(SlotSpec(label=_slot_label(names), commander_names=names, slot_key=slot_key_for(oracle_ids)))
    return specs


@dataclass
class ScopeStats:
    """Build summary for one scope (the global matrix, or one commander slot): pool size, how many
    decks fed it, the summed novelty weight, how many card pairs ended up nonzero, and how many
    survived into PMI after the `MIN_PAIR_COUNT` mask. `pmi_pairs` well below `nnz_pairs` is the
    expected shape — most pairs co-occur once or twice and are masked as coincidence."""

    label: str
    n_cards: int
    deck_count: int
    total_weight: float
    nnz_pairs: int
    pmi_pairs: int

    @property
    def density(self) -> float:
        """Fraction of the upper-triangle card-pair space with a nonzero co-occurrence count."""
        possible_pairs = self.n_cards * (self.n_cards - 1) / 2
        return self.nnz_pairs / possible_pairs if possible_pairs else 0.0


def build_and_save(
    conn: sqlite3.Connection, *, out_dir: Path = KB_DEV_DIR, min_decks: int = MIN_DECK_COUNT
) -> list[ScopeStats]:
    """Build the global + every configured slot's co-occurrence/PMI/lift matrices and write them
    under `out_dir`. Returns per-scope stats for the sanity CLI / devlog."""
    out_dir.mkdir(parents=True, exist_ok=True)

    card_index = build_card_index(conn, min_decks=min_decks)
    card_index.to_parquet(out_dir / "card_index.parquet", index=False)
    card_identities = build_card_color_identities(conn, card_index)

    stats: list[ScopeStats] = []

    def _save(label: str, result: CooccurrenceResult, *, slot_key: str | None) -> ScopeStats:
        sparse.save_npz(out_dir / f"cooccur_{label}.npz", result.weighted_matrix())
        pmi = compute_pmi(result)
        lift = compute_lift(result)
        tscore = compute_tscore(result)
        sparse.save_npz(out_dir / f"pmi_{label}.npz", pmi)
        sparse.save_npz(out_dir / f"lift_{label}.npz", lift)
        sparse.save_npz(out_dir / f"tscore_{label}.npz", tscore)

        # Same deck population `result` was itself built over -- see
        # compute_color_conditioned_tscore's own docstring for why this must be scoped, not global.
        color_identity_deck_counts = build_color_identity_deck_counts(conn, slot_key=slot_key)
        tscore_color = compute_color_conditioned_tscore(result, card_identities, color_identity_deck_counts)
        sparse.save_npz(out_dir / f"tscore_color_{label}.npz", tscore_color)

        return ScopeStats(
            label=label,
            n_cards=result.n_cards,
            deck_count=result.deck_count,
            total_weight=result.total_weight,
            nnz_pairs=result.nnz_pairs,
            pmi_pairs=pmi.nnz // 2,
        )

    global_result = build_cooccurrence(conn, card_index, slot_key=None)
    stats.append(_save("global", global_result, slot_key=None))

    slot_labels = {}
    for spec in configured_slots(conn):
        slot_result = build_cooccurrence(conn, card_index, slot_key=spec.slot_key)
        stats.append(_save(spec.label, slot_result, slot_key=spec.slot_key))
        slot_labels[spec.label] = spec.slot_key

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "min_deck_count": min_decks,
        "precon_cut_rate_full_weight": PRECON_CUT_RATE_FULL_WEIGHT,
        "precon_overlap_floor": PRECON_OVERLAP_FLOOR,
        "precon_diff_full_trust_ceiling": PRECON_DIFF_FULL_TRUST_CEILING,
        "smoothing_k": SMOOTHING_K,
        "min_pair_count": MIN_PAIR_COUNT,
        "max_dense_matrix_bytes": MAX_DENSE_MATRIX_BYTES,
        "n_cards": len(card_index),
        "slot_labels": slot_labels,
        "scopes": [
            {
                "label": s.label,
                "n_cards": s.n_cards,
                "deck_count": s.deck_count,
                "total_weight": s.total_weight,
                "cooccur_nnz_pairs": s.nnz_pairs,
                "pmi_nnz_pairs": s.pmi_pairs,
                "density": s.density,
            }
            for s in stats
        ],
    }
    (out_dir / "cooccurrence_manifest.json").write_text(json.dumps(manifest, indent=2))

    return stats


def load_card_index(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    """The shared card->row index every matrix in this module is aligned to."""
    return pd.read_parquet(out_dir / "card_index.parquet")


def load_pmi(label: str, out_dir: Path = KB_DEV_DIR) -> sparse.csr_matrix:
    """Saved PMI matrix for one scope ("global" or a slot label, e.g. "krenko")."""
    return sparse.load_npz(out_dir / f"pmi_{label}.npz")


def load_tscore(label: str, out_dir: Path = KB_DEV_DIR) -> sparse.csr_matrix:
    """Saved (pooled-null) t-score matrix for one scope."""
    return sparse.load_npz(out_dir / f"tscore_{label}.npz")


def load_tscore_color(label: str, out_dir: Path = KB_DEV_DIR) -> sparse.csr_matrix:
    """Saved colour-conditioned t-score matrix for one scope -- the metric the report settles on."""
    return sparse.load_npz(out_dir / f"tscore_color_{label}.npz")


def top_associated(
    pmi: sparse.csr_matrix,
    card_index: pd.DataFrame,
    oracle_id: str,
    *,
    k: int = 20,
    include_cross_category: bool = False,
) -> pd.DataFrame:
    """Top-`k` cards by PMI against `oracle_id` within this matrix's scope. Excludes land/nonland
    cross-category matches by default (`card_categories.same_category_mask` -- a land's near-fixed
    per-deck slot count means its PMI with any one nonland card mostly reflects mana-base
    requirements, not a real association with that specific card); pass
    `include_cross_category=True` to opt back in."""
    match = card_index.index[card_index["oracle_id"] == oracle_id]
    if len(match) == 0:
        raise KeyError(f"{oracle_id!r} is not in this card index (below the min-deck-count pool cutoff?)")
    row = int(card_index.loc[match[0], "row"])
    query_is_land = bool(card_index.loc[match[0], "is_land"])

    vec = np.asarray(pmi[row].todense()).ravel()
    order = np.argsort(-vec)
    rows_by_index = card_index.set_index("row").sort_index()
    is_land_by_row = rows_by_index["is_land"].to_numpy()
    allowed = same_category_mask(is_land_by_row, query_is_land, include_cross_category=include_cross_category)

    results = []
    for other_row in order:
        if other_row == row or vec[other_row] == 0:
            continue
        if not allowed[other_row]:
            continue
        other = rows_by_index.loc[int(other_row)]
        results.append({"oracle_id": other["oracle_id"], "name": other["name"], "pmi": float(vec[other_row])})
        if len(results) >= k:
            break
    return pd.DataFrame(results)


def _resolve_card_name(conn: sqlite3.Connection, name: str) -> str:
    """Resolve a card name to its `oracle_id` for the CLI, exiting with a clear message rather than a
    traceback if it is unknown."""
    row = conn.execute(
        "SELECT oracle_id FROM card_names WHERE name_normalized = ?", (normalize_name(name),)
    ).fetchone()
    if row is None:
        raise SystemExit(f"Card {name!r} not found in card_names.")
    return row[0]


def _cmd_build(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        stats = build_and_save(conn)
    print(f"Built co-occurrence/PMI/lift for {len(stats)} scopes under {KB_DEV_DIR}:")
    for s in stats:
        print(
            f"  {s.label:20s} decks={s.deck_count:5d} total_weight={s.total_weight:8.1f} "
            f"cooccur_pairs={s.nnz_pairs:9,d} pmi_pairs={s.pmi_pairs:9,d} density={s.density:.4%}"
        )


def _cmd_top(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        oracle_id = _resolve_card_name(conn, args.card)
    card_index = load_card_index()
    pmi = load_pmi(args.slot or "global")
    table = top_associated(pmi, card_index, oracle_id, k=args.k, include_cross_category=args.include_cross_category)
    if table.empty:
        print(f"No associations found for {args.card!r} in scope {args.slot or 'global'!r}.")
        return
    print(f"Top {len(table)} associated with {args.card!r} ({args.slot or 'global'}):")
    for _, row in table.iterrows():
        print(f"  {row['pmi']:+7.3f}  {row['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build & save global + per-slot matrices to data/kb/dev/")
    build_p.set_defaults(func=_cmd_build)

    top_p = sub.add_parser("top", help="Top-k PMI associations for a card")
    top_p.add_argument("card", help="Card name")
    top_p.add_argument(
        "--slot", default=None,
        help="Slot label to scope to (e.g. 'krenko', 'yoshimaru_bruse_tarl'); default: global",
    )
    top_p.add_argument("-k", type=int, default=20)
    top_p.add_argument(
        "--include-cross-category", action="store_true",
        help="Include land/nonland cross-category matches (excluded by default -- see "
        "edhcut.analysis.card_categories)",
    )
    top_p.set_defaults(func=_cmd_top)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
