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

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.ingest.archidekt import _resolve_local_oracle_id, slot_key_for
from edhcut.ingest.precon_retention import (
    PRECON_DIFF_FULL_TRUST_CEILING,
    PRECON_OVERLAP_FLOOR,
    best_matching_precon,
    deck_precon_trust,
)
from edhcut.ingest.scryfall import normalize_name

MIN_DECK_COUNT = 3
PRECON_CUT_RATE_FULL_WEIGHT = 0.05
SMOOTHING_K = 1.0
MIN_PAIR_COUNT = 3

KB_DEV_DIR = CONFIG.paths.kb_dir / "dev"

Pair = tuple[int, int]


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
    `row` (0-based, `oracle_id`-sorted — the index every matrix in this module shares)."""
    rows = conn.execute(
        """
        SELECT dc.oracle_id, c.name, COUNT(DISTINCT dc.deck_id) AS deck_count
        FROM deck_cards dc
        JOIN cards c ON c.oracle_id = dc.oracle_id
        GROUP BY dc.oracle_id
        HAVING COUNT(DISTINCT dc.deck_id) >= ?
        ORDER BY dc.oracle_id
        """,
        (min_decks,),
    ).fetchall()
    df = pd.DataFrame(rows, columns=["oracle_id", "name", "deck_count"])
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
    conn: sqlite3.Connection, card_index: pd.DataFrame, *, slot_key: str | None = None
) -> CooccurrenceResult:
    """Weighted (novelty-adjusted) + raw co-occurrence counts over `card_index`'s row universe,
    scoped to one commander slot's decks (`slot_key`) or every deck (`slot_key=None`, global)."""
    oracle_to_row = dict(zip(card_index["oracle_id"], card_index["row"]))
    row_to_oracle = {row: oracle_id for oracle_id, row in oracle_to_row.items()}
    n = len(card_index)

    query = (
        "SELECT dc.deck_id, dc.oracle_id, dc.qty, d.commander_oracle_id, d.partner_oracle_id "
        "FROM deck_cards dc JOIN decks d ON d.deck_id = dc.deck_id"
    )
    params: tuple = ()
    if slot_key is not None:
        query += " WHERE d.slot_key = ?"
        params = (slot_key,)

    by_deck: dict[int, set[int]] = {}
    deck_qty: dict[int, dict[str, int]] = {}
    deck_commanders: dict[int, list[str]] = {}
    for deck_id, oracle_id, qty, commander_oracle_id, partner_oracle_id in conn.execute(query, params):
        deck_qty.setdefault(deck_id, {})[oracle_id] = qty
        if deck_id not in deck_commanders:
            deck_commanders[deck_id] = [oid for oid in (commander_oracle_id, partner_oracle_id) if oid]
        row = oracle_to_row.get(oracle_id)
        if row is None:
            continue  # below the pool's min-deck-count threshold, not part of this index
        by_deck.setdefault(deck_id, set()).add(row)

    weight_overrides = _precon_card_weight_overrides(conn, deck_qty, deck_commanders)

    pair_raw: Counter = Counter()
    pair_weighted: Counter = Counter()
    raw_marginal = np.zeros(n, dtype=np.int64)
    weighted_marginal = np.zeros(n, dtype=np.float64)

    for deck_id, rows in by_deck.items():
        overrides = weight_overrides.get(deck_id, {})
        card_weight = {r: overrides.get(row_to_oracle[r], 1.0) for r in rows}
        for r in rows:
            raw_marginal[r] += 1
            weighted_marginal[r] += card_weight[r]
        for i, j in combinations(sorted(rows), 2):
            pair_raw[(i, j)] += 1
            pair_weighted[(i, j)] += card_weight[i] * card_weight[j]

    return CooccurrenceResult(
        n_cards=n,
        pair_raw=dict(pair_raw),
        pair_weighted=dict(pair_weighted),
        raw_marginal=raw_marginal,
        weighted_marginal=weighted_marginal,
        total_weight=float(len(by_deck)),
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

    stats: list[ScopeStats] = []

    def _save(label: str, result: CooccurrenceResult) -> ScopeStats:
        sparse.save_npz(out_dir / f"cooccur_{label}.npz", result.weighted_matrix())
        pmi = compute_pmi(result)
        lift = compute_lift(result)
        tscore = compute_tscore(result)
        sparse.save_npz(out_dir / f"pmi_{label}.npz", pmi)
        sparse.save_npz(out_dir / f"lift_{label}.npz", lift)
        sparse.save_npz(out_dir / f"tscore_{label}.npz", tscore)
        return ScopeStats(
            label=label,
            n_cards=result.n_cards,
            deck_count=result.deck_count,
            total_weight=result.total_weight,
            nnz_pairs=result.nnz_pairs,
            pmi_pairs=pmi.nnz // 2,
        )

    global_result = build_cooccurrence(conn, card_index, slot_key=None)
    stats.append(_save("global", global_result))

    slot_labels = {}
    for spec in configured_slots(conn):
        slot_result = build_cooccurrence(conn, card_index, slot_key=spec.slot_key)
        stats.append(_save(spec.label, slot_result))
        slot_labels[spec.label] = spec.slot_key

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "min_deck_count": min_decks,
        "precon_cut_rate_full_weight": PRECON_CUT_RATE_FULL_WEIGHT,
        "precon_overlap_floor": PRECON_OVERLAP_FLOOR,
        "precon_diff_full_trust_ceiling": PRECON_DIFF_FULL_TRUST_CEILING,
        "smoothing_k": SMOOTHING_K,
        "min_pair_count": MIN_PAIR_COUNT,
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
    return pd.read_parquet(out_dir / "card_index.parquet")


def load_pmi(label: str, out_dir: Path = KB_DEV_DIR) -> sparse.csr_matrix:
    return sparse.load_npz(out_dir / f"pmi_{label}.npz")


def load_tscore(label: str, out_dir: Path = KB_DEV_DIR) -> sparse.csr_matrix:
    return sparse.load_npz(out_dir / f"tscore_{label}.npz")


def top_associated(
    pmi: sparse.csr_matrix, card_index: pd.DataFrame, oracle_id: str, *, k: int = 20
) -> pd.DataFrame:
    """Top-`k` cards by PMI against `oracle_id` within this matrix's scope."""
    match = card_index.index[card_index["oracle_id"] == oracle_id]
    if len(match) == 0:
        raise KeyError(f"{oracle_id!r} is not in this card index (below the min-deck-count pool cutoff?)")
    row = int(card_index.loc[match[0], "row"])

    vec = np.asarray(pmi[row].todense()).ravel()
    order = np.argsort(-vec)
    rows_by_index = card_index.set_index("row")

    results = []
    for other_row in order:
        if other_row == row or vec[other_row] == 0:
            continue
        other = rows_by_index.loc[int(other_row)]
        results.append({"oracle_id": other["oracle_id"], "name": other["name"], "pmi": float(vec[other_row])})
        if len(results) >= k:
            break
    return pd.DataFrame(results)


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
    table = top_associated(pmi, card_index, oracle_id, k=args.k)
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
    top_p.set_defaults(func=_cmd_top)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
