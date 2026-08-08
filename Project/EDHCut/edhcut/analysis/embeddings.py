"""Card embeddings + similarity index (plan `EDHCut_PLAN.md` §2.4, §7, task 6.2).
Depends on task 6.1 only for the pattern it reuses (card pool / KB-dir conventions), not on
6.1's own matrices -- Word2Vec trains directly off `deck_cards`, not off any PMI/t-score
association scores.

Two complementary vector spaces, both written to one `embeddings.parquet` (`oracle_id`-indexed,
NaN where a card has no vector in that space):

- **`w2v_<i>`** (gensim Word2Vec, `VECTOR_SIZE` dims): each deck is a "sentence" of oracle_ids
  (commanders excluded -- `deck_cards` never has them, same pool `cooccurrence.py` uses). A
  deck's card order is arbitrary (however the DB query happened to return rows), not a real
  sequence, so `ShuffledDeckCorpus` reshuffles each deck's own card order every time gensim
  iterates the corpus (once per training epoch) rather than training on one fixed, meaningless
  order -- this still matters even with a `window` larger than a typical deck, because gensim's
  own per-instance *dynamic* window (a random `randint(0, window)` reduction on every training
  instance, part of the original word2vec algorithm) means no single pass already sees every
  pair regardless of order. Cards below `MIN_DECK_COUNT` occurrences are dropped from the
  vocabulary by gensim's own `min_count`, same threshold `cooccurrence.py` uses for its own pool
  -- a thin card just doesn't get a `w2v_*` row at all rather than a noisy one.

- **`tfidf_<i>` / `types_<i>` + structured cost/P-T columns**, jointly the "`tfidf`" space:
  covers every commander-legal card regardless of deck presence -- the plan's own stated reason
  is cards too rare (or entirely unharvested) for Word2Vec to place well, sharpest for the thin
  Orysa slot. Built over the *whole* `cards` table (`legal_commander = 1`), not scoped to the
  deck-derived pool. **Four independently-normalized, fixed-weight blocks** (redesigned twice
  after the plan's original single-TF-IDF-over-everything spec, per explicit user feedback --
  see the two rounds below):

  1. `tfidf_<i>` (`ORACLE_TEXT_DIMS` dims): TF-IDF+SVD over `oracle_text` *alone*, plain
     bag-of-words with `_ORACLE_TEXT_TOKEN_PATTERN` (keeps a mana/tap symbol like `{T}` or `{W}`
     as its own literal token -- sklearn's default requires 2+ word characters, silently
     dropping a single-letter symbol) and **no stopword filtering at all**. A round attempting
     to fake word-order sensitivity onto TF-IDF (splitting on Magic's `<cost>: <effect>` colon,
     tagging tokens, adding bigrams) was tried and explicitly reverted per user feedback: TF-IDF
     is bag-of-words and has no real notion of order -- forcing structure onto it via
     preprocessing tricks is fighting the method rather than accepting what it actually is.
     Stopword filtering was reverted too, for a related reason: generic English stopword lists
     strip exactly the words Magic's own templating leans on ("when"/"whenever"/"if"/"each"/
     "one"/"more"/... -- confirmed live that all but "unless" are in sklearn's own
     `ENGLISH_STOP_WORDS`), and a hand-curated exception list can never be complete (this was
     tried too, then abandoned as fundamentally incomplete rather than kept partial). Skipping
     stopword removal entirely relies on TF-IDF's own IDF term to do that job instead -- a word
     appearing in most cards' oracle_text already gets a near-zero weight automatically, no
     hand-picked list required, and nothing gets lost by mistake.
  2. `types_<i>` (`TYPES_DIMS` dims): a *separate* TF-IDF+SVD over `type_line` alone (own
     vectorizer/SVD, not combined with oracle_text's, and not through the custom analyzer above
     -- type_line has no mana symbols or ability structure to speak of). Round-1 concatenated
     oracle_text + type_line into one bag-of-words before TF-IDF -- checked live and found it
     backfired for short "vanilla" cards: Avacyn's Pilgrim's whole document reduces to a handful
     of tokens after stripping symbols, so whichever few survive dominate the vector's direction;
     the shared word "Human" (a type_line word) ended up mattering more than the actual mana
     ability, ranking Llanowar Elves (a near-identical `{T}: Add {G}.` dork, but "Elf" not
     "Human") 496th. Splitting text into its own weighted block, independent of type_line, fixes
     this at the source rather than by further reweighting.
  3. `struct_cmc_bucket_<i>` + `struct_pip_<wubrg>` ("mana cost" block): mana value as a
     Gaussian bump across integer CMC buckets 0..`CMC_BUCKET_MAX` (see `_cmc_bucket_vector`'s own
     docstring for why a single z-scored number can't express "penalize a bigger CMC gap more"
     under cosine similarity at all -- found live, per explicit user request, after two other
     cards' oracle-text similarity to a 2-drop query card wasn't being discounted at all for
     being a 5-drop) rather than a single z-scored number; per-color pip counts parsed from
     `mana_cost` (hybrid/Phyrexian symbols counting once per named color) -- operationalizes "a
     triple-green card should sit further from a multicolor card" as a real per-color intensity
     signal. The two sub-signals are independently L2-normalized and re-combined at a fixed
     `CMC_SUBWEIGHT`/`PIP_SUBWEIGHT` (0.5/0.5) *within* this block, same reasoning as the
     block-level weights below applied one level down (9 CMC-bucket dims vs. 5 pip dims
     shouldn't get to matter purely by dimension count either).
  4. `struct_power`/`struct_toughness` (+ `has_*`/`*_is_variable` flags, see `_parse_stat`)
     ("power/toughness" block).

  **Fixed per-block weights, an explicit user design choice rather than a derived/tuned
  value** (round 3 of the *block* redesign, after round 1's single `struct_weight` scalar on a
  lumped struct block was found to *still* let mana cost/P-T dominate purely by which block
  happened to have larger raw magnitude, and round 2's 0.6/0.15/0.15/0.1 split -- while it did
  fix the CMC-*distance* bug, that signal could only ever contribute 7.5% to the total score,
  nowhere near enough to keep a near-perfect oracle-text match at a wildly different CMC, e.g.
  Princess Lucrezia (CMC 6, `{T}: Add {U}.`) outranking closer-cost cards for Avacyn's Pilgrim
  (CMC 1, `{T}: Add {W}.`)): oracle text 0.5, types 0.1, mana cost 0.3, power/toughness 0.1.
  Each block is L2-normalized to unit length *per row* first (so no block's raw magnitude, nor a
  document's raw length/sparsity, biases the balance -- the actual bug round 1 was chasing),
  then scaled by its weight. **Power/toughness's 0.1 doesn't apply to non-creatures**
  (`struct_has_power == 0`): it has nothing to weight, so its share is redistributed evenly onto
  mana cost and types instead (0.35/0.15), keeping weights summing to 1 every row regardless of
  card type.

Sanity CLI mirrors the plan's own named check: `python -m edhcut.analysis.embeddings neighbors
"Cultivate" --space w2v` should surface "Kodama's Reach" (already confirmed via co-occurrence/
PMI in task 6.1, reproduced here from an entirely different method).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# gensim is imported lazily (inside train_word2vec/word2vec_frame) rather than at module level:
# its shipped C source doesn't build on Python 3.14 (CPython changed PyLongObject internals
# gensim's Cython code still assumes -- confirmed live, fails identically even with a working
# MSVC compiler), so it only has wheels through 3.13. scikit-learn has no such issue. Run
# anything touching Word2Vec under the repo's separate `venv311/` (Python 3.11) -- see
# pyproject.toml's `embeddings` extra and module docstring.

from edhcut.analysis.card_categories import same_category_mask
from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.ingest.scryfall import normalize_name

MIN_DECK_COUNT = 3  # same pool threshold as cooccurrence.py's card index

VECTOR_SIZE = 64
WINDOW = 100  # comfortably >= a full 99-card deck; "large window" per the plan
MIN_COUNT = MIN_DECK_COUNT
EPOCHS = 10
SHUFFLES_PER_EPOCH = 3
SEED = 42

ORACLE_TEXT_DIMS = 64
TYPES_DIMS = 16

# Fixed per-block weights for the "tfidf" space -- an explicit user design choice, not a tuned
# value (see module docstring). Must sum to 1.0; power/toughness's share is redistributed onto
# mana cost + types for non-creatures (see _build_space_matrix). Revised from an initial
# 0.6/0.15/0.15/0.1 split once live testing showed mana cost's 15% (7.5% for the CMC-distance
# sub-signal specifically) wasn't enough to keep a near-perfect oracle-text match (e.g. Princess
# Lucrezia, CMC 6, vs. Avacyn's Pilgrim, CMC 1 -- both literally "{T}: Add <color>.") from
# outranking much closer-cost cards.
WEIGHT_ORACLE_TEXT = 0.5
WEIGHT_TYPES = 0.1
WEIGHT_MANA_COST = 0.3
WEIGHT_POWER_TOUGHNESS = 0.1

# Mana value is encoded as a Gaussian bump across integer buckets 0..CMC_BUCKET_MAX (the last
# bucket catching everything at or above it) rather than a single z-scored number -- see
# `_cmc_bucket_vector`'s docstring for why a single scalar under cosine similarity can't express
# "penalize a bigger CMC gap more."
CMC_BUCKET_MAX = 8
CMC_BUCKET_SIGMA = 1.0
CMC_BUCKET_COLS = [f"struct_cmc_bucket_{i}" for i in range(CMC_BUCKET_MAX + 1)]
PIP_STRUCT_COLS = ["struct_pip_w", "struct_pip_u", "struct_pip_b", "struct_pip_r", "struct_pip_g"]
# Sub-weights *within* the mana-cost block (applied before that block's own outer weight,
# WEIGHT_MANA_COST) -- keeps the CMC-distance signal from being drowned out by (or itself
# drowning out) the 5 color-pip dimensions purely because one sub-signal has more raw dimensions
# than the other, the same class of bug WEIGHT_* above was built to fix at the top level.
CMC_SUBWEIGHT = 0.5
PIP_SUBWEIGHT = 0.5
MANA_COST_STRUCT_COLS = CMC_BUCKET_COLS + PIP_STRUCT_COLS
POWER_TOUGHNESS_STRUCT_COLS = [
    "struct_power", "struct_has_power", "struct_power_is_variable",
    "struct_toughness", "struct_has_toughness", "struct_toughness_is_variable",
]

KB_DEV_DIR = CONFIG.paths.kb_dir / "dev"

_WUBRG = "WUBRG"


def build_deck_sentences(conn: sqlite3.Connection) -> list[list[str]]:
    """One "sentence" (list of oracle_ids) per deck, from `deck_cards` directly -- commanders
    are never included (the table excludes them by 5.3-B's own design), and no per-slot split:
    unlike 6.1's matrices, Word2Vec trains on the whole corpus at once."""
    decks: dict[int, list[str]] = {}
    for deck_id, oracle_id in conn.execute("SELECT deck_id, oracle_id FROM deck_cards ORDER BY deck_id"):
        decks.setdefault(deck_id, []).append(oracle_id)
    return list(decks.values())


class ShuffledDeckCorpus:
    """Re-iterable gensim corpus: every time it's iterated (once per training epoch), yields
    `shuffles_per_epoch` independently-shuffled copies of each deck's card list. A fresh
    `random.Random` per instance means repeated construction with the same `seed` reproduces the
    same shuffle sequence across separate runs."""

    def __init__(self, decks: list[list[str]], *, shuffles_per_epoch: int = SHUFFLES_PER_EPOCH, seed: int = SEED):
        self.decks = decks
        self.shuffles_per_epoch = shuffles_per_epoch
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[list[str]]:
        for deck in self.decks:
            for _ in range(self.shuffles_per_epoch):
                shuffled = list(deck)
                self._rng.shuffle(shuffled)
                yield shuffled


def train_word2vec(
    sentences: list[list[str]],
    *,
    vector_size: int = VECTOR_SIZE,
    window: int = WINDOW,
    min_count: int = MIN_COUNT,
    epochs: int = EPOCHS,
    shuffles_per_epoch: int = SHUFFLES_PER_EPOCH,
    seed: int = SEED,
    workers: int = 4,
) -> "Word2Vec":
    from gensim.models import Word2Vec  # local import -- see module docstring / top-of-file note

    corpus = ShuffledDeckCorpus(sentences, shuffles_per_epoch=shuffles_per_epoch, seed=seed)
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        seed=seed,
        workers=workers,
        epochs=epochs,
    )
    return model


def word2vec_frame(model: Word2Vec) -> pd.DataFrame:
    """`oracle_id` + `w2v_0..vector_size-1` for every card in the trained vocabulary."""
    oracle_ids = list(model.wv.index_to_key)
    cols = [f"w2v_{i}" for i in range(model.vector_size)]
    if not oracle_ids:
        return pd.DataFrame(columns=["oracle_id", *cols])
    vectors = np.asarray([model.wv[oid] for oid in oracle_ids])
    df = pd.DataFrame(vectors, columns=cols)
    df.insert(0, "oracle_id", oracle_ids)
    return df


def load_text_corpus(conn: sqlite3.Connection) -> pd.DataFrame:
    """`oracle_id`, `name`, `oracle_text`, `type_line`, `is_land` for every commander-legal card
    -- the `tfidf` space's own universe, independent of deck presence. `oracle_text`/`type_line`
    are kept as two separate columns (not concatenated) -- see module docstring for why each
    gets its own independent TF-IDF+SVD pipeline rather than one shared bag-of-words. `is_land`
    feeds `nearest_neighbors`'s land/nonland category filter (`card_categories.py`)."""
    rows = conn.execute(
        "SELECT oracle_id, name, COALESCE(oracle_text, ''), COALESCE(type_line, ''), is_land "
        "FROM cards WHERE legal_commander = 1"
    ).fetchall()
    return pd.DataFrame(rows, columns=["oracle_id", "name", "oracle_text", "type_line", "is_land"])


# sklearn's own TfidfVectorizer default -- used for type_line, where mana/tap symbols never
# appear so there's no need for the oracle-text-specific pattern below.
_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"

# Matches either a full `{...}` mana/tap symbol as one token, or a normal 2+-char word --
# sklearn's default pattern requires 2+ word characters and so silently drops a bare mana symbol
# like `{T}` or `{W}` entirely (a single letter inside non-word-character braces). Used only for
# oracle_text, where "{T}: Add {G}." losing both its symbols to tokenization was the actual bug
# (two different-colored 1-mana dorks tokenizing to virtually the same "add" bag-of-words).
_ORACLE_TEXT_TOKEN_PATTERN = r"\{[^}]+\}|\b\w\w+\b"


def build_tfidf_svd(
    texts: list[str],
    *,
    n_components: int = VECTOR_SIZE,
    min_df: int = 2,
    token_pattern: str = _DEFAULT_TOKEN_PATTERN,
    stop_words: str | None = "english",
    seed: int = SEED,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stop_words, token_pattern=token_pattern)
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return svd.fit_transform(tfidf)


_MANA_SYMBOL_RE = re.compile(r"\{([^}]+)\}")
_LEADING_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _cmc_bucket_vector(cmc: float | None, *, max_bucket: int = CMC_BUCKET_MAX, sigma: float = CMC_BUCKET_SIGMA) -> list[float]:
    """Encode a mana value as a Gaussian bump across integer CMC buckets `0..max_bucket` (the
    last bucket catches everything at or above it) instead of a single z-scored number.

    Cosine similarity is scale-invariant along a shared direction: two colorless cards at CMC 2
    and CMC 5 (vectors `(2,0,0,...)` and `(5,0,0,...)`, or their z-scored equivalents) are
    *perfectly* cosine-similar regardless of the actual 3-point gap, since cosine only measures
    angle, not magnitude, along a shared axis -- so a single-scalar `struct_cmc` fundamentally
    cannot express "penalize a bigger CMC gap more," no matter how it's weighted. A bump
    encoding fixes this at the representation level: two cards' bump vectors overlap less (lower
    cosine similarity) the further apart their CMCs are, smoothly and monotonically -- exactly
    the "a 5-drop should be penalized more than a 3-drop when replacing a 2-drop" behavior asked
    for, and the standard trick for putting an ordinal/distance-sensitive quantity into a
    cosine-similarity space at all."""
    value = 0.0 if cmc is None else max(0.0, min(float(cmc), max_bucket))
    return [math.exp(-((i - value) ** 2) / (2 * sigma**2)) for i in range(max_bucket + 1)]


def _pip_counts(mana_cost: str | None) -> dict[str, int]:
    """Per-color pip counts from a mana cost string, e.g. `{2}{G}{G}{G}` -> `{"G": 3}`. A
    hybrid or Phyrexian symbol (`{G/U}`, `{G/P}`) counts once for each named color -- a rough
    but reasonable approximation of "this card wants this color," not a strict mana-value
    accounting."""
    counts = {c: 0 for c in _WUBRG}
    if not mana_cost:
        return counts
    for symbol in _MANA_SYMBOL_RE.findall(mana_cost):
        for part in symbol.split("/"):
            if part in counts:
                counts[part] += 1
    return counts


def _parse_stat(raw: str | None) -> tuple[float, bool]:
    """Parse a Scryfall power/toughness string into `(numeric_value, is_variable)`.
    Non-creature cards (`raw is None`) get `(0.0, False)`. A clean numeric string parses
    directly. A dual-faced creature's power/toughness is stored as both faces joined with
    `" // "` (e.g. `"1 // 4"`, from `scryfall.py`'s `_face_field`) -- a real, well-defined
    printed baseline (the *front* face's own stat), not a guess, so it takes that face's own
    number and is **not** flagged `is_variable` for this case. Only after that does a genuine
    variable stat (`*`, `1+*`, `7-*`) fall back to its own leading number as a rough baseline
    (0.0 if there isn't one, e.g. bare `*`), flagged `is_variable` so a consumer can tell a real
    printed stat from a guess."""
    if raw is None:
        return 0.0, False
    try:
        return float(raw), False
    except ValueError:
        pass
    if " // " in raw:
        front_face = raw.split(" // ", 1)[0]
        try:
            return float(front_face), False
        except ValueError:
            pass  # front face is itself variable (e.g. "* // 3") -- fall through below
    match = _LEADING_NUMBER_RE.match(raw)
    return (float(match.group()) if match else 0.0), True


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if not std:
        return series - series.mean()
    return (series - series.mean()) / std


def build_structured_features(conn: sqlite3.Connection, oracle_ids: list[str]) -> pd.DataFrame:
    """`struct_*` feature block for `oracle_ids` (assumed to be `cards.legal_commander = 1` rows,
    same universe as `load_text_corpus`) -- see module docstring for what each column captures
    and why it's appended to the TF-IDF space rather than Word2Vec's."""
    placeholders = ",".join("?" * len(oracle_ids))
    rows = conn.execute(
        f"SELECT oracle_id, mana_cost, cmc, power, toughness FROM cards WHERE oracle_id IN ({placeholders})",
        oracle_ids,
    ).fetchall()
    by_id = {r[0]: r[1:] for r in rows}

    records = []
    for oracle_id in oracle_ids:
        mana_cost, cmc, power, toughness = by_id[oracle_id]
        pips = _pip_counts(mana_cost)
        power_value, power_variable = _parse_stat(power)
        toughness_value, toughness_variable = _parse_stat(toughness)
        record = {
            "oracle_id": oracle_id,
            "struct_power": power_value,
            "struct_has_power": 1.0 if power is not None else 0.0,
            "struct_power_is_variable": 1.0 if power_variable else 0.0,
            "struct_toughness": toughness_value,
            "struct_has_toughness": 1.0 if toughness is not None else 0.0,
            "struct_toughness_is_variable": 1.0 if toughness_variable else 0.0,
        }
        for i, weight in enumerate(_cmc_bucket_vector(cmc)):
            record[f"struct_cmc_bucket_{i}"] = weight
        for color in _WUBRG:
            record[f"struct_pip_{color.lower()}"] = pips[color]
        records.append(record)

    df = pd.DataFrame(records)
    # Only the continuous (power/toughness) columns get z-scored. The 0/1 flags
    # (has_power/is_variable) stay literal booleans so "is this even a creature" stays directly
    # readable from the parquet rather than becoming some z-scored fraction, and so their scale
    # doesn't implicitly change if the corpus's own creature ratio shifts on a future rebuild.
    # `struct_cmc_bucket_*` also stays un-z-scored -- it's already a bounded Gaussian-bump
    # encoding (see `_cmc_bucket_vector`); z-scoring would distort the distance-decay shape that
    # is the entire point of the encoding.
    continuous_cols = [
        c for c in df.columns
        if c != "oracle_id"
        and not c.startswith("struct_has_")
        and not c.endswith("_is_variable")
        and not c.startswith("struct_cmc_bucket_")
    ]
    for col in continuous_cols:
        df[col] = _zscore(df[col])
    return df


def build_and_save(
    conn: sqlite3.Connection,
    *,
    out_dir: Path = KB_DEV_DIR,
    tfidf_min_df: int = 2,
    oracle_text_dims: int = ORACLE_TEXT_DIMS,
    types_dims: int = TYPES_DIMS,
) -> dict[str, int]:
    """Build both spaces and write one `embeddings.parquet` under `out_dir`. Returns coverage
    counts for the sanity CLI / devlog. `tfidf_min_df` (and the two `*_dims` params) are exposed
    mainly for tests -- a tiny fixture corpus needs a lower `min_df`/dimensionality than the real
    ~31k-card corpus to have any vocabulary left after pruning at all."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences = build_deck_sentences(conn)
    model = train_word2vec(sentences)
    w2v_df = word2vec_frame(model)

    text_df = load_text_corpus(conn)

    oracle_matrix = build_tfidf_svd(
        text_df["oracle_text"].tolist(),
        n_components=oracle_text_dims,
        min_df=tfidf_min_df,
        token_pattern=_ORACLE_TEXT_TOKEN_PATTERN,
        stop_words=None,
    )
    oracle_df = pd.DataFrame(oracle_matrix, columns=[f"tfidf_{i}" for i in range(oracle_matrix.shape[1])])
    oracle_df.insert(0, "oracle_id", text_df["oracle_id"].values)

    types_matrix = build_tfidf_svd(text_df["type_line"].tolist(), n_components=types_dims, min_df=tfidf_min_df)
    types_df = pd.DataFrame(types_matrix, columns=[f"types_{i}" for i in range(types_matrix.shape[1])])
    types_df.insert(0, "oracle_id", text_df["oracle_id"].values)

    struct_df = build_structured_features(conn, text_df["oracle_id"].tolist())

    merged = text_df[["oracle_id", "name", "is_land"]].merge(oracle_df, on="oracle_id", how="left")
    merged = merged.merge(types_df, on="oracle_id", how="left")
    merged = merged.merge(struct_df, on="oracle_id", how="left")
    merged = merged.merge(w2v_df, on="oracle_id", how="left")

    merged.to_parquet(out_dir / "embeddings.parquet", index=False)

    return {
        "n_cards_total": len(merged),
        "n_word2vec_vocab": len(w2v_df),
        "n_deck_sentences": len(sentences),
        "word2vec_vector_size": model.vector_size,
    }


def load_embeddings(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "embeddings.parquet")


def _space_columns(df: pd.DataFrame, space: str) -> list[str]:
    """Every column `nearest_neighbors` needs present (non-null) for a card to be scoped into
    that space's search at all."""
    if space == "w2v":
        return [c for c in df.columns if c.startswith("w2v_")]
    if space == "tfidf":
        oracle_cols = [c for c in df.columns if c.startswith("tfidf_")]
        types_cols = [c for c in df.columns if c.startswith("types_")]
        return oracle_cols + types_cols + MANA_COST_STRUCT_COLS + POWER_TOUGHNESS_STRUCT_COLS
    raise ValueError(f"Unknown space {space!r}; expected 'w2v' or 'tfidf'")


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    return matrix / norms[:, None]


def _build_space_matrix(scoped: pd.DataFrame, space: str) -> np.ndarray:
    """`w2v` returns its columns as-is (already one homogeneous block). `tfidf` builds the four
    weighted blocks described in the module docstring: each is L2-normalized to unit length per
    row *before* being scaled by its fixed weight, so no block's raw magnitude (nor a card's
    text length/sparsity) can bias the balance on its own -- the concatenation is only ever a sum
    of already-comparable, deliberately-weighted parts.

    `scoped` always carries every column regardless of `space` (it's only row-filtered by
    `nearest_neighbors`, via `_space_columns`, not column-filtered) -- `space` must be passed
    explicitly rather than guessed from which columns are present."""
    if space == "w2v":
        cols = [c for c in scoped.columns if c.startswith("w2v_")]
        return scoped[cols].to_numpy(dtype=np.float64)

    oracle_cols = [c for c in scoped.columns if c.startswith("tfidf_")]
    types_cols = [c for c in scoped.columns if c.startswith("types_")]

    oracle_block = _l2_normalize_rows(scoped[oracle_cols].to_numpy(dtype=np.float64))
    types_block = _l2_normalize_rows(scoped[types_cols].to_numpy(dtype=np.float64))

    # Mana cost is itself two sub-signals (CMC distance, color pips) with very different
    # dimension counts (9 vs 5) -- normalized *separately* and re-combined at a fixed 50/50
    # sub-weight before the block's own outer weight applies, so neither sub-signal can drown
    # out the other purely by raw dimension count (the same class of bug WEIGHT_* fixes one
    # level up). The final _l2_normalize_rows call re-normalizes the combined pair back to unit
    # length so `cost_block` still has the same "unit norm before its outer weight" contract
    # every other block has.
    cmc_sub = _l2_normalize_rows(scoped[CMC_BUCKET_COLS].to_numpy(dtype=np.float64)) * CMC_SUBWEIGHT
    pip_sub = _l2_normalize_rows(scoped[PIP_STRUCT_COLS].to_numpy(dtype=np.float64)) * PIP_SUBWEIGHT
    cost_block = _l2_normalize_rows(np.concatenate([cmc_sub, pip_sub], axis=1))

    pt_block = _l2_normalize_rows(scoped[POWER_TOUGHNESS_STRUCT_COLS].to_numpy(dtype=np.float64))

    is_creature = scoped["struct_has_power"].to_numpy(dtype=np.float64) == 1.0
    half_pt = WEIGHT_POWER_TOUGHNESS / 2.0
    w_types = np.where(is_creature, WEIGHT_TYPES, WEIGHT_TYPES + half_pt)
    w_cost = np.where(is_creature, WEIGHT_MANA_COST, WEIGHT_MANA_COST + half_pt)
    w_pt = np.where(is_creature, WEIGHT_POWER_TOUGHNESS, 0.0)

    return np.concatenate(
        [
            oracle_block * WEIGHT_ORACLE_TEXT,
            types_block * w_types[:, None],
            cost_block * w_cost[:, None],
            pt_block * w_pt[:, None],
        ],
        axis=1,
    )


def nearest_neighbors(
    df: pd.DataFrame, oracle_id: str, *, space: str, k: int = 10, include_cross_category: bool = False
) -> pd.DataFrame:
    """Top-`k` cosine-similarity neighbors of `oracle_id` within one space ('w2v' or 'tfidf' --
    see module docstring for the latter's four fixed-weight blocks). Excludes land/nonland
    cross-category matches by default (`card_categories.same_category_mask` -- a land's ability
    text or deck-composition profile can look textually/distributionally similar to a nonland's
    without being a sane substitution suggestion, and vice versa); pass
    `include_cross_category=True` to opt back in."""
    cols = _space_columns(df, space)
    scoped = df.dropna(subset=cols).reset_index(drop=True)
    match = scoped.index[scoped["oracle_id"] == oracle_id]
    if len(match) == 0:
        raise KeyError(f"{oracle_id!r} has no vector in the {space!r} space.")
    query_is_land = bool(scoped.loc[match[0], "is_land"])

    allowed = same_category_mask(
        scoped["is_land"].to_numpy(), query_is_land, include_cross_category=include_cross_category
    )
    scoped = scoped[allowed].reset_index(drop=True)
    idx = scoped.index[scoped["oracle_id"] == oracle_id][0]

    matrix = _build_space_matrix(scoped, space)
    normed = _l2_normalize_rows(matrix)
    sims = normed @ normed[idx]

    order = np.argsort(-sims)
    results = []
    for other_idx in order:
        if other_idx == idx:
            continue
        results.append(
            {
                "oracle_id": scoped.loc[other_idx, "oracle_id"],
                "name": scoped.loc[other_idx, "name"],
                "similarity": float(sims[other_idx]),
            }
        )
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
    print(f"Built embeddings under {KB_DEV_DIR}:")
    print(json.dumps(stats, indent=2))


def _cmd_neighbors(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        oracle_id = _resolve_card_name(conn, args.card)
    df = load_embeddings()
    table = nearest_neighbors(df, oracle_id, space=args.space, k=args.k, include_cross_category=args.include_cross_category)
    if table.empty:
        print(f"No neighbors found for {args.card!r} in the {args.space!r} space.")
        return
    print(f"Top {len(table)} {args.space} neighbors of {args.card!r}:")
    for _, row in table.iterrows():
        print(f"  {row['similarity']:+.3f}  {row['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build & save embeddings.parquet to data/kb/dev/")
    build_p.set_defaults(func=_cmd_build)

    neighbors_p = sub.add_parser("neighbors", help="Top-k nearest neighbors for a card")
    neighbors_p.add_argument("card", help="Card name")
    neighbors_p.add_argument("--space", choices=["w2v", "tfidf"], default="w2v")
    neighbors_p.add_argument("-k", type=int, default=10)
    neighbors_p.add_argument(
        "--include-cross-category", action="store_true",
        help="Include land/nonland cross-category matches (excluded by default -- see "
        "edhcut.analysis.card_categories)",
    )
    neighbors_p.set_defaults(func=_cmd_neighbors)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
