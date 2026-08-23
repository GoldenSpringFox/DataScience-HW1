"""Card embeddings (`edhcut.analysis.embeddings`): mana-cost and power/toughness parsing (hybrid,
Phyrexian, split and dual-faced cards are each their own trap), decks-as-sentences construction
and the shuffled corpus's reproducibility, the TF-IDF/SVD text space, and nearest-neighbour
lookup."""

import re
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from edhcut.analysis.embeddings import (
    CMC_BUCKET_COLS,
    CMC_BUCKET_MAX,
    MANA_COST_STRUCT_COLS,
    PIP_STRUCT_COLS,
    POWER_TOUGHNESS_STRUCT_COLS,
    WEIGHT_MANA_COST,
    WEIGHT_ORACLE_TEXT,
    WEIGHT_POWER_TOUGHNESS,
    WEIGHT_TYPES,
    ShuffledDeckCorpus,
    _build_space_matrix,
    _cmc_bucket_vector,
    _DEFAULT_TOKEN_PATTERN,
    _ORACLE_TEXT_TOKEN_PATTERN,
    _parse_stat,
    _pip_counts,
    build_and_save,
    build_deck_sentences,
    build_structured_features,
    build_tfidf_svd,
    load_embeddings,
    load_text_corpus,
    nearest_neighbors,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, **kwargs) -> None:
    defaults = dict(
        name=oracle_id,
        mana_cost=None,
        cmc=None,
        type_line=None,
        oracle_text=None,
        power=None,
        toughness=None,
        is_land=0,
        legal_commander=1,
    )
    defaults.update(kwargs)
    conn.execute(
        "INSERT INTO cards (oracle_id, name, mana_cost, cmc, type_line, oracle_text, power, "
        "toughness, is_land, legal_commander) VALUES (:oracle_id, :name, :mana_cost, :cmc, "
        ":type_line, :oracle_text, :power, :toughness, :is_land, :legal_commander)",
        {"oracle_id": oracle_id, **defaults},
    )


def _insert_deck(conn, deck_id, cards: list[str]) -> None:
    conn.execute(
        "INSERT INTO decks (deck_id, source, source_id) VALUES (?, 'archidekt', ?)",
        (deck_id, f"d{deck_id}"),
    )
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
        [(deck_id, oid) for oid in cards],
    )


# --- _pip_counts -------------------------------------------------------------------------------

def test_pip_counts_simple_cost() -> None:
    assert _pip_counts("{2}{G}{G}{G}") == {"W": 0, "U": 0, "B": 0, "R": 0, "G": 3}


def test_pip_counts_hybrid_and_phyrexian_count_each_named_color() -> None:
    counts = _pip_counts("{G/U}{G/P}")
    assert counts["G"] == 2
    assert counts["U"] == 1


def test_pip_counts_none_or_empty() -> None:
    assert _pip_counts(None) == {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}
    assert _pip_counts("") == {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}


# --- _parse_stat ---------------------------------------------------------------------------------

def test_parse_stat_none_is_not_a_creature() -> None:
    assert _parse_stat(None) == (0.0, False)


def test_parse_stat_clean_numeric() -> None:
    assert _parse_stat("3") == (3.0, False)
    assert _parse_stat("2.5") == (2.5, False)


def test_parse_stat_variable_with_leading_number() -> None:
    assert _parse_stat("1+*") == (1.0, True)
    assert _parse_stat("7-*") == (7.0, True)


def test_parse_stat_bare_star_has_no_baseline() -> None:
    assert _parse_stat("*") == (0.0, True)


# --- _parse_stat / _pip_counts on dual-faced cards ----------------------------------------------
# Real DFCs (transform/MDFC) store power/toughness and sometimes mana_cost as both faces joined
# with " // " (scryfall.py's _face_field) -- not a "*"-style guess, a well-defined printed stat.

def test_parse_stat_dual_faced_creature_uses_front_face_and_is_not_variable() -> None:
    # Real data: Ulvenwald Captive // Ulvenwald Abomination is "1 // 4" power, "2 // 6" toughness.
    assert _parse_stat("1 // 4") == (1.0, False)
    assert _parse_stat("2 // 6") == (2.0, False)


def test_parse_stat_dual_faced_creature_with_genuinely_variable_front_face() -> None:
    # A hypothetical front face that's itself "*"-based -- no clean baseline on either side, so
    # this must still fall back to the variable-stat path rather than crash or misparse.
    assert _parse_stat("* // 3") == (0.0, True)


def test_pip_counts_split_card_mana_cost_counts_both_halves() -> None:
    # Real data: Fire // Ice's joined mana_cost is "{1}{R} // {1}{U}" -- both halves' colors
    # should count, since the regex just scans for brace groups regardless of "//".
    counts = _pip_counts("{1}{R} // {1}{U}")
    assert counts["R"] == 1
    assert counts["U"] == 1


# --- build_deck_sentences -------------------------------------------------------------------

def test_build_deck_sentences_groups_by_deck(db) -> None:
    for oid in ["a", "b", "c"]:
        _insert_card(db, oid)
    _insert_deck(db, 1, ["a", "b"])
    _insert_deck(db, 2, ["b", "c"])
    db.commit()

    sentences = build_deck_sentences(db)
    assert sorted(sentences, key=len) == [sorted(["a", "b"]), sorted(["b", "c"])] or {
        frozenset(s) for s in sentences
    } == {frozenset(["a", "b"]), frozenset(["b", "c"])}


# --- ShuffledDeckCorpus -----------------------------------------------------------------------

def test_shuffled_deck_corpus_preserves_deck_membership() -> None:
    decks = [["a", "b", "c"], ["d", "e"]]
    corpus = ShuffledDeckCorpus(decks, shuffles_per_epoch=3, seed=1)
    sentences = list(corpus)
    assert len(sentences) == len(decks) * 3
    # every yielded sentence is some permutation of its source deck
    for i, deck in enumerate(decks):
        for sentence in sentences[i * 3 : i * 3 + 3]:
            assert sorted(sentence) == sorted(deck)


def test_shuffled_deck_corpus_reproducible_with_same_seed() -> None:
    decks = [["a", "b", "c", "d", "e", "f"]]
    first = list(ShuffledDeckCorpus(decks, shuffles_per_epoch=5, seed=7))
    second = list(ShuffledDeckCorpus(decks, shuffles_per_epoch=5, seed=7))
    assert first == second


def test_shuffled_deck_corpus_is_reiterable_with_fresh_shuffles() -> None:
    # gensim iterates a corpus_iterable once for build_vocab and again per training epoch --
    # each __iter__ call must keep advancing the same rng, not reset to the same shuffle.
    decks = [["a", "b", "c", "d", "e", "f", "g", "h"]]
    corpus = ShuffledDeckCorpus(decks, shuffles_per_epoch=1, seed=3)
    first_pass = list(corpus)
    second_pass = list(corpus)
    assert first_pass != second_pass


# --- load_text_corpus / build_tfidf_svd --------------------------------------------------------

def test_load_text_corpus_keeps_oracle_text_and_type_line_separate(db) -> None:
    # Kept as two independent columns (not concatenated) -- each gets its own TF-IDF+SVD
    # pipeline now, see module docstring for why a shared bag-of-words backfired.
    _insert_card(db, "a", oracle_text="Draw a card.", type_line="Sorcery")
    _insert_card(db, "b", oracle_text=None, type_line="Land")
    db.commit()

    df = load_text_corpus(db)
    row_a = df[df["oracle_id"] == "a"].iloc[0]
    assert row_a["oracle_text"] == "Draw a card."
    assert row_a["type_line"] == "Sorcery"
    row_b = df[df["oracle_id"] == "b"].iloc[0]
    assert row_b["oracle_text"] == ""
    assert row_b["type_line"] == "Land"


def test_load_text_corpus_excludes_non_commander_legal_cards(db) -> None:
    _insert_card(db, "legal", legal_commander=1)
    _insert_card(db, "banned", legal_commander=0)
    db.commit()

    df = load_text_corpus(db)
    assert set(df["oracle_id"]) == {"legal"}


def test_build_tfidf_svd_shape() -> None:
    texts = [
        "Destroy target creature.",
        "Destroy all creatures.",
        "Draw two cards.",
        "Draw a card, then discard a card.",
        "Search your library for a land card.",
    ]
    reduced = build_tfidf_svd(texts, n_components=3, min_df=1)
    assert reduced.shape == (5, 3)


# --- oracle-text tokenizer: {T}/{W}-style mana symbols must survive as real tokens -------------

def test_default_token_pattern_drops_bare_mana_symbols() -> None:
    # sklearn's own default requires 2+ word characters -- a single letter inside braces never
    # matches, which is the actual bug: "{T}: Add {G}." and "{T}: Add {W}." used to tokenize to
    # virtually the same bag-of-words ("add"), losing the one thing that actually differs.
    tokens = re.findall(_DEFAULT_TOKEN_PATTERN, "{T}: Add {G}.".lower())
    assert tokens == ["add"]


def test_oracle_text_token_pattern_keeps_mana_symbols_as_their_own_tokens() -> None:
    tokens = re.findall(_ORACLE_TEXT_TOKEN_PATTERN, "{T}: Add {G}.".lower())
    assert "{t}" in tokens
    assert "{g}" in tokens
    assert "add" in tokens


def test_oracle_text_token_pattern_distinguishes_different_colors() -> None:
    # The actual regression: two single-color mana dorks differing only in which color they add
    # must not produce an identical token set once mana symbols are real tokens.
    green_dork_tokens = set(re.findall(_ORACLE_TEXT_TOKEN_PATTERN, "{t}: add {g}."))
    white_dork_tokens = set(re.findall(_ORACLE_TEXT_TOKEN_PATTERN, "{t}: add {w}."))
    assert green_dork_tokens != white_dork_tokens


# --- oracle_text TF-IDF: plain bag-of-words, no stopword filtering -----------------------------
# A round trying to fake word-order sensitivity onto TF-IDF (cost/effect tagging + bigrams via a
# custom analyzer) was tried and reverted per user feedback: TF-IDF is bag-of-words and forcing
# structure onto it via preprocessing is fighting the method. A hand-curated "MTG-significant
# words" stopword exception list was tried too and abandoned as fundamentally incomplete --
# instead oracle_text gets no stopword filtering at all, relying on TF-IDF's own IDF term to
# down-weight ubiquitous words automatically.

def test_build_tfidf_svd_no_stopwords_keeps_mtg_significant_words() -> None:
    # "whenever"/"each" would be silently stripped by stop_words="english" -- confirmed live
    # that sklearn's own ENGLISH_STOP_WORDS contains them (and "when"/"at"/"if"/"one"/"more"/
    # "may"/"until", all but "unless").
    vectorizer_input = ["Whenever this creature attacks, each opponent loses 1 life."]
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(min_df=1, stop_words=None, token_pattern=_ORACLE_TEXT_TOKEN_PATTERN)
    vec.fit(vectorizer_input)
    assert "whenever" in vec.vocabulary_
    assert "each" in vec.vocabulary_


def test_build_tfidf_svd_stop_words_none_disables_filtering() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    with_filtering = TfidfVectorizer(min_df=1, stop_words="english")
    with_filtering.fit(["the quick brown fox"])
    without_filtering = TfidfVectorizer(min_df=1, stop_words=None)
    without_filtering.fit(["the quick brown fox"])
    assert "the" not in with_filtering.vocabulary_
    assert "the" in without_filtering.vocabulary_


def test_build_and_save_oracle_text_pipeline_keeps_significant_words(db, tmp_path, monkeypatch) -> None:
    import edhcut.analysis.embeddings as embeddings_module

    class _FakeWV:
        index_to_key: list[str] = []

        def __getitem__(self, key):  # pragma: no cover - never called, empty vocab
            raise KeyError(key)

    class _FakeModel:
        wv = _FakeWV()
        vector_size = embeddings_module.VECTOR_SIZE

    monkeypatch.setattr(embeddings_module, "train_word2vec", lambda sentences: _FakeModel())

    calls: list[dict] = []
    real_build_tfidf_svd = embeddings_module.build_tfidf_svd

    def _spy(texts, **kwargs):
        calls.append(kwargs)
        return real_build_tfidf_svd(texts, **kwargs)

    monkeypatch.setattr(embeddings_module, "build_tfidf_svd", _spy)

    _insert_card(db, "a", oracle_text="Whenever this attacks, each opponent loses life.", type_line="Creature")
    _insert_card(db, "b", oracle_text="Add mana.", type_line="Artifact")
    db.commit()

    build_and_save(db, out_dir=tmp_path, tfidf_min_df=1)
    # oracle_text is built first -- its call must explicitly disable stopword filtering
    assert calls[0].get("stop_words") is None


# --- build_structured_features --------------------------------------------------------------

def test_build_structured_features_flags_creature_vs_noncreature(db) -> None:
    _insert_card(db, "bear", mana_cost="{1}{G}", cmc=2.0, power="2", toughness="2")
    _insert_card(db, "ring", mana_cost="{1}", cmc=1.0, power=None, toughness=None)
    db.commit()

    df = build_structured_features(db, ["bear", "ring"]).set_index("oracle_id")
    assert df.loc["bear", "struct_has_power"] == 1.0
    assert df.loc["ring", "struct_has_power"] == 0.0
    assert df.loc["bear", "struct_has_toughness"] == 1.0
    assert df.loc["ring", "struct_has_toughness"] == 0.0


def test_build_structured_features_flags_variable_power(db) -> None:
    _insert_card(db, "variable", mana_cost="{2}{G}", cmc=3.0, power="1+*", toughness="1+*")
    _insert_card(db, "fixed", mana_cost="{2}{G}", cmc=3.0, power="3", toughness="3")
    db.commit()

    df = build_structured_features(db, ["variable", "fixed"]).set_index("oracle_id")
    assert df.loc["variable", "struct_power_is_variable"] == 1.0
    assert df.loc["fixed", "struct_power_is_variable"] == 0.0


def test_build_structured_features_pip_columns_reflect_mana_cost(db) -> None:
    _insert_card(db, "mono_green", mana_cost="{2}{G}{G}{G}", cmc=5.0)
    _insert_card(db, "gold", mana_cost="{1}{G}{U}", cmc=3.0)
    db.commit()

    df = build_structured_features(db, ["mono_green", "gold"]).set_index("oracle_id")
    # z-scored, but mono_green must carry strictly more green pip weight than the gold card
    assert df.loc["mono_green", "struct_pip_g"] > df.loc["gold", "struct_pip_g"]


def test_build_structured_features_zscores_power_toughness_not_cmc_buckets(db) -> None:
    _insert_card(db, "a", cmc=1.0, power="1", toughness="1")
    _insert_card(db, "b", cmc=2.0, power="2", toughness="2")
    _insert_card(db, "c", cmc=3.0, power="3", toughness="3")
    db.commit()

    df = build_structured_features(db, ["a", "b", "c"])
    assert df["struct_power"].mean() == pytest.approx(0.0, abs=1e-9)
    # cmc buckets are a bounded Gaussian-bump encoding, not z-scored -- values stay in [0, 1]
    for col in CMC_BUCKET_COLS:
        assert df[col].between(0.0, 1.0).all()


# --- _cmc_bucket_vector --------------------------------------------------------------------

def test_cmc_bucket_vector_peaks_at_its_own_bucket() -> None:
    vec = _cmc_bucket_vector(3.0)
    assert len(vec) == CMC_BUCKET_MAX + 1
    assert vec[3] == 1.0
    assert vec[3] == max(vec)


def test_cmc_bucket_vector_clamps_above_max() -> None:
    vec = _cmc_bucket_vector(15.0)
    assert vec[CMC_BUCKET_MAX] == 1.0


def test_cmc_bucket_vector_none_is_bucket_zero() -> None:
    vec = _cmc_bucket_vector(None)
    assert vec[0] == 1.0


def test_cmc_bucket_vector_similarity_decays_with_distance() -> None:
    # The whole point: cosine similarity between two cards' bump vectors must fall off smoothly
    # as their CMC gap grows -- "penalize a 5-drop more than a 3-drop when replacing a 2-drop."
    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    two = _cmc_bucket_vector(2.0)
    three = _cmc_bucket_vector(3.0)
    five = _cmc_bucket_vector(5.0)
    assert cosine(two, three) > cosine(two, five)


# --- dual-faced cards: the full pipeline end to end ---------------------------------------------
# Real transform/MDFC cards store mana_cost/oracle_text/power/toughness as scryfall.py already
# joins them (front-face-only for mana_cost on a typical transform card, " // "-joined for
# oracle_text/power/toughness when both faces have one) -- these tests use the exact shapes
# found live in the DB (Ulvenwald Captive // Ulvenwald Abomination, Fire // Ice) rather than
# invented ones.

def _insert_dfc_card(conn, oracle_id, **kwargs) -> None:
    defaults = dict(
        name=f"{oracle_id} Front // {oracle_id} Back",
        mana_cost="{1}{G}",
        cmc=2.0,
        type_line="Creature — Werewolf Horror // Creature — Eldrazi Werewolf",
        oracle_text="Defender\n{T}: Add {G}. // {T}: Add {C}{C}.",
        power="1 // 4",
        toughness="2 // 6",
        is_land=0,
        legal_commander=1,
    )
    defaults.update(kwargs)
    _insert_card(conn, oracle_id, **defaults)


def test_build_structured_features_dual_faced_creature_power_toughness(db) -> None:
    _insert_dfc_card(db, "captive")
    db.commit()

    row = build_structured_features(db, ["captive"]).set_index("oracle_id").loc["captive"]
    assert row["struct_has_power"] == 1.0
    assert row["struct_has_toughness"] == 1.0
    assert row["struct_power_is_variable"] == 0.0
    assert row["struct_toughness_is_variable"] == 0.0


def test_load_text_corpus_dual_faced_oracle_text_is_tokenizable(db) -> None:
    _insert_dfc_card(db, "captive")
    db.commit()

    df = load_text_corpus(db)
    row = df[df["oracle_id"] == "captive"].iloc[0]
    tokens = re.findall(_ORACLE_TEXT_TOKEN_PATTERN, row["oracle_text"].lower())
    assert "{t}" in tokens
    assert "{g}" in tokens
    assert "{c}" in tokens


def test_build_and_save_includes_dual_faced_cards_without_crashing(db, tmp_path, monkeypatch) -> None:
    # End-to-end: a DFC mixed in with normal cards must not crash TF-IDF/SVD or structured
    # feature building, and must come out with a real, usable vector.
    import edhcut.analysis.embeddings as embeddings_module

    class _FakeWV:
        index_to_key: list[str] = []

        def __getitem__(self, key):  # pragma: no cover - never called, empty vocab
            raise KeyError(key)

    class _FakeModel:
        wv = _FakeWV()
        vector_size = embeddings_module.VECTOR_SIZE

    monkeypatch.setattr(embeddings_module, "train_word2vec", lambda sentences: _FakeModel())

    _insert_dfc_card(db, "captive")
    _insert_card(
        db, "bear", mana_cost="{1}{G}", cmc=2.0, oracle_text="Bear.", type_line="Creature — Bear",
        power="2", toughness="2",
    )
    db.commit()

    stats = build_and_save(db, out_dir=tmp_path, tfidf_min_df=1)
    assert stats["n_cards_total"] == 2

    out = load_embeddings(tmp_path)
    captive = out.set_index("oracle_id").loc["captive"]
    assert captive.filter(like="tfidf_").notna().all()
    assert captive.filter(like="types_").notna().all()
    assert captive["struct_power_is_variable"] == 0.0

    table = nearest_neighbors(out, "captive", space="tfidf", k=1)
    assert list(table["oracle_id"]) == ["bear"]


# --- nearest_neighbors -------------------------------------------------------------------------

def test_nearest_neighbors_ranks_by_cosine_similarity() -> None:
    df = pd.DataFrame(
        {
            "oracle_id": ["a", "b", "c"],
            "name": ["A", "B", "C"],
            "is_land": [False, False, False],
            "w2v_0": [1.0, 1.0, -1.0],
            "w2v_1": [0.0, 0.1, 0.0],
        }
    )
    table = nearest_neighbors(df, "a", space="w2v", k=2)
    assert list(table["oracle_id"]) == ["b", "c"]
    assert table.iloc[0]["similarity"] > table.iloc[1]["similarity"]


def test_nearest_neighbors_unknown_card_raises() -> None:
    df = pd.DataFrame({"oracle_id": ["a"], "name": ["A"], "w2v_0": [1.0]})
    with pytest.raises(KeyError):
        nearest_neighbors(df, "missing", space="w2v")


def _tfidf_row(oracle_id: str, name: str, *, is_creature: bool, is_land: bool = False, cmc: float = 2.0, **overrides) -> dict:
    """A complete row for the 'tfidf' space -- every column `_space_columns`/`_build_space_matrix`
    require present, defaulted to a small nonzero value so no block is degenerately all-zero
    unless a test deliberately overrides one."""
    row = {
        "oracle_id": oracle_id, "name": name, "is_land": is_land,
        "tfidf_0": 1.0, "tfidf_1": 0.5,
        "types_0": 1.0,
        "struct_pip_w": 0.0, "struct_pip_u": 0.0,
        "struct_pip_b": 0.0, "struct_pip_r": 0.0, "struct_pip_g": 1.0,
        "struct_power": 1.0 if is_creature else 0.0,
        "struct_has_power": 1.0 if is_creature else 0.0,
        "struct_power_is_variable": 0.0,
        "struct_toughness": 1.0 if is_creature else 0.0,
        "struct_has_toughness": 1.0 if is_creature else 0.0,
        "struct_toughness_is_variable": 0.0,
    }
    for i, weight in enumerate(_cmc_bucket_vector(cmc)):
        row[f"struct_cmc_bucket_{i}"] = weight
    row.update(overrides)
    return row


def test_nearest_neighbors_tfidf_space_includes_struct_columns() -> None:
    df = pd.DataFrame(
        [
            _tfidf_row("a", "A", is_creature=True, cmc=2.0),
            _tfidf_row("b", "B", is_creature=True, cmc=5.0),
        ]
    )
    # sanity: the function runs end to end and returns the other card
    table = nearest_neighbors(df, "a", space="tfidf", k=1)
    assert list(table["oracle_id"]) == ["b"]


# --- nearest_neighbors: land/nonland category filter (default on) ------------------------------

def test_nearest_neighbors_excludes_cross_category_by_default() -> None:
    df = pd.DataFrame(
        [
            _tfidf_row("dork", "Dork", is_creature=True, is_land=False),
            # An identical vector, but a land -- must not surface as a "neighbor" of a creature.
            _tfidf_row("land", "Land", is_creature=False, is_land=True, tfidf_0=1.0, tfidf_1=0.5),
        ]
    )
    table = nearest_neighbors(df, "dork", space="tfidf", k=5)
    assert table.empty


def test_nearest_neighbors_include_cross_category_opts_back_in() -> None:
    df = pd.DataFrame(
        [
            _tfidf_row("dork", "Dork", is_creature=True, is_land=False),
            _tfidf_row("land", "Land", is_creature=False, is_land=True, tfidf_0=1.0, tfidf_1=0.5),
        ]
    )
    table = nearest_neighbors(df, "dork", space="tfidf", k=5, include_cross_category=True)
    assert list(table["oracle_id"]) == ["land"]


def test_nearest_neighbors_same_category_matches_still_returned() -> None:
    df = pd.DataFrame(
        [
            _tfidf_row("plains", "Plains", is_creature=False, is_land=True, tfidf_0=1.0, tfidf_1=0.5),
            _tfidf_row("island", "Island", is_creature=False, is_land=True, tfidf_0=1.0, tfidf_1=0.5),
        ]
    )
    table = nearest_neighbors(df, "plains", space="tfidf", k=5)
    assert list(table["oracle_id"]) == ["island"]


# --- _build_space_matrix: fixed per-block weights, redistributed for non-creatures -------------

def _segment_norms(matrix: np.ndarray, row: int = 0) -> tuple[float, float, float, float]:
    n_oracle, n_types = 2, 1
    n_cost, n_pt = len(MANA_COST_STRUCT_COLS), len(POWER_TOUGHNESS_STRUCT_COLS)
    vec = matrix[row]
    oracle = np.linalg.norm(vec[:n_oracle])
    types = np.linalg.norm(vec[n_oracle : n_oracle + n_types])
    cost = np.linalg.norm(vec[n_oracle + n_types : n_oracle + n_types + n_cost])
    pt = np.linalg.norm(vec[n_oracle + n_types + n_cost :])
    return oracle, types, cost, pt


def test_build_space_matrix_weights_each_block_for_a_creature() -> None:
    df = pd.DataFrame([_tfidf_row("a", "A", is_creature=True)])
    matrix = _build_space_matrix(df, "tfidf")
    oracle, types, cost, pt = _segment_norms(matrix)
    assert oracle == pytest.approx(WEIGHT_ORACLE_TEXT)
    assert types == pytest.approx(WEIGHT_TYPES)
    assert cost == pytest.approx(WEIGHT_MANA_COST)
    assert pt == pytest.approx(WEIGHT_POWER_TOUGHNESS)


def test_build_space_matrix_redistributes_pt_weight_for_a_noncreature() -> None:
    df = pd.DataFrame([_tfidf_row("a", "A", is_creature=False)])
    matrix = _build_space_matrix(df, "tfidf")
    oracle, types, cost, pt = _segment_norms(matrix)
    half_pt = WEIGHT_POWER_TOUGHNESS / 2
    assert oracle == pytest.approx(WEIGHT_ORACLE_TEXT)
    assert types == pytest.approx(WEIGHT_TYPES + half_pt)
    assert cost == pytest.approx(WEIGHT_MANA_COST + half_pt)
    assert pt == pytest.approx(0.0)


def test_build_space_matrix_weights_sum_to_one_regardless_of_creature_status() -> None:
    creature = _build_space_matrix(pd.DataFrame([_tfidf_row("a", "A", is_creature=True)]), "tfidf")
    noncreature = _build_space_matrix(pd.DataFrame([_tfidf_row("b", "B", is_creature=False)]), "tfidf")
    for matrix in (creature, noncreature):
        oracle, types, cost, pt = _segment_norms(matrix)
        assert oracle + types + cost + pt == pytest.approx(1.0)


# --- build_and_save (TF-IDF/structured only, no gensim) -----------------------------------------

def test_build_and_save_writes_tfidf_and_struct_columns_without_word2vec(db, tmp_path, monkeypatch) -> None:
    # Exercise the non-gensim half of the pipeline end to end. train_word2vec is monkeypatched
    # out so this test runs in any venv, including the repo's main Python 3.14 one that has no
    # gensim installed at all (see module docstring).
    import edhcut.analysis.embeddings as embeddings_module

    class _FakeWV:
        index_to_key: list[str] = []

        def __getitem__(self, key):  # pragma: no cover - never called, empty vocab
            raise KeyError(key)

    class _FakeModel:
        wv = _FakeWV()
        vector_size = embeddings_module.VECTOR_SIZE

    monkeypatch.setattr(embeddings_module, "train_word2vec", lambda sentences: _FakeModel())

    _insert_card(db, "bear", mana_cost="{1}{G}", cmc=2.0, oracle_text="Bear.", type_line="Creature", power="2", toughness="2")
    _insert_card(db, "ring", mana_cost="{1}", cmc=1.0, oracle_text="Add mana.", type_line="Artifact")
    _insert_deck(db, 1, ["bear", "ring"])
    db.commit()

    stats = build_and_save(db, out_dir=tmp_path, tfidf_min_df=1)
    assert stats["n_cards_total"] == 2
    assert stats["n_word2vec_vocab"] == 0

    out = pd.read_parquet(tmp_path / "embeddings.parquet")
    assert {"oracle_id", "name", "is_land", "tfidf_0", "types_0", "struct_cmc_bucket_0", "struct_pip_g"}.issubset(out.columns)
    assert set(out["oracle_id"]) == {"bear", "ring"}


# --- Word2Vec (gensim-only; skipped where gensim has no wheel, e.g. this repo's Python 3.14 ---
# main venv -- run these under venv311 instead, see module docstring / pyproject.toml).

def test_train_word2vec_places_co_occurring_cards_close() -> None:
    pytest.importorskip("gensim")
    from edhcut.analysis.embeddings import train_word2vec, word2vec_frame

    # Two disjoint 4-card "archetypes" that never share a deck. With only 8 words total in the
    # vocabulary, a single-pair comparison is noisy (word2vec's negative sampling assumes a much
    # bigger vocabulary) -- average over every intra- vs inter-group pair instead for a stable
    # signal, and use many more repetitions than a real deck slot would need.
    group_a = ["ramp1", "ramp2", "ramp3", "ramp4"]
    group_b = ["aggro1", "aggro2", "aggro3", "aggro4"]
    decks = [list(group_a) for _ in range(200)] + [list(group_b) for _ in range(200)]
    model = train_word2vec(decks, vector_size=4, window=10, min_count=1, epochs=30, workers=1)
    df = word2vec_frame(model).set_index("oracle_id")

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    intra = [cosine(df.loc[x].to_numpy(), df.loc[y].to_numpy()) for x, y in combinations(group_a, 2)]
    intra += [cosine(df.loc[x].to_numpy(), df.loc[y].to_numpy()) for x, y in combinations(group_b, 2)]
    inter = [cosine(df.loc[x].to_numpy(), df.loc[y].to_numpy()) for x in group_a for y in group_b]
    assert sum(intra) / len(intra) > sum(inter) / len(inter)


def test_word2vec_frame_shape_matches_vocab_and_vector_size() -> None:
    pytest.importorskip("gensim")
    from edhcut.analysis.embeddings import train_word2vec, word2vec_frame

    decks = [["a", "b", "c"], ["a", "b"], ["b", "c"]]
    model = train_word2vec(decks, vector_size=5, window=5, min_count=1, epochs=2, workers=1)
    df = word2vec_frame(model)
    assert set(df.columns) == {"oracle_id", "w2v_0", "w2v_1", "w2v_2", "w2v_3", "w2v_4"}
    assert set(df["oracle_id"]) == {"a", "b", "c"}


def test_build_and_save_includes_word2vec_columns_when_gensim_available(db, tmp_path) -> None:
    pytest.importorskip("gensim")
    from edhcut.analysis.embeddings import build_and_save

    _insert_card(db, "bear", mana_cost="{1}{G}", cmc=2.0, oracle_text="Bear.", type_line="Creature", power="2", toughness="2")
    _insert_card(db, "ring", mana_cost="{1}", cmc=1.0, oracle_text="Add mana.", type_line="Artifact")
    for deck_id in range(5):
        _insert_deck(db, deck_id, ["bear", "ring"])
    db.commit()

    stats = build_and_save(db, out_dir=tmp_path, tfidf_min_df=1)
    assert stats["n_word2vec_vocab"] == 2

    out = pd.read_parquet(tmp_path / "embeddings.parquet")
    assert "w2v_0" in out.columns
    assert out.set_index("oracle_id").loc["bear"].filter(like="w2v_").notna().all()
