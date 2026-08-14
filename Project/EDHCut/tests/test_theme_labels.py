import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from edhcut.analysis.theme_labels import (
    _benjamini_hochberg,
    _singularize,
    build_theme_labels,
    build_theme_tag_map,
    color_purity,
    granularity_report,
    topic_theme_enrichment,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, *, name=None, color_identity=()) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cards (oracle_id, name, color_identity, legal_commander) VALUES (?, ?, ?, 1)",
        (oracle_id, name or oracle_id, json.dumps(list(color_identity))),
    )


def _tag(conn, oracle_id, tag, *, source="tagger_bulk") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO card_tags (oracle_id, tag, source) VALUES (?, ?, ?)", (oracle_id, tag, source)
    )


def _theme(conn, theme, kind, slug, *, num_decks=100) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO edhrec_themes (theme, kind, slug, num_decks, fetched_at) VALUES (?, ?, ?, ?, ?)",
        (theme, kind, slug, num_decks, "2026-01-01T00:00:00"),
    )


# --- _singularize -------------------------------------------------------------------------------

def test_singularize_handles_irregular_plurals() -> None:
    assert _singularize("elves") == "elf"
    assert _singularize("mice") == "mouse"
    assert _singularize("fungi") == "fungus"


def test_singularize_handles_regular_plurals() -> None:
    assert _singularize("goblins") == "goblin"
    assert _singularize("armies") == "army"  # "-ies" -> "-y"
    assert _singularize("boxes") == "box"


def test_singularize_leaves_non_plural_words_alone() -> None:
    assert _singularize("humans") == "human"
    assert _singularize("chess") == "chess"  # ends in "ss", not treated as plural


# --- build_theme_tag_map -------------------------------------------------------------------------

def test_theme_tag_map_exact_slug_match(db) -> None:
    _insert_card(db, "c1")
    _tag(db, "c1", "aristocrats")
    _theme(db, "Aristocrats", "theme", "aristocrats")
    m = build_theme_tag_map(db)
    row = m.loc[m["theme"] == "Aristocrats"].iloc[0]
    assert row["tag"] == "aristocrats"
    assert row["match_source"] == "exact"


def test_theme_tag_map_typal_singularization_match(db) -> None:
    _insert_card(db, "c1")
    _tag(db, "c1", "typal-elf")
    _theme(db, "Elves", "typal", "elves")
    m = build_theme_tag_map(db)
    row = m.loc[m["theme"] == "Elves"].iloc[0]
    assert row["tag"] == "typal-elf"
    assert row["match_source"] == "typal"


def test_theme_tag_map_alias_match(db) -> None:
    _insert_card(db, "c1")
    _tag(db, "c1", "draw")
    _theme(db, "Card Draw", "theme", "card-draw")
    m = build_theme_tag_map(db)
    row = m.loc[m["theme"] == "Card Draw"].iloc[0]
    assert row["tag"] == "draw"
    assert row["match_source"] == "alias"


def test_theme_tag_map_no_match_is_none(db) -> None:
    _theme(db, "Totally Unmapped Theme", "theme", "totally-unmapped-theme")
    m = build_theme_tag_map(db)
    row = m.loc[m["theme"] == "Totally Unmapped Theme"].iloc[0]
    assert row["tag"] is None
    assert row["match_source"] == "none"


# --- build_theme_labels --------------------------------------------------------------------------

def test_theme_labels_renames_matched_tag_to_theme_name(db) -> None:
    for i in range(25):
        _insert_card(db, f"c{i}")
        _tag(db, f"c{i}", "aristocrats")
    _theme(db, "Aristocrats", "theme", "aristocrats")

    labels = build_theme_labels(db, min_cards=1, max_cards=1000)
    row = labels.loc[labels["oracle_id"] == "c0"].iloc[0]
    assert row["label"] == "Aristocrats"
    assert row["source"] == "theme"
    assert row["edhrec_theme"] == "Aristocrats"


def test_theme_labels_unmatched_tag_keeps_raw_tag_as_label(db) -> None:
    for i in range(25):
        _insert_card(db, f"c{i}")
        _tag(db, f"c{i}", "some-mechanical-tag")

    labels = build_theme_labels(db, min_cards=1, max_cards=1000)
    row = labels.loc[labels["oracle_id"] == "c0"].iloc[0]
    assert row["label"] == "some-mechanical-tag"
    assert row["source"] == "tag"
    assert pd.isna(row["edhrec_theme"])


def test_theme_labels_respects_min_and_max_card_count(db) -> None:
    for i in range(5):
        _insert_card(db, f"rare{i}")
        _tag(db, f"rare{i}", "too-rare-tag")
    for i in range(25):
        _insert_card(db, f"ok{i}")
        _tag(db, f"ok{i}", "in-range-tag")

    labels = build_theme_labels(db, min_cards=10, max_cards=1000)
    assert "too-rare-tag" not in set(labels["label"])
    assert "in-range-tag" in set(labels["label"])


def test_theme_labels_pool_restricts_which_cards_count_toward_size(db) -> None:
    for i in range(25):
        _insert_card(db, f"c{i}")
        _tag(db, f"c{i}", "shared-tag")

    pool = {f"c{i}" for i in range(5)}  # only 5 of the 25 tagged cards are "in the pool"
    labels = build_theme_labels(db, pool_oracle_ids=pool, min_cards=10, max_cards=1000)
    assert "shared-tag" not in set(labels["label"])  # only 5 pool cards carry it, below min_cards=10

    labels_unrestricted = build_theme_labels(db, min_cards=10, max_cards=1000)
    assert "shared-tag" in set(labels_unrestricted["label"])  # 25 cards globally, passes


# --- _benjamini_hochberg -------------------------------------------------------------------------

def test_benjamini_hochberg_matches_hand_computed_example() -> None:
    p = np.array([0.04, 0.005, 0.20, 0.01, 0.03])  # deliberately out of order
    q = _benjamini_hochberg(p)
    # Sorted ascending: 0.005, 0.01, 0.03, 0.04, 0.20 -> BH q: 0.025, 0.025, 0.05, 0.05, 0.20
    expected_sorted_q = {0.005: 0.025, 0.01: 0.025, 0.03: 0.05, 0.04: 0.05, 0.20: 0.20}
    for pi, qi in zip(p, q):
        assert qi == pytest.approx(expected_sorted_q[pi])


def test_benjamini_hochberg_never_exceeds_one_and_is_monotone_in_sorted_order() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(50)
    q = _benjamini_hochberg(p)
    assert (q <= 1.0).all()
    order = np.argsort(p)
    assert (np.diff(q[order]) >= -1e-12).all()  # non-decreasing once read in p-sorted order


def test_benjamini_hochberg_empty_input() -> None:
    assert len(_benjamini_hochberg(np.array([]))) == 0


# --- topic_theme_enrichment + granularity_report --------------------------------------------------

def _synthetic_memberships() -> pd.DataFrame:
    """Two overlapping soft topics over a 10-card pool -- see the module test's own inline
    comments for the hand-computed lift/share/recall/resolution values this is built to check."""
    topic0 = [f"c{i}" for i in range(1, 7)]  # c1..c6
    topic1 = [f"c{i}" for i in range(5, 11)]  # c5..c10
    rows = [{"oracle_id": oid, "topic_id": 0, "weight": 1.0} for oid in topic0]
    rows += [{"oracle_id": oid, "topic_id": 1, "weight": 1.0} for oid in topic1]
    return pd.DataFrame(rows)


def _synthetic_labels() -> pd.DataFrame:
    a = [f"c{i}" for i in (1, 2, 3, 4)]
    b = [f"c{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8)]
    c = [f"c{i}" for i in (9, 10)]
    rows = []
    for label, members in (("A", a), ("B", b), ("C", c)):
        for oid in members:
            rows.append({"oracle_id": oid, "label": label, "source": "tag", "edhrec_theme": None})
    return pd.DataFrame(rows)


def test_topic_theme_enrichment_hand_computed_values() -> None:
    memberships = _synthetic_memberships()
    labels = _synthetic_labels()

    result = topic_theme_enrichment(memberships, labels, min_cards=1, min_lift=0.0, fdr=1.0)

    row_a0 = result[(result["topic_id"] == 0) & (result["label"] == "A")].iloc[0]
    assert row_a0["n_in_topic"] == 4
    assert row_a0["share_of_topic"] == pytest.approx(4 / 6)
    assert row_a0["recall_of_label"] == pytest.approx(1.0)
    assert row_a0["resolution"] == pytest.approx(4 / 6)
    assert row_a0["lift"] == pytest.approx(4 / (6 * 4 / 10))

    row_b0 = result[(result["topic_id"] == 0) & (result["label"] == "B")].iloc[0]
    assert row_b0["n_in_topic"] == 6
    assert row_b0["share_of_topic"] == pytest.approx(1.0)
    assert row_b0["recall_of_label"] == pytest.approx(6 / 8)
    assert row_b0["resolution"] == pytest.approx(6 / 8)

    row_b1 = result[(result["topic_id"] == 1) & (result["label"] == "B")].iloc[0]
    assert row_b1["n_in_topic"] == 4
    assert row_b1["share_of_topic"] == pytest.approx(4 / 6)
    assert row_b1["recall_of_label"] == pytest.approx(4 / 8)

    row_c1 = result[(result["topic_id"] == 1) & (result["label"] == "C")].iloc[0]
    assert row_c1["n_in_topic"] == 2
    assert row_c1["resolution"] == pytest.approx(min(2 / 6, 2 / 2))

    # topic 0 x C never appears -- C has zero overlap with topic 0's membership
    assert result[(result["topic_id"] == 0) & (result["label"] == "C")].empty

    # p-values match a direct hypergeom.sf computation (N=10 universe)
    assert row_a0["p"] == pytest.approx(float(hypergeom.sf(4 - 1, 10, 4, 6)))


def test_topic_theme_enrichment_min_cards_filters_small_overlaps() -> None:
    memberships = _synthetic_memberships()
    labels = _synthetic_labels()
    result = topic_theme_enrichment(memberships, labels, min_cards=3, min_lift=0.0, fdr=1.0)
    # label C's only overlap (topic 1, k=2) is below min_cards=3
    assert result[result["label"] == "C"].empty


def test_topic_theme_enrichment_significance_filters_are_applied() -> None:
    memberships = _synthetic_memberships()
    labels = _synthetic_labels()
    lenient = topic_theme_enrichment(memberships, labels, min_cards=1, min_lift=0.0, fdr=1.0)
    strict = topic_theme_enrichment(memberships, labels, min_cards=1, min_lift=1000.0, fdr=1.0)
    assert len(strict) < len(lenient)
    assert strict.empty


def test_granularity_report_hand_computed_counts() -> None:
    memberships = _synthetic_memberships()
    labels = _synthetic_labels()

    report = granularity_report(
        memberships, labels, min_cards=1, min_lift=0.0, fdr=1.0, thresholds=(0.3, 0.5, 0.7, 0.8)
    )

    assert report["vocab_size"] == 3  # labels A, B, C
    assert report["n_topics"] == 2
    assert report["n_labels_covered"] == 3
    # best resolution per label: A=4/6=.667, B=max(.75, .5)=.75, C=2/6=.333
    assert report["n_labels_resolved"]["0.30"] == 3
    assert report["n_labels_resolved"]["0.50"] == 2
    assert report["n_labels_resolved"]["0.70"] == 1
    assert report["n_labels_resolved"]["0.80"] == 0
    assert report["median_labels_per_topic"] == pytest.approx(2.0)  # both topics enrich 2 labels


def test_granularity_report_handles_noncontiguous_topic_ids() -> None:
    # A hierarchy cut's topic_id is a node_id (arbitrary, non-contiguous across the whole tree,
    # never 0..n_topics-1) -- found live against the real symnmf_hierarchy build, where this
    # silently produced median_labels_per_topic=0.0 for every cut (a `reindex(range(n_topics))`
    # against topic_ids like 1000/2000 matches nothing). Same fixture as the hand-computed test
    # above, topic_id 0->1000 and 1->2000, so the correct answer is identical: median 2.0.
    memberships = _synthetic_memberships()
    memberships["topic_id"] = memberships["topic_id"].map({0: 1000, 1: 2000})
    labels = _synthetic_labels()

    report = granularity_report(memberships, labels, min_cards=1, min_lift=0.0, fdr=1.0)

    assert report["n_topics"] == 2
    assert report["median_labels_per_topic"] == pytest.approx(2.0)


def test_granularity_report_empty_enrichment_returns_zeroed_fields() -> None:
    memberships = _synthetic_memberships()
    labels = _synthetic_labels()
    report = granularity_report(memberships, labels, min_cards=1, min_lift=1000.0, fdr=1.0)
    assert report["n_labels_covered"] == 0
    assert all(n == 0 for n in report["n_labels_resolved"].values())
    assert report["median_labels_per_topic"] == 0.0


# --- color_purity ---------------------------------------------------------------------------------

def test_color_purity_top_n_all_same_color_is_pure(db) -> None:
    colors = {"c1": ["R"], "c2": ["R"], "c3": ["R"], "c4": ["U"], "c5": ["U"], "c6": ["G"]}
    for oid, ci in colors.items():
        _insert_card(db, oid, color_identity=ci)
    weights = {"c1": 6, "c2": 5, "c3": 4, "c4": 3, "c5": 2, "c6": 1}
    memberships = pd.DataFrame(
        [{"oracle_id": oid, "topic_id": 0, "weight": w} for oid, w in weights.items()]
    )

    purity_df, null = color_purity(memberships, db, top_n=3, n_null_draws=50, seed=1)

    row = purity_df.iloc[0]
    assert row["topic_id"] == 0
    assert row["modal_color_identity"] == "R"
    assert row["purity"] == pytest.approx(1.0)
    assert len(null) == 50
    assert ((null >= 0) & (null <= 1)).all()


def test_color_purity_full_pool_gives_exact_deterministic_share(db) -> None:
    colors = {"c1": ["R"], "c2": ["R"], "c3": ["R"], "c4": ["U"], "c5": ["U"], "c6": ["G"]}
    for oid, ci in colors.items():
        _insert_card(db, oid, color_identity=ci)
    memberships = pd.DataFrame(
        [{"oracle_id": oid, "topic_id": 0, "weight": 1.0} for oid in colors]
    )

    purity_df, _ = color_purity(memberships, db, top_n=6, n_null_draws=5, seed=1)
    row = purity_df.iloc[0]
    assert row["modal_color_identity"] == "R"
    assert row["purity"] == pytest.approx(3 / 6)


def test_color_purity_colorless_cards_get_c_identity(db) -> None:
    _insert_card(db, "c1", color_identity=[])
    _insert_card(db, "c2", color_identity=[])
    memberships = pd.DataFrame(
        [{"oracle_id": oid, "topic_id": 0, "weight": 1.0} for oid in ("c1", "c2")]
    )
    purity_df, _ = color_purity(memberships, db, top_n=2, n_null_draws=5, seed=1)
    assert purity_df.iloc[0]["modal_color_identity"] == "C"
    assert purity_df.iloc[0]["purity"] == pytest.approx(1.0)
