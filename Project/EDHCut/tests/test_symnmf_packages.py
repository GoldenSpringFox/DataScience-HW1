import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from edhcut.analysis.cooccurrence import build_card_index
from edhcut.analysis.nmf_packages import card_topics, topic_members
from edhcut.analysis.symnmf_packages import (
    KSweepResult,
    _init_H,
    _relative_residual,
    best_k,
    build_and_save,
    build_color_conditioned_S,
    fit_symnmf,
    load_card_memberships,
    load_components,
    load_pool_index,
    load_S,
    sweep_k,
    topic_proportions_for_deck,
)
from edhcut.db import connect


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "edhcut.db"
    with connect(db_path) as conn:
        yield conn


def _insert_card(conn, oracle_id, *, name=None, color_identity=(), type_line="Creature") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cards (oracle_id, name, color_identity, type_line, legal_commander) "
        "VALUES (?, ?, ?, ?, 1)",
        (oracle_id, name or oracle_id, json.dumps(list(color_identity)), type_line),
    )


def _insert_deck(conn, deck_id, *, commander_oracle_id, slot_key, cards) -> None:
    conn.execute(
        "INSERT INTO decks (deck_id, source, source_id, commander_oracle_id, slot_key) "
        "VALUES (?, 'archidekt', ?, ?, ?)",
        (deck_id, f"d{deck_id}", commander_oracle_id, slot_key),
    )
    conn.executemany(
        "INSERT INTO deck_cards (deck_id, oracle_id, qty) VALUES (?, ?, 1)",
        [(deck_id, oid) for oid in cards],
    )


def _seed_toy_corpus(conn) -> None:
    """Two color-locked, never-co-occurring packages -- mono-red goblins, mono-green ramp --
    across 8 distinct commander slots per color so `compute_near_uniform_weights` and the
    color-conditioned null model both have real (if small) signal to work with.

    One "off" deck per slot runs the commander alone (no a/b/c or d/e/f) -- without it, every
    eligible red/green deck would run its whole package at a 100% inclusion rate, which color
    identity alone perfectly explains (t_color collapses to exactly 0 -- confirmed correct
    behavior of `compute_color_conditioned_tscore`, not a bug, but a degenerate toy corpus for
    testing it: play_rate must be < 100% within the eligible pool for there to be any co-
    occurrence *above* what color-identity eligibility alone predicts)."""
    _insert_card(conn, "cmdr-r", name="Red Commander", color_identity=["R"], type_line="Legendary Creature")
    _insert_card(conn, "cmdr-g", name="Green Commander", color_identity=["G"], type_line="Legendary Creature")
    for oid, name, ci in [
        ("a", "Goblin A", ["R"]), ("b", "Goblin B", ["R"]), ("c", "Goblin C", ["R"]),
        ("d", "Ramp D", ["G"]), ("e", "Ramp E", ["G"]), ("f", "Ramp F", ["G"]),
    ]:
        _insert_card(conn, oid, name=name, color_identity=ci)

    deck_id = 1
    for slot in range(8):
        for _ in range(3):
            _insert_deck(conn, deck_id, commander_oracle_id="cmdr-r", slot_key=f"red-{slot}", cards=["a", "b", "c"])
            deck_id += 1
        _insert_deck(conn, deck_id, commander_oracle_id="cmdr-r", slot_key=f"red-{slot}", cards=[])
        deck_id += 1
    for slot in range(8):
        for _ in range(3):
            _insert_deck(conn, deck_id, commander_oracle_id="cmdr-g", slot_key=f"green-{slot}", cards=["d", "e", "f"])
            deck_id += 1
        _insert_deck(conn, deck_id, commander_oracle_id="cmdr-g", slot_key=f"green-{slot}", cards=[])
        deck_id += 1
    conn.commit()


# --- fit_symnmf / _relative_residual / _init_H ---------------------------------------------------

def test_relative_residual_zero_for_exact_reconstruction() -> None:
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    S = sparse.csr_matrix(H @ H.T)
    s_frob_sq = float((S.data ** 2).sum())
    assert _relative_residual(S, H, s_frob_sq) == pytest.approx(0.0, abs=1e-9)


def test_relative_residual_positive_for_imperfect_reconstruction() -> None:
    H_true = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    S = sparse.csr_matrix(H_true @ H_true.T)
    H_wrong = np.ones((3, 2))
    s_frob_sq = float((S.data ** 2).sum())
    assert _relative_residual(S, H_wrong, s_frob_sq) > 0.05


def test_init_H_shape_nonnegative_and_seed_reproducible() -> None:
    h1 = _init_H(10, 3, s_mean=2.0, seed=7)
    h2 = _init_H(10, 3, s_mean=2.0, seed=7)
    h3 = _init_H(10, 3, s_mean=2.0, seed=8)
    assert h1.shape == (10, 3)
    assert (h1 >= 0).all()
    np.testing.assert_array_equal(h1, h2)
    assert not np.allclose(h1, h3)


def _block_S(n_per_block: int = 4) -> sparse.csr_matrix:
    n = 2 * n_per_block
    dense = np.zeros((n, n))
    dense[:n_per_block, :n_per_block] = 5.0
    dense[n_per_block:, n_per_block:] = 5.0
    np.fill_diagonal(dense, 0.0)
    return sparse.csr_matrix(dense)


def test_fit_symnmf_recovers_block_structure() -> None:
    # Zero diagonal (S never has self-pairs -- see module docstring) puts a hard floor under the
    # achievable residual: (HH^T)_ii = sum(H_i,:^2) can't be 0 unless H_i,: is all-zero, which
    # would also zero out that row's off-diagonal entries. For this exact 2-block, off-diagonal
    # value 5.0 matrix, the true rank-2 optimum works out to relative_residual = 0.5 (solved by
    # hand: minimize 4*h^4 + 12*(5-h^2)^2 over h, the diagonal-vs-off-diagonal trade-off) -- not
    # a solver artifact, so the assertion checks convergence *to* that value, not "near zero".
    S = _block_S()
    result = fit_symnmf(S, k=2, seed=0, max_iter=500)

    assert result.relative_residual < result.residual_history[0]  # fitting actually helped
    assert result.relative_residual == pytest.approx(0.5, abs=0.01)
    # the two blocks should end up dominated by different columns of H
    block_a_topic = np.argmax(result.H[0])
    block_b_topic = np.argmax(result.H[4])
    assert block_a_topic != block_b_topic


def test_fit_symnmf_converges_before_max_iter_on_easy_problem() -> None:
    S = _block_S()
    result = fit_symnmf(S, k=2, seed=0, max_iter=1000, tol=1e-4)
    assert result.n_iter < 1000


# --- sweep_k / best_k ------------------------------------------------------------------------

def test_best_k_maximizes_stability_ties_favor_larger_k() -> None:
    sweep = [
        KSweepResult(k=15, relative_residual=0.6, stability=0.7, fit_seconds=1.0),
        KSweepResult(k=20, relative_residual=0.5, stability=0.9, fit_seconds=1.0),
        KSweepResult(k=25, relative_residual=0.4, stability=0.8, fit_seconds=1.0),
    ]
    assert best_k(sweep).k == 20


def test_best_k_breaks_exact_ties_toward_larger_k() -> None:
    sweep = [
        KSweepResult(k=15, relative_residual=0.6, stability=0.85, fit_seconds=1.0),
        KSweepResult(k=30, relative_residual=0.5, stability=0.85, fit_seconds=1.0),
    ]
    assert best_k(sweep).k == 30


def test_sweep_k_returns_one_result_per_grid_value() -> None:
    S = _block_S()
    sweep = sweep_k(S, k_grid=[2, 3], stability_seeds=[1, 2])
    assert [r.k for r in sweep] == [2, 3]
    for r in sweep:
        assert 0.0 <= r.stability <= 1.0 + 1e-9
        assert r.relative_residual >= 0.0


# --- build_color_conditioned_S -------------------------------------------------------------------

def test_build_color_conditioned_S_symmetric_and_package_structured(db) -> None:
    _seed_toy_corpus(db)
    card_index = build_card_index(db, min_decks=1)

    S, stats = build_color_conditioned_S(db, card_index)

    assert S.shape == (6, 6)
    assert (S != S.T).nnz == 0  # symmetric
    assert (S.data >= 0).all()  # already clipped to non-negative
    # build_cooccurrence's own deck_count only counts decks with >=1 deck_cards row -- the 16
    # "off" decks (no cards) never appear there, only in the decks table itself.
    assert stats["n_decks_total"] == 48

    row = dict(zip(card_index["oracle_id"], card_index["row"]))
    dense = S.toarray()
    assert dense[row["a"], row["b"]] > 0  # within-package: positive
    assert dense[row["a"], row["d"]] == 0  # cross-package (never co-occur): zero


# --- build_and_save end-to-end ----------------------------------------------------------------

def test_build_and_save_writes_tables_and_manifest_and_separates_packages(db, tmp_path) -> None:
    out_dir = tmp_path / "kb"
    out_dir.mkdir()
    _seed_toy_corpus(db)
    card_index = build_card_index(db, min_decks=1)
    card_index.to_parquet(out_dir / "card_index.parquet", index=False)

    stats = build_and_save(db, out_dir=out_dir, k_grid=[2, 3])

    assert stats.n_cards_total == 6  # commanders never appear in deck_cards, only a..f do
    assert stats.n_basic_lands_excluded == 0
    assert stats.n_cards_in_pool == 6

    memberships = load_card_memberships(out_dir)
    assert set(memberships["oracle_id"]) <= {"a", "b", "c", "d", "e", "f"}

    h = load_components(out_dir)
    assert h.shape == (6, stats.chosen_k)  # cards x k -- the transpose of nmf_packages' own H

    pool = load_pool_index(out_dir)
    assert set(pool["oracle_id"]) == {"a", "b", "c", "d", "e", "f"}

    S = load_S(out_dir)
    assert S.shape == (6, 6)

    manifest = json.loads((out_dir / "symnmf_manifest.json").read_text())
    assert manifest["chosen_k"] == stats.chosen_k
    assert manifest["n_cards_in_pool"] == 6

    # sanity: the goblin package and the ramp package should not share a dominant topic
    a_topic = card_topics(memberships, "a").sort_values("share", ascending=False).iloc[0]["topic_id"]
    d_topic = card_topics(memberships, "d").sort_values("share", ascending=False).iloc[0]["topic_id"]
    assert a_topic != d_topic


def test_topic_proportions_for_deck_projects_new_deck(db, tmp_path) -> None:
    out_dir = tmp_path / "kb"
    out_dir.mkdir()
    _seed_toy_corpus(db)
    card_index = build_card_index(db, min_decks=1)
    card_index.to_parquet(out_dir / "card_index.parquet", index=False)

    build_and_save(db, out_dir=out_dir, k_grid=[2])

    h = load_components(out_dir)
    pool = load_pool_index(out_dir)

    proportions = topic_proportions_for_deck(h, pool, {"a", "b", "c"})
    assert proportions
    assert sum(proportions.values()) == pytest.approx(1.0)

    empty = topic_proportions_for_deck(h, pool, {"not-in-pool"})
    assert empty == {}


# --- re-exported card_topics / topic_members ------------------------------------------------------

def test_card_topics_and_topic_members_are_reexported_from_nmf_packages() -> None:
    from edhcut.analysis import symnmf_packages

    assert symnmf_packages.card_topics is card_topics
    assert symnmf_packages.topic_members is topic_members


# --- symnmf_hierarchy.build_and_save wiring (reuses this module's own saved artifacts) -------------

def test_hierarchy_build_and_save_wires_up_to_flat_build_artifacts(db, tmp_path) -> None:
    from edhcut.analysis.symnmf_hierarchy import load_hierarchy
    from edhcut.analysis.symnmf_hierarchy import build_and_save as build_hierarchy_and_save

    out_dir = tmp_path / "kb"
    out_dir.mkdir()
    _seed_toy_corpus(db)
    card_index = build_card_index(db, min_decks=1)
    card_index.to_parquet(out_dir / "card_index.parquet", index=False)
    build_and_save(db, out_dir=out_dir, k_grid=[2])

    stats = build_hierarchy_and_save(
        db, out_dir=out_dir, top_k=2, child_k=2, min_node_cards=2, min_child_stability=0.0, max_depth=2, seed=0
    )
    assert stats["n_nodes"] >= 1

    nodes = load_hierarchy(out_dir=out_dir)
    assert set(nodes.keys()) == {n for n in range(stats["n_nodes"])}
    # the two never-co-occurring packages should end up in different branches of the tree
    root = next(n for n in nodes.values() if n.parent_id is None)
    assert set(root.members["oracle_id"]) == {"a", "b", "c", "d", "e", "f"}
