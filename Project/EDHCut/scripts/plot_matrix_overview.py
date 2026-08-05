"""Ad hoc visualization for a full co-occurrence/PMI/lift matrix from `data/kb/dev/`
(task 6.1, `edhcut.analysis.cooccurrence`). Not part of the tested package/pipeline.

The matrices are too big to look at as literal numbers (a few thousand cards squared), so this
produces two complementary views:
  1. A full sparsity/intensity map, rows & columns reordered by deck popularity (most-included
     first) so any structure is visible instead of the alphabetical-by-oracle_id default order.
  2. A numeric-annotated heatmap of just the top-N most popular cards, small enough to read
     actual values off directly.

Usage:
    python scripts/plot_matrix_overview.py cooccur kyler
    python scripts/plot_matrix_overview.py pmi global --top 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from edhcut.analysis.cooccurrence import KB_DEV_DIR, load_card_index


def plot_overview(kind: str, slot: str, *, top_n: int, out_dir: Path) -> None:
    card_index = load_card_index()
    matrix = sparse.load_npz(KB_DEV_DIR / f"{kind}_{slot}.npz")

    # Popularity *within this scope*, not the shared card_index's global deck_count — a card
    # can be globally popular (e.g. Mountain, dominated by other slots' corpora) while never
    # appearing in this scope's decks at all, in which case its row here is all zero. Using
    # global popularity to pick/order the "top N" would silently pull in such irrelevant cards
    # (their row would just look blank) instead of leaving them out. Row-wise absolute sum of
    # this matrix is a reasonable in-scope popularity proxy and is exactly 0 for any card this
    # scope never touches, so it can't make that mistake.
    in_scope_weight = np.abs(matrix).sum(axis=1).A1
    order = np.argsort(-in_scope_weight)
    names = card_index["name"].to_numpy()[order]
    reordered = matrix[order, :][:, order]

    # --- 1. full matrix, popularity-ordered ---
    dense_full = reordered.toarray()
    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.abs(dense_full).max() or 1.0
    im = ax.imshow(dense_full, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="none")
    ax.set_title(
        f"{kind} ({slot}): full {matrix.shape[0]}x{matrix.shape[1]} matrix, "
        f"{matrix.nnz // 2:,} pairs, {matrix.nnz / matrix.shape[0]**2:.2%} dense\n"
        f"rows/cols ordered by deck popularity (most-included card first)",
        fontsize=9,
    )
    ax.set_xlabel("cards, popularity-ordered")
    ax.set_ylabel("cards, popularity-ordered")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    full_path = out_dir / f"{kind}_{slot}_full.png"
    fig.savefig(full_path, dpi=150)
    plt.close(fig)
    print(f"Saved {full_path}")

    # --- 2. numeric-annotated preview of the top-N most popular (in-scope) cards ---
    n_nonzero = int((in_scope_weight[order] > 0).sum())
    if n_nonzero < top_n:
        print(f"Only {n_nonzero} cards have any co-occurrence in scope {slot!r}; capping preview there.")
        top_n = max(1, n_nonzero)
    sub = dense_full[:top_n, :top_n]
    sub_names = names[:top_n]
    fig, ax = plt.subplots(figsize=(0.5 * top_n + 2, 0.5 * top_n + 2))
    vmax2 = np.abs(sub).max() or 1.0
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2)
    ax.set_xticks(range(top_n))
    ax.set_yticks(range(top_n))
    ax.set_xticklabels(sub_names, rotation=90, fontsize=7)
    ax.set_yticklabels(sub_names, fontsize=7)
    for i in range(top_n):
        for j in range(top_n):
            if sub[i, j] != 0:
                ax.text(j, i, f"{sub[i, j]:.2g}", ha="center", va="center", fontsize=5)
    ax.set_title(f"{kind} ({slot}): top {top_n} most-included cards, actual values", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    preview_path = out_dir / f"{kind}_{slot}_top{top_n}.png"
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    print(f"Saved {preview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["cooccur", "pmi", "lift"])
    parser.add_argument("slot", help="Scope label, e.g. 'global', 'krenko', 'kyler'")
    parser.add_argument("--top", type=int, default=25, dest="top_n")
    parser.add_argument("--out-dir", type=Path, default=KB_DEV_DIR)
    args = parser.parse_args()
    plot_overview(args.kind, args.slot, top_n=args.top_n, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
