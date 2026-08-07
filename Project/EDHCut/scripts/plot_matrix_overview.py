"""Ad hoc visualization for a full co-occurrence/PMI/lift matrix from `data/kb/dev/`
(task 6.1, `edhcut.analysis.cooccurrence`). Not part of the tested package/pipeline.

The matrices are too big to look at as literal numbers (a few thousand cards squared), so this
produces two complementary views:
  1. A sparsity/intensity map of every card *relevant to this scope*, reordered by deck
     popularity (most-included first) so any structure is visible instead of the
     alphabetical-by-oracle_id default order. Cards that never appear in this scope at all
     (e.g. Mountain in a Kyler-scoped matrix -- present only because `card_index.parquet`'s
     row numbering is shared across every scope, see `cooccurrence.py`'s module docstring) are
     dropped from the *picture*, not the underlying `.npz` file -- the stored matrix keeps its
     original shared indexing; this script just doesn't waste pixels on rows that are
     guaranteed all-zero.
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
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
from scipy import sparse

from edhcut.analysis.cooccurrence import KB_DEV_DIR, load_card_index


def _cmap_and_norm(data: np.ndarray, vmax: float, *, scale: str, center: float | None = None):
    """Pick a colormap/norm that only spans the sign range actually present in `data`.

    `cooccur` weight is never negative (it's counts), so a diverging red/blue map would
    reserve half its range (and the colorbar) for a blue side that never gets drawn. `pmi`
    can go negative, where the diverging map earns its keep on its own 0-centered terms.

    `lift`'s neutral point isn't 0 (it's never negative either) -- it's 1.0, "exactly chance."
    Pass `center=1.0` to get a diverging blue/red map around that instead of around 0; zero
    cells (the diagonal, plus pairs masked out by `min_pair_count`) are treated as "no data"
    and left white rather than colored as if they meant "far below chance."
    """
    if center is not None:
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad(color="white")
        masked = np.ma.masked_equal(data, 0)
        present = data[data != 0]
        vmin = float(present.min()) if present.size else center
        vmax_actual = float(present.max()) if present.size else center
        # TwoSlopeNorm requires vmin < vcenter < vmax strictly; nudge open if the real data
        # doesn't straddle the center (e.g. every visible lift value happens to be > 1).
        vmin = min(vmin, center - 1e-9)
        vmax_actual = max(vmax_actual, center + 1e-9)
        return cmap, TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax_actual), masked

    has_negative = bool(np.any(data < 0))
    if has_negative:
        cmap = plt.get_cmap("RdBu_r")
        if scale == "log":
            return cmap, SymLogNorm(linthresh=1.0, vmin=-vmax, vmax=vmax), data
        return cmap, plt.Normalize(vmin=-vmax, vmax=vmax), data

    cmap = plt.get_cmap("Reds").copy()
    if scale == "log":
        cmap.set_bad(color="white")
        masked = np.ma.masked_equal(data, 0)
        positive = data[data > 0]
        vmin = float(positive.min()) if positive.size else 1.0
        return cmap, LogNorm(vmin=vmin, vmax=vmax), masked
    return cmap, plt.Normalize(vmin=0, vmax=vmax), data


def plot_overview(kind: str, slot: str, *, top_n: int, out_dir: Path, full_scale: str = "linear") -> None:
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

    # Cards with zero in-scope weight sort to the end (argsort descending); everything before
    # that point is relevant to this scope, everything after is guaranteed all-zero (a card
    # from some other scope riding along in the shared index) and not worth a pixel.
    n_nonzero = int((in_scope_weight[order] > 0).sum())

    # --- 1. relevant-cards-only matrix, popularity-ordered ---
    dense_full = reordered.toarray()[:n_nonzero, :n_nonzero]
    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.abs(dense_full).max() or 1.0
    # Log scale (when requested) log-compresses magnitude so low-count pairs -- flattened to
    # near-white under a linear scale by a handful of staple cards -- become visible; masks
    # true zeros separately from "just small" so they don't get crushed into the same color.
    cmap, norm, plot_data = _cmap_and_norm(dense_full, vmax, scale=full_scale)
    im = ax.imshow(plot_data, cmap=cmap, norm=norm, interpolation="none")
    hidden = matrix.shape[0] - n_nonzero
    scale_note = " (log color scale)" if full_scale == "log" else ""
    ax.set_title(
        f"{kind} ({slot}): {n_nonzero}x{n_nonzero} cards relevant to this scope{scale_note}\n"
        f"({hidden:,} of {matrix.shape[0]:,} cards in the shared index never appear here, hidden), "
        f"{matrix.nnz // 2:,} pairs, {matrix.nnz / max(n_nonzero, 1)**2:.2%} dense\n"
        f"rows/cols ordered by deck popularity (most-included card first)",
        fontsize=8,
    )
    ax.set_xlabel("cards, popularity-ordered")
    ax.set_ylabel("cards, popularity-ordered")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    suffix = "_full_log" if full_scale == "log" else "_full"
    full_path = out_dir / f"{kind}_{slot}{suffix}.png"
    fig.savefig(full_path, dpi=150)
    plt.close(fig)
    print(f"Saved {full_path}")

    # --- 2. numeric-annotated preview of the top-N most popular (in-scope) cards ---
    if n_nonzero < top_n:
        print(f"Only {n_nonzero} cards have any co-occurrence in scope {slot!r}; capping preview there.")
        top_n = max(1, n_nonzero)
    sub = dense_full[:top_n, :top_n]
    sub_names = names[:top_n]
    fig, ax = plt.subplots(figsize=(0.5 * top_n + 2, 0.5 * top_n + 2))
    vmax2 = np.abs(sub).max() or 1.0
    center2 = 1.0 if kind == "lift" else None
    cmap2, norm2, sub_plot = _cmap_and_norm(sub, vmax2, scale="linear", center=center2)
    im = ax.imshow(sub_plot, cmap=cmap2, norm=norm2)
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
    parser.add_argument(
        "--full-scale",
        choices=["linear", "log"],
        default="linear",
        help="Color scale for the full-matrix plot (view 1). 'log' surfaces subtle "
        "differences between low-count pairs that a linear scale flattens out.",
    )
    args = parser.parse_args()
    plot_overview(args.kind, args.slot, top_n=args.top_n, out_dir=args.out_dir, full_scale=args.full_scale)


if __name__ == "__main__":
    main()
