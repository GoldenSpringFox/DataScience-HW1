"""Generate Chapter 1 + Data figures for the report."""
import sys, json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb

OUT = Path(r"C:/Aviv/University/Semester 8/Data Science/Homework - Group/Project/Report/Images")
OUT.mkdir(parents=True, exist_ok=True)

# --- house style -------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
KEEP_C, DROP_C, COIN_C, NEUTRAL = "#1b7f5e", "#c2452d", "#b8860b", "#9aa3ad"

ci, tscore_w, lift, n2r, r2n, dc = kb.load()
pairs = json.loads((SP / "pairs.json").read_text())


def fig_metric_separation():
    """Four measures, three groups of hand-labelled pairs."""
    groups = [("keep", "synergy (keep)", KEEP_C, "o"),
              ("drop", "no synergy (drop)", DROP_C, "s"),
              ("coincidence", "coincidence", COIN_C, "^")]
    measures = [("lift", "Lift", True), ("t", "t-score", False),
                ("comb", "Synergy score\n(t x log(1+lift))", False), ("jac", "Jaccard overlap", False)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    rng = np.random.default_rng(0)
    for ax, (key, title, logx) in zip(axes, measures):
        for yi, (gk, glabel, color, marker) in enumerate(groups):
            vals = [r[key] for r in pairs[gk]]
            y = np.full(len(vals), yi) + rng.uniform(-0.13, 0.13, len(vals))
            ax.scatter(vals, y, s=34, c=color, marker=marker, alpha=0.85,
                       edgecolors="white", linewidths=0.5, zorder=3)
        if logx:
            ax.set_xscale("log")
        if key == "jac":
            ax.axvline(0.03, color="#333", ls="--", lw=1, zorder=2)
            ax.text(0.036, 2.35, "gate = 0.03", fontsize=7.5, color="#333", va="center")
        ax.set_yticks(range(3))
        ax.set_yticklabels([g[1] for g in groups] if ax is axes[0] else [])
        ax.set_ylim(-0.6, 2.6)
        ax.set_title(title)
        ax.invert_yaxis()
    fig.suptitle("Each measure against the same hand-labelled pairs", y=1.04, fontsize=11)
    fig.savefig(OUT / "fig1_metric_separation.png")
    plt.close(fig)
    print("wrote fig1_metric_separation.png")


def fig_sanguine_bond():
    """Why each metric picks what it picks, for one card."""
    r = n2r["Sanguine Bond"]
    lr = lift.getrow(r).toarray().ravel()
    tr = tscore_w.getrow(r).toarray().ravel()
    comb = tr * np.log1p(np.maximum(lr, 0))
    counts = ci.set_index("row")["deck_count"].sort_index().values
    mask = lr > 0
    idx = np.flatnonzero(mask)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(counts[idx], lr[idx], s=9, c=NEUTRAL, alpha=0.35, edgecolors="none",
               label=f"all {len(idx):,} partners", zorder=1)
    picks = [("lift", lr, "#7b5cd6", "o"), ("t-score", tr, "#0d7fa8", "s"),
             ("synergy score", comb, KEEP_C, "D")]
    for label, vec, color, marker in picks:
        top = np.argsort(-vec)[:10]
        ax.scatter(counts[top], lr[top], s=52, facecolors="none", edgecolors=color,
                   linewidths=1.6, marker=marker, label=f"top 10 by {label}", zorder=3)
    eb = n2r["Exquisite Blood"]
    ax.annotate("Exquisite Blood", (counts[eb], lr[eb]), textcoords="offset points",
                xytext=(10, 6), fontsize=8.5, weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#333"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("partner's deck count (log)")
    ax.set_ylabel("lift (log)")
    ax.set_title("Sanguine Bond's partners: lift chases the rare corner,\nt-score the popular one, the product lands in between")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.savefig(OUT / "fig2_sanguine_bond_partners.png")
    plt.close(fig)
    print("wrote fig2_sanguine_bond_partners.png")


def _gated_adjacency():
    """Sparsified graph with the Jaccard gate applied - max_lift must be measured here."""
    adj = sparse.load_npz(str(SP / "adj_nostaple.npz")).tocsr()
    binary = (adj > 0).astype(np.int32).tocsr()
    degree = np.asarray(binary.sum(axis=1)).ravel()
    upper = sparse.triu(adj, k=1).tocoo()
    u, v, w = upper.row, upper.col, upper.data
    shared = np.asarray(binary[u].multiply(binary[v]).sum(axis=1)).ravel()
    jac = shared / (degree[u] + degree[v] - shared)
    keep = jac >= kb.MIN_JACCARD
    n = adj.shape[0]
    half = sparse.coo_matrix((w[keep], (u[keep], v[keep])), shape=(n, n)).tocsr()
    return (half + half.T).tocsr()


def fig_playrate_maxlift():
    """The staple detector: play rate vs strongest affinity to anything."""
    gated = _gated_adjacency()
    binary = (gated > 0).tocsr()
    counts = ci.set_index("row")["deck_count"].sort_index().values
    n_decks = 13207
    max_lift = np.zeros(gated.shape[0])
    L = lift.tocsr()
    for i in range(gated.shape[0]):
        s, e = binary.indptr[i], binary.indptr[i + 1]
        if e == s:
            continue
        max_lift[i] = L[i, binary.indices[s:e]].toarray().max()
    play = counts / n_decks * 100
    ok = (max_lift > 0) & (play > 0)
    sel = ok & (max_lift < 3.5) & (play >= 10)
    print(f"  max_lift<3.5 & playrate>=10%  selects {int(sel.sum())} cards:",
          ", ".join(sorted(r2n[i] for i in np.flatnonzero(sel))))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(play[ok], max_lift[ok], s=7, c=NEUTRAL, alpha=0.3, edgecolors="none", zorder=1)
    ax.axhline(3.5, color="#333", ls="--", lw=1, zorder=2)
    ax.axvline(10, color="#333", ls="--", lw=1, zorder=2)
    ax.fill_between([10, 100], 0.1, 3.5, color=DROP_C, alpha=0.10, zorder=0)
    LABEL = {"Sol Ring", "Command Tower", "Swords to Plowshares", "Fellwar Stone"}
    for name in kb.STAPLES:
        if name not in n2r:
            continue
        i = n2r[name]
        ax.scatter([play[i]], [max_lift[i]], s=46, c=DROP_C, edgecolors="white",
                   linewidths=0.6, zorder=4)
        if name in LABEL:
            ax.annotate(name, (play[i], max_lift[i]), textcoords="offset points",
                        xytext=(6, -9), fontsize=7, color=DROP_C)
    for name, dx, dy in [("Evolving Wilds", 6, 5), ("Lion Sash", 6, 4)]:
        if name in n2r:
            i = n2r[name]
            ax.scatter([play[i]], [max_lift[i]], s=46, c=KEEP_C, edgecolors="white",
                       linewidths=0.6, zorder=4)
            ax.annotate(name, (play[i], max_lift[i]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=7, color=KEEP_C, weight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("play rate, % of decks (log)")
    ax.set_ylabel("strongest lift to any card (log)")
    ax.set_title("A generic staple is a popular card with no strong affinity to anything")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=DROP_C, label="excluded as format staple"),
        Line2D([], [], marker="o", ls="", color=KEEP_C, label="kept"),
    ], loc="lower left", fontsize=7.5)
    fig.savefig(OUT / "fig3_staple_detector.png")
    plt.close(fig)
    print("wrote fig3_staple_detector.png")


def fig_card_popularity():
    counts = np.sort(ci["deck_count"].values)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(np.arange(1, len(counts) + 1), counts, lw=1.6, color="#0d7fa8")
    ax.axhline(np.median(counts), color=DROP_C, ls="--", lw=1)
    ax.text(1.4, np.median(counts) * 1.25, f"median = {int(np.median(counts))} decks",
            fontsize=8, color=DROP_C)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("card rank by popularity (log)")
    ax.set_ylabel("decks containing the card (log)")
    ax.set_title(f"Card popularity across {len(counts):,} cards in the analysis pool")
    fig.savefig(OUT / "fig0_card_popularity.png")
    plt.close(fig)
    print("wrote fig0_card_popularity.png")


if __name__ == "__main__":
    fig_card_popularity()
    fig_metric_separation()
    fig_sanguine_bond()
    fig_playrate_maxlift()
