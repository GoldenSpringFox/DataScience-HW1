"""Chapter 1 figures, rebuilt as a four-beat story. Linear axes throughout."""
import sys, json
from pathlib import Path
import numpy as np, scipy.sparse as sparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb

OUT = Path(r"C:/Aviv/University/Semester 8/Data Science/Homework - Group/Project/Report/Images")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
KEEP_C, DROP_C, COIN_C, NEUTRAL, T_C = "#1b7f5e", "#c2452d", "#b8860b", "#9aa3ad", "#0d7fa8"

ci, tscore, lift, n2r, r2n, dc = kb.load()
counts = ci.set_index("row")["deck_count"].sort_index().values
co = sparse.load_npz("data/kb/dev/cooccur_global.npz").tocsr()
pairs = json.loads((SP / "pairs.json").read_text())


def _top_pairs(matrix, n, min_value=None):
    m = sparse.triu(matrix, k=1).tocoo()
    order = np.argsort(-m.data)[:n]
    return [(m.row[i], m.col[i], m.data[i]) for i in order]


def fig_popularity():
    c = np.sort(ci["deck_count"].values)[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(np.arange(1, len(c) + 1), c, lw=1.8, color=T_C)
    ax.axhline(np.median(c), color=DROP_C, ls="--", lw=1)
    ax.text(len(c) * 0.42, np.median(c) + 260, f"median card: {int(np.median(c))} decks",
            fontsize=8.5, color=DROP_C)
    ax.set_xlabel("cards, ordered from most to least played")
    ax.set_ylabel("decks containing the card")
    ax.set_title("Almost every card is played in almost no decks")
    ax.set_xlim(0, len(c)); ax.set_ylim(0, None)
    fig.savefig(OUT / "fig1_card_popularity.png"); plt.close(fig)
    print("fig1_card_popularity.png")


def _pair_bars(ax, rows, value_fmt, color, xlabel):
    labels = [f"{r_[0]} + {r_[1]}" for r_ in rows]
    bars = [r_[2] for r_ in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, bars, color=color, alpha=0.85, height=0.7)
    for yi, r_ in zip(y, rows):
        ax.text(r_[2] + max(bars) * 0.015, yi, value_fmt.format(r_[3]),
                va="center", fontsize=7.4, color=color)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(bars) * 1.42)


def fig_lift_fails():
    top = _top_pairs(lift, 10)
    rows = [(r2n[i], r2n[j], int(co[i, j]), v) for i, j, v in top]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    _pair_bars(ax, rows, "lift = {:,.0f}", COIN_C, "decks containing both cards")
    ax.set_title("Lift's ten highest-scoring pairs in the whole corpus rest on three decks each")
    fig.savefig(OUT / "fig2_lift_fails.png"); plt.close(fig)
    print("fig2_lift_fails.png")
    for r_ in rows[:3]:
        print("   ", r_)


def fig_tscore_fails():
    basics = {n2r[n] for n in kb.BASICS if n in n2r}
    top = [t_ for t_ in _top_pairs(tscore, 60)
           if t_[0] not in basics and t_[1] not in basics][:10]
    rows = [(r2n[i], r2n[j], int(min(counts[i], counts[j])), v) for i, j, v in top]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    _pair_bars(ax, rows, "t = {:,.1f}", T_C, "decks containing the rarer of the two cards")
    ax.set_title("t-score's ten highest-scoring pairs are between cards everybody already plays")
    fig.savefig(OUT / "fig3_tscore_fails.png"); plt.close(fig)
    print("fig3_tscore_fails.png")
    for r_ in rows[:3]:
        print("   ", r_)


def _strip(ax, key, xlabel, gate=None):
    groups = [("keep", "real synergy", KEEP_C, "o"),
              ("drop", "no synergy", DROP_C, "s"),
              ("coincidence", "coincidence", COIN_C, "^")]
    rng = np.random.default_rng(0)
    for yi, (gk, glabel, color, marker) in enumerate(groups):
        vals = [r[key] for r in pairs[gk]]
        y = np.full(len(vals), yi) + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(vals, y, s=46, c=color, marker=marker, alpha=0.9,
                   edgecolors="white", linewidths=0.6, zorder=3)
    if gate is not None:
        ax.axvline(gate, color="#333", ls="--", lw=1.2, zorder=2)
        ax.text(gate + 0.008, -0.42, f"gate = {gate}", fontsize=8, color="#333")
    ax.set_yticks(range(3)); ax.set_yticklabels([g[1] for g in groups])
    ax.set_ylim(-0.6, 2.5); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)


def fig_product_works():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.1))
    _strip(axes[0], "t", "t-score")
    _strip(axes[1], "comb", "synergy score  (t-score x log(1 + lift))")
    axes[1].set_yticklabels([])
    for ax, sub in zip(axes, ["kills the coincidences, but 6 of 9 generic pairs\noutscore the weakest real one",
                              "generic overlap drops to 2 of 9, at the cost of\nletting one coincidence back in"]):
        ax.set_title(sub, fontsize=9, color="#444")
    fig.suptitle("Multiplying the two measures roughly thirds the overlap between real and generic pairs",
                 y=1.10, fontsize=11)
    fig.savefig(OUT / "fig4_product_works.png"); plt.close(fig)
    print("fig4_product_works.png")


def fig_jaccard():
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    _strip(ax, "jac", "Jaccard overlap of the two cards' neighbourhoods", gate=0.03)
    ax.set_title("Neighbourhood overlap decides what no score could: which pairings are real")
    fig.savefig(OUT / "fig5_jaccard_gate.png"); plt.close(fig)
    print("fig5_jaccard_gate.png")


def fig_staples():
    from figs import _gated_adjacency
    gated = _gated_adjacency(); binary = (gated > 0).tocsr()
    L = lift.tocsr(); max_lift = np.zeros(gated.shape[0])
    for i in range(gated.shape[0]):
        s, e = binary.indptr[i], binary.indptr[i + 1]
        if e > s:
            max_lift[i] = L[i, binary.indices[s:e]].toarray().max()
    play = counts / 13207 * 100
    # restrict to the decision region so the axes can stay linear
    ok = (max_lift > 0) & (play >= 1) & (max_lift <= 40)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.scatter(play[ok], max_lift[ok], s=12, c=NEUTRAL, alpha=0.35, edgecolors="none", zorder=1)
    ax.axhline(3.5, color="#333", ls="--", lw=1.1); ax.axvline(10, color="#333", ls="--", lw=1.1)
    ax.fill_between([10, 95], 0, 3.5, color=DROP_C, alpha=0.10, zorder=0)
    LABEL = {"Sol Ring": (8, -4), "Command Tower": (-8, -12), "Swords to Plowshares": (8, 4)}
    for name in kb.STAPLES:
        i = n2r[name]
        ax.scatter([play[i]], [max_lift[i]], s=52, c=DROP_C, edgecolors="white", lw=0.6, zorder=4)
        if name in LABEL:
            ax.annotate(name, (play[i], max_lift[i]), textcoords="offset points",
                        xytext=LABEL[name], fontsize=7.4, color=DROP_C)
    for name in ("Evolving Wilds", "Skullclamp"):
        if name in n2r and play[n2r[name]] >= 1:
            i = n2r[name]
            ax.scatter([play[i]], [max_lift[i]], s=52, c=KEEP_C, edgecolors="white", lw=0.6, zorder=4)
            ax.annotate(name, (play[i], max_lift[i]), textcoords="offset points",
                        xytext=(7, 3), fontsize=7.2, color=KEEP_C, weight="bold")
    ax.set_xlabel("play rate (% of all decks)")
    ax.set_ylabel("strongest lift to any other card")
    ax.set_xlim(0, 95); ax.set_ylim(0, 40)
    ax.set_title("A format staple is a card everyone plays that wants nothing in particular")
    fig.savefig(OUT / "fig6_staple_detector.png"); plt.close(fig)
    print("fig6_staple_detector.png")


if __name__ == "__main__":
    fig_popularity(); fig_lift_fails(); fig_tscore_fails()
    fig_product_works(); fig_jaccard(); fig_staples()
