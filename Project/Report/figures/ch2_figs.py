"""Chapter 2 figures."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
OUT = Path(__file__).resolve().parents[1] / "Images"   # Project/Report/Images

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
KEEP_C, DROP_C, COIN_C, NEUTRAL, T_C = "#1b7f5e", "#c2452d", "#b8860b", "#9aa3ad", "#0d7fa8"
RANK_CAP = 10000   # past this the pool thins out and the bins get meaninglessly wide

D = pickle.load(open(SP / "ch2.pkl", "rb"))
cards, share, floored, S = D["cards"], D["share"], D["floored"], D["S"]
names = D["names"]
cards["theme"] = cards["community"].map(names).fillna("(generic)")

# Shorter, plainer names for the graph. The tag labels repeat themselves across sibling
# communities ("synergy-instant/synergy-sorcery" vs "synergy-sorcery/synergy-instant"), and
# community 58 is creature-type-agnostic typal payoffs rather than any one type.
DISPLAY = {
    164: "goblins",
    87:  "power matters",
    0:   "equipment",
    41:  "red removal",
    75:  "tokens",
    73:  "sorceries",
    58:  "typal payoffs (any type)",
    8:   "+1/+1 counters",
    116: "enchantments",
    93:  "instants",
    61:  "life payment",
    43:  "auras",
    115: "sacrifice outlets",
    51:  "draw / counterspells",
}


def fig_purity():
    """**Figure 14.** Colour purity per community against a same-size random null — the check that
    the packages are not just colour groupings."""
    assigned = cards[cards["community"] >= 0]
    rng = np.random.default_rng(0)
    pool = assigned["color_identity"].values
    real, null, sizes = [], [], []
    for cid, grp in assigned.groupby("community"):
        if len(grp) < 8:
            continue
        real.append(grp["color_identity"].value_counts().iloc[0] / len(grp))
        draws = [pd.Series(rng.choice(pool, len(grp))).value_counts().iloc[0] / len(grp)
                 for _ in range(20)]
        null.append(np.mean(draws))
        sizes.append(len(grp))
    real, null = np.array(real), np.array(null)
    print(f"  purity: real median {np.median(real):.1%}, null median {np.median(null):.1%}, "
          f"{len(real)} communities >=8 cards")

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    bins = np.linspace(0, 1, 26)
    ax.hist(null, bins=bins, color=NEUTRAL, alpha=0.75, label=f"same-size random draws (median {np.median(null):.0%})")
    ax.hist(real, bins=bins, color=KEEP_C, alpha=0.8, label=f"our communities (median {np.median(real):.0%})")
    ax.axvline(np.median(null), color=NEUTRAL, ls="--", lw=1.2)
    ax.axvline(np.median(real), color=KEEP_C, ls="--", lw=1.2)
    ax.set_xlabel("share of the community sharing one color identity")
    ax.set_ylabel("number of communities")
    ax.set_title("Communities are far more color-consistent than chance, but color is not the whole story")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "fig7_color_purity.png"); plt.close(fig)
    print("fig7_color_purity.png")


def fig_seeds():
    """Seed-stability chart. **Not used in the final writeup** — the numbers are quoted in the text
    instead (see `seed_stability.py`)."""
    d = json.loads((SP / "seed_stability.json").read_text())
    real_ari = np.mean([r["ari"] for r in d["real"]]); null_ari = np.mean([r["ari"] for r in d["null"]])
    real_tm = np.mean([r["topic_match"] for r in d["real"]]); null_tm = np.mean([r["topic_match"] for r in d["null"]])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(2); wdt = 0.36
    ax.bar(x - wdt/2, [real_ari, real_tm], wdt, color=KEEP_C, label="our graph")
    ax.bar(x + wdt/2, [null_ari, null_tm], wdt, color=NEUTRAL, label="same graph, connections rewired at random")
    for xi, (a, b) in enumerate([(real_ari, null_ari), (real_tm, null_tm)]):
        ax.text(xi - wdt/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8.5, color=KEEP_C)
        ax.text(xi + wdt/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8.5, color="#6b727a")
    ax.set_xticks(x); ax.set_xticklabels(["agreement on which community\neach card lands in (ARI)",
                                          "similarity of the communities\nthemselves (cosine)"])
    ax.set_ylim(0, 1.0); ax.set_ylabel("agreement between runs at different seeds")
    ax.set_title("The communities survive a change of random seed; a structureless graph does not")
    ax.legend(loc="upper right", fontsize=8); ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig8_seed_stability.png"); plt.close(fig)
    print("fig8_seed_stability.png")


def fig_topics_per_card(bin_size=10):
    """**Figure 13.** Packages per card against EDHREC popularity rank, in bins of `bin_size` cards."""
    import sqlite3
    from edhcut.config import CONFIG
    con = sqlite3.connect(CONFIG.paths.db_path)
    rank = dict(con.execute("SELECT oracle_id, edhrec_rank FROM cards WHERE edhrec_rank IS NOT NULL"))
    df = cards.copy()
    df["n"] = floored.sum(axis=1)
    df["rank"] = df["oracle_id"].map(rank)
    a = df[(df["n"] > 0) & df["rank"].notna() & (df["rank"] <= RANK_CAP)]
    a = a.sort_values("rank").reset_index(drop=True)
    nb = len(a) // bin_size
    x = [a["rank"].iloc[k * bin_size:(k + 1) * bin_size].mean() for k in range(nb)]
    y = [a["n"].iloc[k * bin_size:(k + 1) * bin_size].mean() for k in range(nb)]

    left = [a["rank"].iloc[k * bin_size] for k in range(nb)]
    width = [(a["rank"].iloc[min((k + 1) * bin_size, len(a) - 1)] - a["rank"].iloc[k * bin_size])
             for k in range(nb)]

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.bar(left, y, width=width, align="edge", color=KEEP_C, alpha=0.9,
           linewidth=0, label=f"each bar = {bin_size} cards")
    ax.axhline(1.0, color=DROP_C, ls="--", lw=1.3, zorder=3)
    ax.text(max(x) * 0.985, 1.0, "a hard partition can only ever say 1", fontsize=8,
            color=DROP_C, ha="right", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92))
    ax.set_xlabel("EDHREC rank  (1 = the most played card in the format)")
    ax.set_ylabel("packages the card belongs to")
    ax.set_xlim(0, max(x) * 1.01)
    ax.set_title("The more popular a card, the more packages it belongs to")
    ax.grid(axis="x", visible=False); ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "fig9_packages_by_popularity.png"); plt.close(fig)
    q = a["n"].groupby(pd.qcut(a["rank"], 5, labels=False)).mean()
    print(f"fig9_packages_by_popularity.png  {nb} bins; mean packages by rank quintile: "
          + ", ".join(f"{v:.2f}" for v in q))


def fig_graph():
    """**Figure 9.** Force-directed layout of the largest communities — the main visual result of
    Question 2. Distance on the page is synergy, because edge weight is what pulls cards together."""
    assigned = cards[cards["community"] >= 0]
    named = [c for c in assigned["community"].value_counts().index
             if names.get(c) and "generic" not in names[c]]
    top = named[:14]
    sel = np.flatnonzero(cards["community"].isin(top).values)
    sub = S[sel][:, sel].tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(len(sel)))
    G.add_weighted_edges_from(zip(sub.row, sub.col, sub.data))
    print(f"  layout on {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges ...", flush=True)
    pos = nx.spring_layout(G, weight="weight", seed=1, iterations=120, k=0.035)
    xy = np.array([pos[i] for i in range(len(sel))])
    comm = cards["community"].values[sel]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    centroids = []
    for j, cid in enumerate(top):
        m = comm == cid
        ax.scatter(xy[m, 0], xy[m, 1], s=11, color=cmap(j % 20), alpha=0.85, edgecolors="none")
        centroids.append([np.median(xy[m, 0]), np.median(xy[m, 1]),
                          DISPLAY.get(cid, names.get(cid, "?").split("/")[0]), cmap(j % 20),
                          int(m.sum())])

    lo, hi = np.percentile(xy, [1.5, 98.5], axis=0)
    pad = (hi - lo) * 0.06
    xlim = (lo[0] - pad[0], hi[0] + pad[0]); ylim = (lo[1] - pad[1], hi[1] + pad[1])
    span = (xlim[1] - xlim[0], ylim[1] - ylim[0])

    # Place biggest first; shove later labels off any already placed, in 2D, and keep them in frame.
    centroids.sort(key=lambda c: -c[4])
    placed = []
    for c in centroids:
        xsep = span[0] * 0.075 * (len(c[2]) / 14 + 0.5)
        ysep = span[1] * 0.045
        for _ in range(140):
            clash = [p for p in placed
                     if abs(p[0] - c[0]) < xsep and abs(p[1] - c[1]) < ysep]
            if not clash:
                break
            p = clash[0]
            dy = ysep if c[1] >= p[1] else -ysep
            c[1] = p[1] + dy * 1.02
            if not (ylim[0] < c[1] < ylim[1]):      # ran out of vertical room -- step sideways
                c[1] = p[1]
                c[0] = p[0] + (xsep if c[0] >= p[0] else -xsep) * 1.02
        c[0] = float(np.clip(c[0], xlim[0] + span[0] * 0.06, xlim[1] - span[0] * 0.06))
        c[1] = float(np.clip(c[1], ylim[0] + span[1] * 0.03, ylim[1] - span[1] * 0.03))
        placed.append(c)
        ax.annotate(c[2], (c[0], c[1]), fontsize=8.2, weight="bold", ha="center", color="#111",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=c[3], lw=0.9, alpha=0.93),
                    zorder=5)
    print(f"  labelled {len(placed)} of {len(top)} communities")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("The 14 largest named packages, placed by pulling connected cards together")
    fig.savefig(OUT / "fig10_community_graph.png"); plt.close(fig)
    print("fig10_community_graph.png")


if __name__ == "__main__":
    fig_purity(); fig_seeds(); fig_topics_per_card(); fig_graph()
