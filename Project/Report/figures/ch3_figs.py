"""Chapter 3 figures: substitutes (text) vs complements (co-occurrence)."""
import sys, sqlite3, itertools
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb
from edhcut.config import CONFIG

OUT = Path(r"C:/Aviv/University/Semester 8/Data Science/Homework - Group/Project/Report/Images")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})
CAT_C, PKG_C = "#0d7fa8", "#1b7f5e"

# Groups taken from Scryfall Tagger, which never entered any of our models.
CATEGORIES = {           # cards that do the same job -- should look alike, needn't be played together
    "sweeper": "boardwipes", "counterspell": "counterspells", "tutor": "tutors",
    "ramp": "ramp spells", "spot-removal": "spot removal",
}
PACKAGES = {             # cards that need each other -- needn't look alike at all
    "landfall": "landfall", "typal-elf": "elves", "typal-goblin": "goblins",
    "synergy-equipment": "equipment", "lifegain-matters": "lifegain", "typal-zombie": "zombies",
}
MAX_PAIRS, SEED = 600, 0
CAP = 12.0   # figure 11 y-axis cap; elves and zombies run past it and are labelled with their real value


def load():
    ci, tscore, lift, n2r, r2n, dc = kb.load()
    emb = pd.read_parquet("data/kb/dev/embeddings.parquet")
    text_cols = [c for c in emb.columns if c.startswith(("tfidf_", "types_", "struct_"))]
    emb = emb[emb["oracle_id"].isin(set(ci["oracle_id"]))]
    vec = np.array(emb.set_index("oracle_id")[text_cols].to_numpy(dtype=np.float32), copy=True)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    oid_row = {o: k for k, o in enumerate(emb["oracle_id"])}
    con = sqlite3.connect(CONFIG.paths.db_path)
    tags = pd.read_sql("SELECT oracle_id, tag FROM card_tags", con)
    oid_of = dict(zip(ci["name"], ci["oracle_id"]))
    return ci, lift, n2r, r2n, dc, vec, oid_row, tags, oid_of


TOP_N = 14   # the most-played members of each group


def sample_pairs(members, rng):
    """All pairs among a group's most-played members.

    Sampling at random from a 4,648-card tag is useless: almost every pair is two obscure cards
    that have never shared a deck, so the co-occurrence is zero for reasons that have nothing to
    do with whether the group is a package or a category.
    """
    if len(members) < 2:
        return []
    return list(itertools.combinations(members[:TOP_N], 2))


def collect():
    ci, lift, n2r, r2n, dc, vec, oid_row, tags, oid_of = load()
    row_of_oid = dict(zip(ci["oracle_id"], ci["row"]))
    rng = np.random.default_rng(SEED)
    L = lift.tocsr()
    rows = []
    excluded = {oid_of[n] for n in kb.STAPLES + kb.BASICS if n in oid_of}
    play = dict(zip(ci["oracle_id"], ci["deck_count"]))
    for kind, groups in (("category", CATEGORIES), ("package", PACKAGES)):
        for tag, label in groups.items():
            oids = [o for o in tags.loc[tags["tag"] == tag, "oracle_id"].unique()
                    if o in row_of_oid and o in oid_row and o not in excluded]
            oids.sort(key=lambda o: -play.get(o, 0))
            for a, b in sample_pairs(oids, rng):
                ra, rb = row_of_oid[a], row_of_oid[b]
                rows.append(dict(kind=kind, group=label,
                                 text=float(vec[oid_row[a]] @ vec[oid_row[b]]),
                                 lift=float(L[ra, rb])))
    df = pd.DataFrame(rows)
    print(df.groupby(["kind", "group"]).agg(pairs=("lift", "size"),
                                            median_text=("text", "median"),
                                            median_lift=("lift", "median")).to_string())
    return df


def fig_scatter(df):
    """One point per group: does it read alike, and is it played together?"""
    g = df.groupby(["kind", "group"]).agg(text=("text", "median"), lift=("lift", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhspan(0, 1.0, color="#c2452d", alpha=0.06)
    ax.axhline(1.0, color="#c2452d", ls="--", lw=1.2, zorder=2)
    for kind, col, lab, mk in (("category", CAT_C, "same job (a functional category)", "o"),
                               ("package", PKG_C, "need each other (a synergy package)", "D")):
        d = g[g["kind"] == kind]
        # elves and zombies run to 40x and 70x; clipping them keeps the axis linear and lets the
        # rest of the picture stay readable, with their real values written on the marker.
        shown = np.array(d["lift"].clip(upper=CAP).to_numpy(dtype=float), copy=True)
        # clipped points would otherwise land on the same spot; stack them instead
        over = np.flatnonzero(d["lift"].to_numpy() > CAP)
        for j, k in enumerate(over):
            shown[k] = CAP - j * 1.5
        ax.scatter(d["text"], shown, s=110, c=col, marker=mk, edgecolors="white",
                   linewidths=1.4, zorder=4, label=lab)
        for j, ((_, r), yv) in enumerate(zip(d.iterrows(), shown)):
            off = r["lift"] > CAP
            if off:
                # clipped points share an x, so label them sideways in alternating directions
                side = 1 if list(over).index(j) % 2 == 0 else -1
                ax.annotate(f"{r['group']}  ({r['lift']:.0f}x)", (r["text"], yv),
                            textcoords="offset points", xytext=(14 * side, 0),
                            ha="left" if side > 0 else "right", va="center",
                            fontsize=8.6, color=col, weight="bold")
                ax.annotate("", (r["text"], yv + 0.55), (r["text"], yv - 0.6),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
                continue
            near = ((g["text"] - r["text"]).abs() < 0.06) & (g["group"] != r["group"]) \
                   & ((g["lift"] / r["lift"]).between(0.6, 1.7))
            dy = -20 if near.any() and r["text"] > g.loc[near, "text"].min() else 12
            ax.annotate(r["group"], (r["text"], yv), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8.6, color=col, weight="bold")
    ax.text(0.99, 1.35, "played together no more than chance", fontsize=8, color="#c2452d",
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 0.95); ax.set_ylim(0, CAP + 1.5)
    ax.set_xlabel("how alike the group's cards read  (median text similarity)")
    ax.set_ylabel("how much more than chance they are played together  (median lift)")
    ax.set_title("Reading alike says nothing about belonging together:\n"
                 "the two kinds of group separate vertically and not at all horizontally")
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9)
    leg.get_frame().set_facecolor("#f2f4f6"); leg.get_frame().set_edgecolor("#d5dade")
    fig.savefig(OUT / "fig11_substitutes_vs_complements.png"); plt.close(fig)
    print("fig11_substitutes_vs_complements.png")


def fig_groups(df):
    order = (df.groupby(["kind", "group"])["lift"].median().reset_index()
               .sort_values(["kind", "lift"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rng = np.random.default_rng(1)
    ticks, labels = [], []
    for y, (_, r) in enumerate(order.iterrows()):
        d = df[(df["kind"] == r["kind"]) & (df["group"] == r["group"])]
        col = CAT_C if r["kind"] == "category" else PKG_C
        ax.scatter(d["lift"], np.full(len(d), y) + rng.uniform(-.16, .16, len(d)),
                   s=8, c=col, alpha=0.25, edgecolors="none")
        ax.scatter([d["lift"].median()], [y], s=95, c=col, edgecolors="white", linewidths=1.6,
                   marker="D", zorder=5)
        ax.text(d["lift"].median() + 1.8, y + 0.28, f"{d['lift'].median():.1f}x",
                fontsize=8, color=col, weight="bold")
        ticks.append(y); labels.append(r["group"])
    ax.axvline(1.0, color="#c2452d", ls="--", lw=1.2)
    ax.set_xlim(-2, 82)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_xlabel("how much more than chance two cards in the group are played together (lift)")
    ax.set_title("Cards that do the same job are played together at chance; cards that need each other, far above it")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="D", ls="", color=CAT_C, label="functional category"),
                       plt.Line2D([], [], marker="D", ls="", color=PKG_C, label="synergy package")],
              loc="lower right", fontsize=8.5)
    fig.savefig(OUT / "fig12_group_lift.png"); plt.close(fig)
    print("fig12_group_lift.png")


def fig_role_agreement():
    r = pd.read_parquet("data/kb/dev/roles.parquet")
    both = r[r["tagger_primary"].notna() & r["heuristic_primary"].notna()].copy()
    both["ok"] = both["tagger_primary"] == both["heuristic_primary"]
    g = both.groupby("tagger_primary").agg(n=("ok", "size"), agree=("ok", "mean"))
    g = g[g["n"] >= 100].sort_values("agree")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(g))
    ax.barh(y, g["agree"], color=PKG_C, alpha=0.88, height=0.68)
    for yi, (idx, row) in enumerate(g.iterrows()):
        ax.text(row["agree"] + 0.012, yi, f"{row['agree']:.0%}  ({int(row['n']):,} cards)",
                va="center", fontsize=7.8, color="#333")
    overall = both["ok"].mean()
    ax.axvline(overall, color="#c2452d", ls="--", lw=1.3)
    ax.text(overall + 0.01, len(g) - 0.4, f"overall {overall:.1%}", fontsize=8.5, color="#c2452d")
    ax.set_yticks(y); ax.set_yticklabels(g.index)
    ax.set_xlim(0, 1.25); ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("share of cards where the two layers name the same role")
    ax.set_title("Crowd tags and our own text rules agree on three quarters of cards, and disagree by role")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig13_role_agreement.png"); plt.close(fig)
    print(f"fig13_role_agreement.png  (overall {overall:.1%} over {len(both):,} cards)")


if __name__ == "__main__":
    df = collect()
    df.to_parquet(SP / "ch3_pairs.parquet")
    fig_scatter(df); fig_groups(df); fig_role_agreement()
