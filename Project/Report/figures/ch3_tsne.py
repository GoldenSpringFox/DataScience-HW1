"""The text space in 2D, with one card's text neighbours and its deck neighbours marked.

The point: in a space built only from what cards say, the cards Skullclamp is *played with* are
scattered all over, while the cards that merely resemble it sit right next to it.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb

OUT = Path(__file__).resolve().parents[1] / "Images"   # Project/Report/Images
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "font.size": 9, "figure.facecolor": "white", "axes.facecolor": "white"})
TEXT_C, CO_C, GREY = "#0d7fa8", "#1b7f5e", "#ccd2d8"
QUERY, N_NB, SAMPLE = "Skullclamp", 8, 6000

ci, tscore, lift, n2r, r2n, dc = kb.load()
emb = pd.read_parquet(kb.KB / "embeddings.parquet")
cols = [c for c in emb.columns if c.startswith(("tfidf_", "types_", "struct_"))]
emb = emb[emb["oracle_id"].isin(set(ci["oracle_id"]))].reset_index(drop=True)
vec = np.array(emb[cols].to_numpy(dtype=np.float32), copy=True)
vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)

oid_of = dict(zip(ci["name"], ci["oracle_id"]))
erow = {o: k for k, o in enumerate(emb["oracle_id"])}
name_e = emb["name"].to_numpy()
is_land_e = emb["is_land"].to_numpy()
play_e = emb["oracle_id"].map(dict(zip(ci["oracle_id"], ci["deck_count"]))).fillna(0).to_numpy()

q = erow[oid_of[QUERY]]
sims = vec @ vec[q]
text_nb = [k for k in np.argsort(-sims) if k != q and not is_land_e[k]][:N_NB]

r = n2r[QUERY]
tr = tscore.getrow(r).toarray().ravel(); lr = lift.getrow(r).toarray().ravel()
comb = tr * np.log1p(np.maximum(lr, 0))
land_row = dict(zip(ci["row"], ci["is_land"]))
co_names = [r2n[i] for i in np.argsort(-comb) if i != r and not land_row.get(i)][:N_NB]
co_nb = [erow[oid_of[n]] for n in co_names if n in oid_of and oid_of[n] in erow]

# a popularity-weighted backdrop, plus every card we need to show
base = list(np.argsort(-play_e)[:SAMPLE])
sel = sorted(set(base) | {q} | set(text_nb) | set(co_nb))
sub = vec[sel]
print(f"t-SNE on {len(sel):,} cards x {sub.shape[1]} dims ...", flush=True)
t0 = time.perf_counter()
xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=0,
          max_iter=750).fit_transform(sub)
print(f"  done in {time.perf_counter()-t0:.0f}s")
pos = {g: k for k, g in enumerate(sel)}

fig, ax = plt.subplots(figsize=(8.0, 6.4))
ax.scatter(xy[:, 0], xy[:, 1], s=5, c=GREY, alpha=0.55, edgecolors="none", zorder=1)
for group, col, lab, mk in ((text_nb, TEXT_C, f"its {N_NB} nearest by text", "o"),
                            (co_nb, CO_C, f"its {N_NB} nearest by co-occurrence", "D")):
    k = [pos[g] for g in group]
    ax.scatter(xy[k, 0], xy[k, 1], s=62, c=col, marker=mk, edgecolors="white",
               linewidths=1.2, zorder=4, label=lab)
qk = pos[q]
ax.scatter([xy[qk, 0]], [xy[qk, 1]], s=230, c="#c2452d", marker="*", edgecolors="white",
           linewidths=1.6, zorder=6)
ax.annotate(QUERY, (xy[qk, 0], xy[qk, 1]), textcoords="offset points", xytext=(0, 15),
            ha="center", fontsize=10.5, weight="bold", color="#c2452d", zorder=7)
placed = []
for g in co_nb:
    k = pos[g]
    dy = -14
    while any(abs(px - xy[k, 0]) < 8 and abs(py - (xy[k, 1] + dy)) < 3 for px, py in placed):
        dy -= 11
    placed.append((xy[k, 0], xy[k, 1] + dy))
    ax.annotate(name_e[g].split(",")[0], (xy[k, 0], xy[k, 1]), textcoords="offset points",
                xytext=(0, dy), ha="center", fontsize=7.4, color=CO_C, zorder=5)

# The text neighbours sit at cosine 0.97+ from the query, so on the full map they land underneath
# it. The inset is the honest way to show that: they are not missing, they are stacked.
ins = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
tk = [pos[g] for g in text_nb]
cx, cy = xy[qk, 0], xy[qk, 1]
span = max(np.abs(xy[tk] - [cx, cy]).max() * 1.9, 1.5)
ins.scatter(xy[:, 0], xy[:, 1], s=8, c=GREY, alpha=0.6, edgecolors="none")
ins.scatter(xy[tk, 0], xy[tk, 1], s=54, c=TEXT_C, edgecolors="white", linewidths=1.0, zorder=4)
ins.scatter([cx], [cy], s=200, c="#c2452d", marker="*", edgecolors="white", linewidths=1.4, zorder=5)
for i, g in enumerate(sorted(text_nb, key=lambda g: -xy[pos[g], 1])[:4]):
    k = pos[g]
    ins.annotate(name_e[g].split(" //")[0], (xy[k, 0], xy[k, 1]), textcoords="offset points",
                 xytext=(7, 9 - i * 13), fontsize=6.8, color=TEXT_C,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
ins.set_xlim(cx - span, cx + span); ins.set_ylim(cy - span, cy + span)
ins.set_xticks([]); ins.set_yticks([])
ins.set_title("zoomed in on Skullclamp", fontsize=8)
for s in ins.spines.values():
    s.set_color("#999"); s.set_linewidth(0.8)
ax.indicate_inset_zoom(ins, edgecolor="#999")

ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Laid out by what cards say, the cards Skullclamp is played with\n"
             "are scattered to the far corners", fontsize=11)
ax.legend(loc="upper left", fontsize=9, frameon=False)
fig.savefig(OUT / "fig13_text_space.png"); plt.close(fig)
print("wrote fig13_text_space.png")
print("  text nb:", ", ".join(name_e[g] for g in text_nb))
print("  co   nb:", ", ".join(name_e[g] for g in co_nb))
