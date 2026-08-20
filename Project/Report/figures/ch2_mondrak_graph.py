"""One card's edges drawn across three packages, everything else greyed out."""
import sys, pickle
from pathlib import Path
import numpy as np, scipy.sparse as sparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
OUT = Path(r"C:/Aviv/University/Semester 8/Data Science/Homework - Group/Project/Report/Images")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "font.size": 9, "figure.facecolor": "white", "axes.facecolor": "white"})

D = pickle.load(open(SP / "ch2.pkl", "rb"))
cards, share, floored, S, names = D["cards"], D["share"], D["floored"], D["S"], D["names"]

QUERY = "Mondrak, Glory Dominus"
COLORS = ["#1b7f5e", "#0d7fa8", "#b8860b"]
BLURB = ["token payoffs", "powerful staples", "Phyrexian typal"]
GREY, GREY_E = "#c9ced4", "#e2e6ea"

i = cards.index[cards["name"] == QUERY][0]
tops = [t for t in np.argsort(-share[i])[:5] if floored[i, t]][:3]
comm = cards["community"].values

# Mondrak's three packages, plus context: the other communities its neighbours sit in
B = (S > 0).tocsr()
nbrs = B[i].indices
context = [c for c in np.unique(comm[nbrs]) if c >= 0 and c not in tops]
sel_comms = list(tops) + context[:9]
sel = np.flatnonzero(np.isin(comm, sel_comms))
if i not in sel:
    sel = np.append(sel, i)
pos_of = {g: k for k, g in enumerate(sel)}

sub = S[sel][:, sel].tocoo()
G = nx.Graph(); G.add_nodes_from(range(len(sel)))
G.add_weighted_edges_from(zip(sub.row, sub.col, sub.data))
print(f"layout on {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges ...", flush=True)
pos = nx.spring_layout(G, weight="weight", seed=3, iterations=140, k=0.05)
xy = np.array([pos[k] for k in range(len(sel))])

fig, ax = plt.subplots(figsize=(8.4, 6.6))
mine = pos_of[i]

# context first, in grey
other = np.array([k for k, g in enumerate(sel) if comm[g] not in tops and g != i])
ax.scatter(xy[other, 0], xy[other, 1], s=9, c=GREY, alpha=0.55, edgecolors="none", zorder=1)

# Mondrak's edges, drawn to wherever they land
for j in B[i].indices:
    if j not in pos_of:
        continue
    k = pos_of[j]
    c = comm[j]
    col = COLORS[tops.index(c)] if c in tops else GREY_E
    ax.plot([xy[mine, 0], xy[k, 0]], [xy[mine, 1], xy[k, 1]],
            color=col, lw=0.8, alpha=0.55 if c in tops else 0.3, zorder=2)

for n_, t in enumerate(tops):
    m = np.array([k for k, g in enumerate(sel) if comm[g] == t])
    ax.scatter(xy[m, 0], xy[m, 1], s=17, c=COLORS[n_], alpha=0.9, edgecolors="none", zorder=3,
               label=f"{share[i, t]:.0%}  {BLURB[n_]}")
    cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
    # push the label away from the query card so it never sits on top of it
    dx, dy = cx - xy[mine, 0], cy - xy[mine, 1]
    norm = max(np.hypot(dx, dy), 1e-9)
    cx, cy = cx + dx / norm * 0.09, cy + dy / norm * 0.09
    ax.annotate(BLURB[n_], (cx, cy), fontsize=9, weight="bold", ha="center", color="#111",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS[n_], lw=1.1, alpha=0.93),
                zorder=6)

ax.scatter([xy[mine, 0]], [xy[mine, 1]], s=170, c="#c2452d", edgecolors="white",
           linewidths=1.6, zorder=7, marker="*")
ax.annotate(QUERY.split(",")[0], (xy[mine, 0], xy[mine, 1]), textcoords="offset points",
            xytext=(0, 14), ha="center", fontsize=10, weight="bold", color="#c2452d", zorder=8)

ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s_ in ax.spines.values():
    s_.set_visible(False)
lo, hi = np.percentile(xy, [1, 99], axis=0); pad = (hi - lo) * 0.07
ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
ax.set_title(f"{QUERY.split(',')[0]}'s connections reach three packages that barely touch each other",
             fontsize=11)
ax.legend(loc="lower left", fontsize=8.5, frameon=False)
fig.savefig(OUT / "fig11_multi_membership_graph.png"); plt.close(fig)
print("wrote fig11_multi_membership_graph.png")
