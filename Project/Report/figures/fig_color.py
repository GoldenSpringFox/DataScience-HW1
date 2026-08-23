"""The colour-correction chart — how much the colour-conditioned null changes each pair's score.
NOT used as a figure in the final writeup: the effect is quoted as numbers in Question 1 instead.
Kept because those numbers come from here.

Earlier description: Colour-conditioned null model: validation pairs, pooled vs corrected.
"""
import sys, sqlite3
from pathlib import Path
import numpy as np, scipy.sparse as sparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb
from figs import OUT, KEEP_C, DROP_C, NEUTRAL

# category assigned by hand from the cards themselves, never from the result
COLOURLESS, SYNERGY, GENERIC = "colourless", "synergy", "generic"
PAIRS = [
    ("Goblin Warchief", "Skirk Prospector", SYNERGY),
    ("Cultivate", "Kodama's Reach", SYNERGY),
    ("Heronblade Elite", "Call the Coppercoats", SYNERGY),
    ("Mesa Enchantress", "All That Glitters", SYNERGY),
    ("Sanguine Bond", "Exquisite Blood", SYNERGY),
    ("Ashnod's Altar", "Zulaport Cutthroat", COLOURLESS),
    ("Sol Ring", "Arcane Signet", COLOURLESS),
    ("Heronblade Elite", "Horn of Gondor", COLOURLESS),
    ("Teferi's Protection", "Smothering Tithe", GENERIC),
    ("Swords to Plowshares", "Loran of the Third Path", GENERIC),
    ("Swords to Plowshares", "Felidar Retreat", GENERIC),
    ("Beast Within", "Snakeskin Veil", GENERIC),
    ("Assassin's Trophy", "Lathril, Blade of the Elves", GENERIC),
]
STYLE = {COLOURLESS: (NEUTRAL, "colourless card involved - must not move"),
         SYNERGY: (KEEP_C, "real synergy - survives"),
         GENERIC: (DROP_C, "generic same-colour pairing - collapses")}

ci, _, _, n2r, r2n, dc = kb.load()
pooled = sparse.load_npz(kb.KB / "tscore_global.npz").tocsr()
colour = sparse.load_npz(kb.KB / "tscore_color_global.npz").tocsr()

rows = []
for a, b, cat in PAIRS:
    i, j = n2r[a], n2r[b]
    p, q = float(pooled[i, j]), float(colour[i, j])
    rows.append((f"{a} + {b}", cat, p, q, (q - p) / p * 100))
rows.sort(key=lambda r: r[4])

fig, ax = plt.subplots(figsize=(7.6, 4.6))
y = np.arange(len(rows))
for k, (label, cat, p, q, chg) in enumerate(rows):
    col = STYLE[cat][0]
    ax.barh(k, chg, color=col, alpha=0.85, height=0.68, zorder=2)
    off = -4 if chg < 0 else 4
    ha = "right" if chg < 0 else "left"
    ax.text(chg + off, k, f"{chg:+.0f}%   (t {p:.1f} -> {q:.1f})", va="center", ha=ha,
            fontsize=6.8, color=col)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
ax.axvline(0, color="#333", lw=1)
ax.set_xlim(-115, 55)
ax.set_xlabel("change in t-score after conditioning the null model on colour legality")
ax.set_title("Correcting for colour legality only fires where legality is the explanation")
ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Line2D([], [], color=c, lw=6, label=l) for c, l in STYLE.values()],
          loc="upper left", fontsize=7.5)
fig.savefig(OUT / "fig7_color_correction.png")
print("wrote fig7_color_correction.png")
for label, cat, p, q, chg in rows:
    print(f"  {cat:10s} {chg:+8.3f}%  {label}")
