"""Ad hoc visualization for the co-occurrence/PMI matrices task 6.1
(`edhcut.analysis.cooccurrence`) builds into `data/kb/dev/`. Not part of the tested
package/pipeline — a quick way to eyeball one card's local association neighborhood as a
heatmap, rather than reading raw numbers off the `top` CLI.

Usage:
    python scripts/plot_pmi_heatmap.py "Skirk Prospector" --slot krenko -k 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from edhcut.analysis.cooccurrence import KB_DEV_DIR, load_card_index, load_pmi, top_associated
from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.ingest.scryfall import normalize_name


def _resolve_oracle_id(card_name: str) -> str:
    with connect(CONFIG.paths.db_path) as conn:
        row = conn.execute(
            "SELECT oracle_id FROM card_names WHERE name_normalized = ?",
            (normalize_name(card_name),),
        ).fetchone()
    if row is None:
        raise SystemExit(f"Card {card_name!r} not found in card_names.")
    return row[0]


def plot_pmi_heatmap(
    card_name: str, *, slot: str = "global", k: int = 20, out_path: Path
) -> None:
    """Heatmap of one card and its `k` strongest PMI associates, so a single card's neighbourhood
    can be eyeballed. Exploratory output — not a report figure."""
    seed_id = _resolve_oracle_id(card_name)
    card_index = load_card_index()
    pmi = load_pmi(slot)

    neighbors = top_associated(pmi, card_index, seed_id, k=k)
    if neighbors.empty:
        raise SystemExit(f"No PMI associations for {card_name!r} in scope {slot!r}.")

    # Seed card first, then its top-k neighbors in descending-PMI order (matches the `top` CLI).
    node_ids = [seed_id, *neighbors["oracle_id"]]
    node_names = [card_name, *neighbors["name"]]
    row_lookup = dict(zip(card_index["oracle_id"], card_index["row"]))
    rows = np.array([row_lookup[oid] for oid in node_ids])

    sub = pmi[rows, :][:, rows].toarray()

    n = len(node_ids)
    fig, ax = plt.subplots(figsize=(0.45 * n + 2, 0.45 * n + 2))
    vmax = max(abs(sub.min()), abs(sub.max()), 1e-6)
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(node_names, rotation=90, fontsize=8)
    ax.set_yticklabels(node_names, fontsize=8)
    ax.set_title(f"PMI neighborhood of {card_name!r} ({slot})", fontsize=11)
    # Seed's own row/column labels in bold so it's easy to find in the grid.
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_fontweight("bold")
    fig.colorbar(im, ax=ax, label="PMI (discounted)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", help="Card name")
    parser.add_argument("--slot", default="global", help="Scope label (default: global)")
    parser.add_argument("-k", type=int, default=20, help="Number of neighbors to plot")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    out_path = args.out or (KB_DEV_DIR / f"pmi_heatmap_{args.card.split(',')[0].strip().lower().replace(' ', '_')}.png")
    plot_pmi_heatmap(args.card, slot=args.slot, k=args.k, out_path=out_path)


if __name__ == "__main__":
    main()
