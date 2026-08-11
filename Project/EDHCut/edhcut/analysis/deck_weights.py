"""Design weights for the metagame sample (plan `EDHCut_PLAN.md` §7 task 5.7): inverse-
probability weighting so a deck's contribution to a *weighted* analysis reflects its
commander('s) true share of the real Commander metagame (per EDHREC), not however many decks
task 5.7 happened to harvest for it.

**Computed at analysis time, never baked into the harvest** -- `decks`/`deck_cards` store no
weight column. Different analyses want different weighting: `communities.py`'s graph wants
near-uniform (diversity matters more than representativeness for finding packages) --
`compute_near_uniform_weights`, below -- while play-rate/staple-detection wants the
metagame-accurate weighting `compute_deck_weights` provides. Each analysis opts in explicitly
rather than one weight being forced on everything.

**The formula**: `w_s = true_share_s / sample_share_s`, both computed only over the commander
slots present in `meta_commanders` -- the only slots with a real EDHREC deck count to compare
against.

- `true_share_s` = that slot's `edhrec_num_decks` as a fraction of the *sum* of
  `edhrec_num_decks` across every `meta_commanders` row -- "what share of the real (measured)
  metagame this commander actually represents."
- `sample_share_s` = how many decks task 5.7 actually harvested for that slot, as a fraction of
  the total harvested across every slot that has *any* harvested decks -- "what share of our
  corpus this commander actually represents." Not restricted by cohort: a roster commander's
  full roster-cohort corpus counts here (that's the whole point -- Krenko's 2,000+ roster decks
  are recognized as heavily oversampled relative to its true metagame share, without needing to
  separately re-harvest it under `meta_sample`; see `edhcut.ingest.archidekt.run_meta_sample`'s
  own roster-exclusion for the harvest-time half of this decision).

A slot with no harvested decks at all (`sample_share_s` undefined, division by zero) is left
out of the returned mapping entirely -- same treatment as a slot outside `meta_commanders`
altogether (the roster's own Yoshimaru+Bruse Tarl and Orysa, both below the 2,300-deck
threshold, so genuinely have no measured metagame share to weight by). **Callers must default a
missing `slot_key` to weight `1.0`** (neutral/unweighted) -- the same "unresolved defaults to
neutral" convention `edhcut.analysis.cooccurrence`'s own precon-card weighting already uses, not
a new one invented here.
"""

from __future__ import annotations

import argparse
import sqlite3

from edhcut.config import CONFIG
from edhcut.db import connect

DEFAULT_WEIGHT = 1.0


def compute_near_uniform_weights(conn: sqlite3.Connection) -> dict[str, float]:
    """`{slot_key: weight}` for every commander slot with at least one deck in `decks`
    (roster and meta_sample alike -- unlike `compute_deck_weights`, not restricted to
    `meta_commanders` rows, since roster slots like Yoshimaru+Bruse Tarl (300 decks) or Orysa
    (29) are also oversampled relative to a typical slot even though they fall below the
    2,300-deck `meta_commanders` threshold).

    `weight_s = median(n_s) / n_s`: flattens every slot's *total* weighted contribution to
    roughly the same value (a slot at the median deck count gets weight ~1.0; Krenko's 2,007
    decks each get weight ~median/2007). This is deliberately **not** `compute_deck_weights`'s
    `true_share/sample_share` formula -- that one restores each commander's *real-world*
    popularity share (still large for a genuinely popular commander like Krenko, just no longer
    inflated further by oversampling), which is what play-rate/staple analyses want. Community
    detection wants the opposite: no commander's real popularity should give it outsized pull
    over which card packages get discovered, since diversity of sampled packages matters more
    than matching the true metagame distribution -- see module docstring."""
    counts = dict(conn.execute("SELECT slot_key, COUNT(*) FROM decks GROUP BY slot_key"))
    if not counts:
        return {}
    values = sorted(counts.values())
    mid = len(values) // 2
    median_n = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return {slot_key: median_n / n for slot_key, n in counts.items()}


def compute_deck_weights(conn: sqlite3.Connection) -> dict[str, float]:
    """`{slot_key: weight}` for every `meta_commanders` slot that has at least one harvested
    deck. See module docstring for the formula and why a missing slot_key means "no weight
    information," not "weight zero" -- callers should fall back to `DEFAULT_WEIGHT`."""
    true_counts = dict(conn.execute("SELECT slot_key, edhrec_num_decks FROM meta_commanders"))
    if not true_counts:
        return {}
    total_true = sum(true_counts.values())

    placeholders = ",".join("?" * len(true_counts))
    sample_counts = dict(conn.execute(
        f"SELECT slot_key, COUNT(*) FROM decks WHERE slot_key IN ({placeholders}) GROUP BY slot_key",
        list(true_counts),
    ))
    if not sample_counts:
        return {}
    total_sample = sum(sample_counts.values())

    return {
        slot_key: (true_counts[slot_key] / total_true) / (sample_n / total_sample)
        for slot_key, sample_n in sample_counts.items()
    }


def deck_weight(weights: dict[str, float], slot_key: str | None, *, default: float = DEFAULT_WEIGHT) -> float:
    """`weights[slot_key]` if known, else `default` -- a thin lookup so call sites read as
    `deck_weight(weights, deck.slot_key)` rather than repeating the `.get(..., 1.0)` idiom."""
    if slot_key is None:
        return default
    return weights.get(slot_key, default)


def _cmd_show(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        weights = compute_deck_weights(conn)
        names = dict(conn.execute(
            f"SELECT slot_key, name FROM meta_commanders WHERE slot_key IN "
            f"({','.join('?' * len(weights))})", list(weights)
        )) if weights else {}

    if not weights:
        print("No deck weights available -- has edhcut.ingest.edhrec_commanders been run, and "
              "does at least one meta_commanders slot have harvested decks?")
        return

    ordered = sorted(weights.items(), key=lambda kv: kv[1])
    print(f"{len(weights)} slot(s) with a computed design weight (most-oversampled first):")
    if len(ordered) <= 2 * args.top:
        shown = ordered
    else:
        shown = ordered[: args.top] + [(None, None)] + ordered[-args.top:]
    for slot_key, weight in shown:
        if slot_key is None:
            print("  ...")
            continue
        print(f"  {weight:8.4f}  {names.get(slot_key, slot_key)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10, help="How many slots to show at each end of the sorted list.")
    parser.set_defaults(func=_cmd_show)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
