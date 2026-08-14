"""Theme/tag ground-truth labels for scoring soft-clustering granularity and color bias (plan
`docs/plans/6.3b-communities-next-iteration.md` step 1) -- the "measuring stick" the color-bias
and granularity work in later steps is scored against, built and validated *before* any new
clustering so a metric bug can't be mistaken for (or mask) a clustering improvement.

Two label sources already exist in the DB, at very different grains:

- `card_tags` (source `tagger_bulk`): 4,183 per-card mechanical/thematic tags, 376,800 taggings
  -- fine-grained, genuinely per-card, but not curated to EDHREC's own theme vocabulary.
- `edhrec_themes`: EDHREC's own 401-row theme/typal taxonomy (270 `theme` + 131 `typal`) -- the
  *target* vocabulary this project actually wants topics to resolve against, but it's a name
  list with global popularity counts, not a per-card membership table (that would need scraping
  the 401 `/themes/<slug>` pages -- out of scope here, see the plan's own note on that).

`build_theme_labels` bridges them: every `card_tags` tag in a usable size range becomes a label
carrying its real per-card membership; where that tag also identifies an EDHREC theme (exact
slug match, the `typal` kind's `typal-<singular>` convention, or a small hand alias list for the
rest), the label is *renamed* to that theme's own name for readability, but the underlying
per-card membership -- and therefore every enrichment statistic computed against it -- is always
the tag's, never an inferred theme membership. This keeps the label vocabulary's size and content
fully reproducible from `card_tags` alone: verified against numbers hand-computed live in the
planning session (953 tags at 20-3,000 cards restricted to the saved k=40 NMF pool, 587/953
enriched, 74/953 resolved at tau=0.10, median topic color purity 66.7%) -- see this module's own
tests, which reproduce those exact figures.

**Two different questions, answered by two different metrics -- do not conflate them**:
`n_labels_covered` (a label is *enriched* in >=1 topic -- recall) is already near-saturated at
k=40 and isn't the number to optimize; `n_labels_resolved(tau)` (a label where some topic's
`min(share_of_topic, recall_of_label) >= tau` -- both dominates and is captured) is the real
granularity objective, since bundling (many enriched-but-unresolved themes crammed into one
topic), not absence, is what the baseline found."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from edhcut.config import CONFIG
from edhcut.db import connect

KB_DEV_DIR = CONFIG.paths.kb_dir / "dev"

MIN_TAG_CARDS = 20
MAX_TAG_CARDS = 3000
MIN_TOPIC_LABEL_CARDS = 5
MIN_LIFT = 3.0
FDR_Q = 0.01
RESOLUTION_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)
DEFAULT_TOP_N_PURITY = 30
DEFAULT_NULL_DRAWS = 2000

# Irregular English plurals seen in `edhrec_themes`' own `typal` rows that a plain suffix-strip
# singularizer gets wrong (e.g. "elves" -> naive "elve", not the real tag "typal-elf"). Not
# exhaustive -- extend as new typal misses turn up; anything not covered here just falls back to
# the plain suffix rule below, and stays an unmapped (but still usable, tag-labeled) entry if
# that guess is wrong.
_IRREGULAR_SINGULAR = {
    "elves": "elf", "faeries": "faerie", "werewolves": "werewolf", "dwarves": "dwarf",
    "wolves": "wolf", "mice": "mouse", "fungi": "fungus", "heroes": "hero", "pegasi": "pegasus",
    "leaves": "leaf", "knives": "knife", "selves": "self", "gods": "god",
}

# A small, deliberately non-exhaustive hand alias list for EDHREC `theme`-kind entries (not
# `typal`, which the automatic rule below handles) whose slug doesn't match a `card_tags` tag
# directly but a genuinely close single-tag analog exists. Most of the 228 non-typal misses
# found live (Tokens, Aristocrats, Voltron, ...) have no clean 1:1 tagger-tag equivalent -- they
# are broad EDHREC umbrellas covering many distinct mechanical tags -- and are deliberately left
# unmapped rather than rolled up into an approximate multi-tag union, which would be a materially
# bigger undertaking than "a small alias dict" and would blur exactly the resolution distinction
# this module exists to measure.
_THEME_ALIASES = {
    "Card Draw": "draw",
    "Removal": "removal",
    "Ramp": "mana-producer",
    "Cycling": "cycle",
    "Discard": "discard",
    "Lifegain": "lifegain",
    "Mill": "mill",
    "Infect": "poisonous",
    "Flying": "gives-flying",
    "Recursion": "recursion",
}


def _singularize(word: str) -> str:
    """Best-effort English singularization for a `typal` theme's slug word -- see
    `_IRREGULAR_SINGULAR` for the cases the plain suffix rule gets wrong."""
    if word in _IRREGULAR_SINGULAR:
        return _IRREGULAR_SINGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def build_theme_tag_map(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per `edhrec_themes` entry: `theme`, `kind`, `slug`, `tag` (the matched
    `card_tags` tag, or `None` if unmatched), `match_source` (`exact` | `typal` | `alias` |
    `none`). Matching order: (1) the theme's own `slug` is a `card_tags` tag verbatim; (2) for
    `typal`-kind themes, `typal-<singularized-slug>` (see `_singularize`); (3) `_THEME_ALIASES`;
    else unmatched. Diagnostic table -- `build_theme_labels` is what actually applies this
    mapping to produce per-card labels."""
    tags = {r[0] for r in conn.execute("SELECT DISTINCT tag FROM card_tags")}
    rows = conn.execute("SELECT theme, kind, slug FROM edhrec_themes").fetchall()

    records = []
    for theme, kind, slug in rows:
        if slug in tags:
            records.append({"theme": theme, "kind": kind, "slug": slug, "tag": slug, "match_source": "exact"})
            continue
        if kind == "typal":
            singular_slug = "-".join(_singularize(w) for w in slug.split("-"))
            candidate = f"typal-{singular_slug}"
            if candidate in tags:
                records.append(
                    {"theme": theme, "kind": kind, "slug": slug, "tag": candidate, "match_source": "typal"}
                )
                continue
        if theme in _THEME_ALIASES and _THEME_ALIASES[theme] in tags:
            records.append(
                {"theme": theme, "kind": kind, "slug": slug, "tag": _THEME_ALIASES[theme], "match_source": "alias"}
            )
            continue
        records.append({"theme": theme, "kind": kind, "slug": slug, "tag": None, "match_source": "none"})
    return pd.DataFrame(records, columns=["theme", "kind", "slug", "tag", "match_source"])


def build_theme_labels(
    conn: sqlite3.Connection,
    *,
    pool_oracle_ids: set[str] | None = None,
    min_cards: int = MIN_TAG_CARDS,
    max_cards: int = MAX_TAG_CARDS,
) -> pd.DataFrame:
    """One row per (`oracle_id`, `label`): `label`, `source` (`theme` if this tag maps to a
    known EDHREC theme, else `tag`), `edhrec_theme` (that theme's name, else `None`).

    Vocabulary = every `card_tags` tag with `min_cards <= n <= max_cards` cards, counted only
    within `pool_oracle_ids` if given (e.g. an NMF/SymNMF build's own card pool -- matches
    whatever universe the enrichment `N` in `topic_theme_enrichment` will use) or globally
    otherwise. A tag's *membership* (which cards get the label) is always the raw `card_tags`
    membership restricted to the same pool -- `theme`-sourced rows only rename the label via
    `build_theme_tag_map`, never substitute a different membership set."""
    tags = pd.read_sql_query("SELECT oracle_id, tag FROM card_tags", conn)
    if pool_oracle_ids is not None:
        tags = tags[tags["oracle_id"].isin(pool_oracle_ids)]

    sizes = tags.groupby("tag").size()
    vocab = set(sizes[(sizes >= min_cards) & (sizes <= max_cards)].index)
    tags = tags[tags["tag"].isin(vocab)].copy()

    theme_map = build_theme_tag_map(conn)
    theme_map = theme_map[theme_map["tag"].notna() & theme_map["tag"].isin(vocab)]
    # A tag matching more than one theme (e.g. two themes sharing a slug) keeps its first match
    # only -- collisions weren't observed live but are guarded rather than silently duplicating
    # label rows.
    tag_to_theme = dict(zip(theme_map["tag"], theme_map["theme"]))

    tags["edhrec_theme"] = tags["tag"].map(tag_to_theme)
    tags["label"] = tags["edhrec_theme"].fillna(tags["tag"])
    tags["source"] = np.where(tags["edhrec_theme"].notna(), "theme", "tag")

    return tags[["oracle_id", "label", "source", "edhrec_theme"]].reset_index(drop=True)


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Standard BH step-up FDR: sorted ascending p_(1)<=...<=p_(n), q_(i) = min_{j>=i}
    (p_(j) * n / j), monotone non-decreasing when read in sorted order."""
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def topic_theme_enrichment(
    memberships: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    min_cards: int = MIN_TOPIC_LABEL_CARDS,
    min_lift: float = MIN_LIFT,
    fdr: float = FDR_Q,
) -> pd.DataFrame:
    """Hypergeometric enrichment of every (topic, label) pair with `>= min_cards` shared cards,
    BH-FDR corrected across the *entire* tested grid (not per-topic -- a multiple-testing family
    is every pair actually tested here), then filtered to `lift >= min_lift AND q <= fdr` --
    statistical significance (BH) first, a minimum effect-size floor (lift) second, so a
    technically-significant but negligible-lift pair (common with a huge pool) doesn't count as
    "enriched."

    `N` = the topic-membership universe (`memberships["oracle_id"].nunique()`), `K` = a topic's
    member count, `n` = a label's member count *within that same universe* (labels outside the
    universe don't inflate `n`), `k` = the overlap. Columns: `topic_id`, `label`, `n_in_topic`
    (=`k`), `label_size` (=`n`), `topic_size` (=`K`), `lift` (`k / (K*n/N)`), `p` (hypergeometric
    upper-tail), `q` (BH-corrected), `share_of_topic` (`k/K` -- how much of the topic this label
    accounts for), `recall_of_label` (`k/n` -- how much of the label this topic captures),
    `resolution` (`min(share_of_topic, recall_of_label)` -- the granularity objective, see
    module docstring)."""
    pool = set(memberships["oracle_id"])
    n_universe = len(pool)
    labels = labels[labels["oracle_id"].isin(pool)]
    label_sizes = labels.groupby("label").size()

    label_by_card: dict[str, list[str]] = labels.groupby("oracle_id")["label"].apply(list).to_dict()

    rows = []
    for topic_id, group in memberships.groupby("topic_id"):
        members = group["oracle_id"].tolist()
        topic_size = len(members)
        counts: dict[str, int] = {}
        for oracle_id in members:
            for label in label_by_card.get(oracle_id, ()):
                counts[label] = counts.get(label, 0) + 1
        for label, k in counts.items():
            if k < min_cards:
                continue
            label_size = int(label_sizes[label])
            expected = topic_size * label_size / n_universe
            if expected <= 0:
                continue
            rows.append(
                {
                    "topic_id": topic_id,
                    "label": label,
                    "n_in_topic": k,
                    "label_size": label_size,
                    "topic_size": topic_size,
                    "lift": k / expected,
                    "p": float(hypergeom.sf(k - 1, n_universe, label_size, topic_size)),
                    "share_of_topic": k / topic_size,
                    "recall_of_label": k / label_size,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result.assign(q=[], resolution=[])

    result["q"] = _benjamini_hochberg(result["p"].to_numpy())
    result["resolution"] = np.minimum(result["share_of_topic"], result["recall_of_label"])

    enriched = result[(result["lift"] >= min_lift) & (result["q"] <= fdr)].reset_index(drop=True)
    return enriched


def granularity_report(
    memberships: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    min_cards: int = MIN_TOPIC_LABEL_CARDS,
    min_lift: float = MIN_LIFT,
    fdr: float = FDR_Q,
    thresholds: tuple[float, ...] = RESOLUTION_THRESHOLDS,
) -> dict:
    """The headline granularity numbers: vocabulary size, how many labels are *covered*
    (enriched in >=1 topic -- recall, near-saturated, not the objective) vs. *resolved* at each
    threshold in `thresholds` (`min(share_of_topic, recall_of_label) >= tau` for the label's
    best-scoring topic -- the actual objective, see module docstring), and median distinct
    labels enriched per topic (the direct measure of bundling)."""
    pool = set(memberships["oracle_id"])
    vocab_size = labels.loc[labels["oracle_id"].isin(pool), "label"].nunique()

    enrichment = topic_theme_enrichment(memberships, labels, min_cards=min_cards, min_lift=min_lift, fdr=fdr)

    all_topic_ids = memberships["topic_id"].unique()
    n_topics = len(all_topic_ids)
    if enrichment.empty:
        return {
            "vocab_size": int(vocab_size),
            "n_topics": int(n_topics),
            "n_labels_covered": 0,
            "n_labels_resolved": {f"{t:.2f}": 0 for t in thresholds},
            "median_labels_per_topic": 0.0,
        }

    n_covered = enrichment["label"].nunique()
    best_resolution = enrichment.groupby("label")["resolution"].max()
    resolved = {f"{t:.2f}": int((best_resolution >= t).sum()) for t in thresholds}
    labels_per_topic = enrichment.groupby("topic_id")["label"].nunique()
    # reindex against the *actual* topic_id values present in `memberships`, not `range(n_topics)`
    # -- a flat NMF/SymNMF build numbers topics 0..k-1 contiguously, but a hierarchy cut's
    # topic_id is a node_id (arbitrary, non-contiguous across the whole tree); assuming a 0..n
    # range here previously produced an all-zero median for any non-contiguous scheme.
    median_per_topic = float(labels_per_topic.reindex(all_topic_ids).fillna(0).median())

    return {
        "vocab_size": int(vocab_size),
        "n_topics": int(n_topics),
        "n_labels_covered": int(n_covered),
        "n_labels_resolved": resolved,
        "median_labels_per_topic": median_per_topic,
    }


def color_purity(
    memberships: pd.DataFrame,
    conn: sqlite3.Connection,
    *,
    top_n: int = DEFAULT_TOP_N_PURITY,
    n_null_draws: int = DEFAULT_NULL_DRAWS,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-topic color-identity purity: among a topic's top-`top_n` members by `weight`, the
    share carrying the topic's single most common color identity (colorless cards get their own
    identity `"C"`, not conflated with "no data"). Also draws `n_null_draws` random `top_n`-card
    samples from the full membership pool as a null baseline -- the distribution a topic with no
    color bias at all would produce by chance alone, for comparison (not a per-topic p-value;
    the planning-session baseline used this same random-null comparison purely descriptively).

    Returns `(per_topic_df, null_draws_array)`. `per_topic_df` columns: `topic_id`, `size`
    (total membership rows for that topic), `modal_color_identity`, `purity`."""
    pool_oracle_ids = list(set(memberships["oracle_id"]))
    ci_rows = conn.execute(
        f"SELECT oracle_id, color_identity FROM cards WHERE oracle_id IN "
        f"({','.join('?' * len(pool_oracle_ids))})", pool_oracle_ids
    ).fetchall()
    ci_map = {
        oracle_id: "".join(sorted(json.loads(raw or "[]"))) or "C" for oracle_id, raw in ci_rows
    }

    rows = []
    for topic_id, group in memberships.groupby("topic_id"):
        top = group.sort_values("weight", ascending=False).head(top_n)
        identities = [ci_map.get(oid, "?") for oid in top["oracle_id"]]
        counts = pd.Series(identities).value_counts()
        modal_ci, modal_n = counts.index[0], int(counts.iloc[0])
        rows.append(
            {
                "topic_id": topic_id,
                "size": len(group),
                "modal_color_identity": modal_ci,
                "purity": modal_n / len(identities),
            }
        )
    purity_df = pd.DataFrame(rows).sort_values("purity", ascending=False).reset_index(drop=True)

    pool_ci = np.array([ci_map.get(oid, "?") for oid in memberships["oracle_id"].unique()])
    rng = np.random.default_rng(seed)
    draws = min(top_n, len(pool_ci))
    null = np.empty(n_null_draws)
    for i in range(n_null_draws):
        sample = rng.choice(pool_ci, draws, replace=False)
        counts = pd.Series(sample).value_counts()
        null[i] = int(counts.iloc[0]) / draws

    return purity_df, null


def _load_memberships(build: str) -> pd.DataFrame:
    if build == "nmf":
        from edhcut.analysis.nmf_packages import load_card_memberships

        return load_card_memberships()
    if build == "symnmf":
        from edhcut.analysis.symnmf_packages import load_card_memberships

        return load_card_memberships()
    raise SystemExit(f"Unknown --build {build!r} (expected 'nmf' or 'symnmf').")


def _cmd_report(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        memberships = _load_memberships(args.build)
        pool = set(memberships["oracle_id"])
        labels = build_theme_labels(conn, pool_oracle_ids=pool)
        theme_map = build_theme_tag_map(conn)

        report = granularity_report(memberships, labels)
        purity_df, null = color_purity(memberships, conn, top_n=args.top_n, n_null_draws=args.null_draws)

    n_theme_matched = int((theme_map["tag"].notna()).sum())
    print(f"Theme->tag mapping: {n_theme_matched} / {len(theme_map)} EDHREC themes matched to a card_tags tag "
          f"({(theme_map['match_source'] == 'exact').sum()} exact, "
          f"{(theme_map['match_source'] == 'typal').sum()} typal, "
          f"{(theme_map['match_source'] == 'alias').sum()} alias).")
    print()
    print(f"Label vocabulary: {report['vocab_size']} tags ({MIN_TAG_CARDS}-{MAX_TAG_CARDS} cards, "
          f"restricted to this build's {len(pool):,}-card pool) across {report['n_topics']} topics.")
    print(f"  covered (enriched in >=1 topic):   {report['n_labels_covered']} / {report['vocab_size']}")
    for thresh, n in report["n_labels_resolved"].items():
        print(f"  resolved at tau={thresh}:            {n} / {report['vocab_size']}")
    print(f"  median distinct labels per topic:  {report['median_labels_per_topic']:.1f}")
    print()
    print(f"Color purity (top-{args.top_n} members by weight, modal color-identity share):")
    print(f"  median across {len(purity_df)} topics: {purity_df['purity'].median():.1%}")
    for thresh in (0.7, 0.8, 0.9):
        print(f"  topics >= {thresh:.0%}: {int((purity_df['purity'] >= thresh).sum())}")
    print(f"  random null ({args.null_draws} draws): median {np.median(null):.1%}, "
          f"p95 {np.percentile(null, 95):.1%}, max {null.max():.1%}")

    if args.json:
        out = {
            "build": args.build,
            "n_pool_cards": len(pool),
            "theme_map_matched": n_theme_matched,
            "theme_map_total": len(theme_map),
            "granularity": report,
            "color_purity_median": float(purity_df["purity"].median()),
            "color_purity_null_median": float(np.median(null)),
            "color_purity_null_p95": float(np.percentile(null, 95)),
            "color_purity_topics_ge_70": int((purity_df["purity"] >= 0.7).sum()),
            "color_purity_topics_ge_80": int((purity_df["purity"] >= 0.8).sum()),
            "color_purity_topics_ge_90": int((purity_df["purity"] >= 0.9).sum()),
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    report_p = sub.add_parser("report", help="Print granularity + color-purity numbers for a saved build")
    report_p.add_argument("--build", choices=["nmf", "symnmf"], default="nmf")
    report_p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_PURITY)
    report_p.add_argument("--null-draws", type=int, default=DEFAULT_NULL_DRAWS)
    report_p.add_argument("--json", default=None, help="Optional path to also write the report as JSON")
    report_p.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
