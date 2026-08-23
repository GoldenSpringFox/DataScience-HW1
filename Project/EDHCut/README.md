# EDHCut

A data-driven recommender for **Magic: The Gathering — Commander (EDH)**: given a decklist, suggest
which card to cut. This repo holds the data pipeline, the analysis, and the exploration notebooks.

> **Course submission — group 34, 67978 "A Needle in a Data Haystack".** The writeup
> (`writeup_group34.pdf`) is the account of *what* we found; this repo is *how*. Every claim and
> figure in it is reproducible from here. If you are grading and want the shortest path:
> [§7](#7-what-to-read-first) maps each of the writeup's three questions onto the notebook, the
> code and the devlog behind it, and [§9](#9-where-the-code-is) maps each one onto the module that
> implements it. The notebooks are committed **with their outputs**, so everything is readable
> without installing or running anything.

**New here? Start at [§7](#7-what-to-read-first).**

---

## 1. Enough Magic to read the notebooks

You do not need to play the game. You need seven facts.

| term | what it means here |
|---|---|
| **Deck** | Exactly 100 cards. **Singleton** — at most one copy of any card (basic lands excepted). |
| **Commander** | One special card that leads the deck. It is *not* one of the other 99, and it constrains what the other 99 may be. |
| **Land / spell** | The two halves of every deck. **Lands** produce *mana*, the resource; you may play only one per turn, so a deck runs ~35–40 of them and they are mostly interchangeable plumbing. **Spells** are everything else — the ~60–65 cards that actually do things. This split matters statistically: lands are near-universal within a colour, so they co-occur with everything in that colour and carry almost no information about *what a deck is trying to do*. **Basic lands** (5 types, unlimited copies allowed) are the extreme case and we exclude them from the analysis entirely; other lands are kept, because some genuinely are synergy pieces. |
| **Colour identity** | Every card belongs to some subset of 5 colours (White, Blue, Black, Red, Green). A deck may only contain cards whose colour identity is a **subset of its commander's**. This is a hard legality rule and it is the single biggest confounder in the whole dataset — two green cards co-occur partly because they're both green, not because they interact. |
| **Staple** | A card played in a huge fraction of all decks because it's generically strong (Sol Ring is in 82.5% of our 13,207 decks). Staples co-occur with *everything*, which makes them statistical noise. |
| **Synergy package** | A small group of cards that are good *specifically together* — e.g. cards that trigger on playing a land, plus cards that let you play extra lands. **Finding these automatically is the core problem of this project.** |
| **Precon** | A pre-built deck sold as a product. Many people buy one and change little, so its 100 cards co-occur enormously — another confounder. |

**The one-sentence framing of the project**: card co-occurrence across 13,207 decklists is a
bipartite deck×card dataset; we want the *mechanical* structure in it, while several strong
non-mechanical signals (colour legality, precon shells, generically-good cards, oversampled
commanders) compete to explain the same co-occurrence.

---

## 2. What's built, and what isn't

Every task below has a devlog under `docs/devlog/` with the full design rationale, the numbers, the
alternatives we rejected and why. **Those files are the fullest account of each stage** — this table
is only a map into them.

| stage | status | devlog |
|---|---|---|
| Project scaffold, SQLite schema, rate-limited HTTP client | done | `5.1` |
| **Card data** — all 31,830 cards from Scryfall, plus name resolution | done | `5.2` |
| **Decklists** — 13,207 decks harvested from Archidekt | done | `5.3a`, `5.3b` |
| **Precon products** + how much of each deck is an unmodified precon | done | `5.3c` |
| **Metagame + themes** from EDHREC | done | `5.4a`–`5.4c` |
| **Mechanic tags** — 376,800 tags over ~950 types (Scryfall Tagger) | done | `5.5` |
| QA & coverage report | done | `5.6` |
| Broadening the corpus from 5 commanders to 1,001 | done | `5.7` |
| **Association metrics** — co-occurrence, PMI, lift, t-score, colour-conditioned t-score | done | `6.1`, `6.1b` |
| **Card embeddings** — TF-IDF + SVD over oracle text, and Word2Vec | done | `6.2` |
| **Synergy communities** — the long arc: Leiden → NMF → symNMF → Louvain → symNMF | done | `6.3`, `6.3c`, `6.3d` |
| **Functional roles** — one of 18 roles (primary + optional secondary) for all 31,830 cards | done | `6.4` |
| Oracle-text mining for power level | **not started** | — |
| **The cut recommender itself** | **not started** | — |
| API, frontend, deployment | **not started** | — |

In short: **the knowledge base is built and validated; the tool that would consume it is not.**
What this is, then, is a thorough analysis of a dataset rather than a shipped
product — and the analysis is where all the interesting content is anyway.

---

## 3. Setup

Python **3.11+** (we run 3.14). From the repo root:

```bash
python -m venv venv
```
```bash
venv/Scripts/pip install -e "Project/EDHCut[notebooks]"
```

On macOS/Linux use `venv/bin/pip`. The `[notebooks]` extra pulls in jupyter, plotly, networkx and
ipysigma; without it the library imports but the notebooks won't run.

One optional extra: `gensim` (used only by the Word2Vec path in `embeddings.py`) has no wheel for
Python 3.14, so there's a second venv `venv311/` just for that. **You almost certainly don't need
it** — everything else, including the TF-IDF embeddings, runs in the main venv.

Verify:

```bash
venv/Scripts/python -m pytest Project/EDHCut -q
```

Expect **460 passed, 3 skipped**.

---

## 4. Getting the data

`Project/EDHCut/data/` is gitignored — it's ~560 MB, mostly a SQLite database and derived matrices.

### Option A — download it (recommended)

**[edhcut data.zip (450 MB)](https://drive.google.com/file/d/1KuzvmON9hQTxn7LKTKySL08ZHkqGGSkU/view?usp=drive_link)**
— unzip into `Project/EDHCut/data/`. It contains everything the analysis reads: `edhcut.db`,
`kb/dev/`, the raw Scryfall bulk files, the fixtures and the harvest logs. The one thing it leaves
out is `http_cache.sqlite`, a 5.6 GB cache of raw HTTP responses that only matters if you intend to
re-run the harvest itself.

The result should look like this:

```
Project/EDHCut/data/
├── edhcut.db          185 MB   the whole corpus: cards, decks, deck_cards, tags, precons
└── kb/dev/            371 MB   derived matrices (co-occurrence, t-score, embeddings, ...)
```

That's all. Nothing needs configuring — paths are resolved relative to the package in
`edhcut/config.py`.

If you only got `edhcut.db`, you can regenerate `kb/dev/` yourself:

```bash
venv/Scripts/python -m edhcut.analysis.cooccurrence build
```

(run from `Project/EDHCut/`; takes a few minutes). The two community notebooks additionally build
and cache their own weighted matrix on first run, ~80s, automatically.

### Option B — rebuild from scratch (hours, hits live APIs — avoid unless you must)

In order: `edhcut.ingest.scryfall` (card data), `edhcut.ingest.tagger_bulk` (mechanic tags),
`edhcut.ingest.edhrec_commanders` + `edhcut.ingest.edhrec` (metagame + themes),
`edhcut.ingest.precons` (precon lists), `edhcut.ingest.archidekt` (the 13k decklists — this is the
long one), then `precon_similarity`, `precon_retention`, `qa_report`. Every module is
`python -m edhcut.ingest.<name>`. Details in `docs/devlog/5.*.md`.

### What's in the corpus

- **13,207 decks** across **1,001 distinct commanders**, harvested from Archidekt
- **31,830 cards** known; **18,979** appear in ≥3 decks (the analysis pool); **16,026** survive
  into the final graph
- **376,800 mechanic tags** over ~950 tag types, from Scryfall Tagger — our only ground-truth-ish
  labelling of what a card *does*
- Heavily **unbalanced by commander**: the top 2 commanders are 24.6% of all decks, against a
  median commander with ~8. This is a deliberate artifact of how we harvested and it is corrected
  (not ignored) — see `docs/devlog/5.7-metagame-harvest.md`.

---

## 5. Running the notebooks

```bash
venv/Scripts/jupyter lab
```

from `Project/EDHCut/`, then open anything in `notebooks/`.

All notebooks are **committed with their outputs**, so you can read every result and figure without
running a single cell. Run them only if you want to interact with the graph or call the helpers.

> **Note on editing:** the notebooks are the source of truth. Edit them directly.

---

## 6. Which notebooks to look at

In reading order. The last two are the main results.

| notebook | what it shows |
|---|---|
| **`01_data_overview.ipynb`** | Start here. What the corpus contains, deck/card distributions, coverage. Basic EDA. |
| **`cooccurrence_metrics.ipynb`** | The association metrics we compare — raw co-occurrence, PMI, lift, t-score — with worked card examples. **This is the core methods notebook for Chapter 1.** |
| **`tscore_metric.ipynb`** | Why t-score specifically, and the colour-identity correction. Read alongside the one above. |
| **`precon_exploration.ipynb`** | The precon confounder: measuring how much of the co-occurrence signal is "people bought a product and didn't change it". |
| **`embeddings_exploration.ipynb`** | Card similarity from oracle text (TF-IDF + SVD, and Word2Vec). **Chapter 3.** |
| **`symNMF_communities.ipynb`** | **The main result.** Soft/overlapping community detection — a card can belong to several synergy packages at once. |
| **`louvain_communities.ipynb`** | The hard-partition comparison. **Identical to the symNMF notebook through the entire pipeline** — same graph, same filters, same helper functions — so the two are directly comparable and the only difference is the clustering algorithm. |
| **`roles_exploration.ipynb`** | Functional roles: what each of the 18 roles means, how the Scryfall Tagger hierarchy is turned into them, and the top 100 most-played cards as images with their assigned roles. **§5 is a hands-on toolkit** — filter cards by role, Scryfall tag, secondary role or play count, and three purpose-built helpers for finding misclassified cards (`disagreements`, `borderline`, `thin_evidence`, `audit_tag`). |
| `my_kyler_deck.ipynb` | A worked example on one real deck. |
| `notebooks/archive/` | A superseded approach (NMF topic modelling), kept for reference. |

### The two community notebooks

Both are structured identically: **§1 setup → §2 build → §3 stats overview → §4 one interactive
graph → §5 helpers**. §3 prints everything in one block; §4 is a live force-directed graph where
each point is a card and connected cards pull together.

Useful helpers, defined in the notebook's own §5 (call them in the last cell):

```python
list_communities()               # every community, with an auto-generated name
show_community(0)                # a community's cards, as card images
show_community("Lion Sash")      # the community a given card belongs to
card_topics("Lion Sash")         # symNMF only: ALL communities a card belongs to, with weights
neighbors("Lion Sash")           # a card's strongest connections, with the metrics behind them
most_played(100)                 # most-played cards and how synergistic each one is
```

`show_community("Lion Sash")` is the fastest way to get a feel for what the algorithm found —
that card comes out 39% "Cats", 30% "Equipment", 24% "equipment tutors", which is genuinely correct
and is exactly what the hard partition cannot express.

---

## 7. What to read first

The writeup asks three questions. Each has notebooks you can read in the browser and a devlog
carrying the full design rationale, the numbers, and the alternatives we rejected. §9 adds the
module that implements each one; `../Report/figures/README.md` maps every figure in the writeup to
the script that draws it.

| question | what it asks | notebooks | devlogs |
|---|---|---|---|
| **1. Two-card combo** | What does it mean for two cards to synergize, and can any pairwise metric tell? | `cooccurrence_metrics.ipynb`, `tscore_metric.ipynb`, `precon_exploration.ipynb` | `6.1`, `6.1b`, `5.3c` |
| **2. "This would go great in my dragon deck!"** | Can coherent card packages be recovered from co-occurrence alone, with no knowledge of the game? | `symNMF_communities.ipynb`, `louvain_communities.ipynb` | `6.3`, `6.3c`, `6.3d` |
| **3. Reading the card explains the card** | Do card text and co-occurrence find the same relationships, or complementary ones? | `embeddings_exploration.ipynb`, `roles_exploration.ipynb` | `6.2`, `6.4` |

`01_data_overview.ipynb` covers the corpus itself — what it contains, how decks and cards are
distributed, coverage — and is the right place to start if you want the data before the analysis.
The four `5.*` devlogs cover acquisition and QA.

**You do not need to run anything.** Every notebook is committed *with its outputs*, so all results
and figures are readable straight from GitHub. Running them yourself needs the ~560 MB data
directory, which is gitignored (§4).

**If you only open one file**, make it `symNMF_communities.ipynb` — it is the main result. §3 prints
the community statistics in one block and §4 is a live force-directed graph of the card network.


## 8. Project layout

```
Project/EDHCut/
├── edhcut/
│   ├── ingest/        one module per data source (Scryfall, Archidekt, EDHREC, Tagger, precons)
│   ├── analysis/      co-occurrence metrics, communities, embeddings, roles, play rates, deck weights
│   ├── config.py      paths + the 5-commander roster + per-source rate limits
│   ├── db.py          SQLite schema and connection
│   ├── http.py        the cached, rate-limited session every fetch goes through
│   └── images.py      card art, for the notebooks
├── notebooks/         all exploration, committed with outputs
├── docs/devlog/       one write-up per task — rationale, numbers, rejected alternatives
├── tests/             460 tests, 3 skipped
└── data/              gitignored; download it (§4)

Project/Report/
├── report.md          the writeup source
├── figures/           one script per figure — see its own README for the figure→script map
└── Images/            the generated figures
```

Every `edhcut/` module carries a module docstring that explains what it does and, more usefully,
*why it does it that way* — which alternatives were tried, which were rejected, and what the
parameters were chosen against. Those docstrings and the matching `docs/devlog/` entry are the
real documentation; this README is only the map.

Most modules also have a CLI. `--help` lists the subcommands:

```bash
venv/Scripts/python -m edhcut.analysis.cooccurrence top "Cultivate"
```

---

## 9. Where the code is

For each question in the writeup, the module that implements it. Everything below lives in
`edhcut/analysis/` and is documented at the top of its own file.

| writeup | what it does | module |
|---|---|---|
| **Q1** Two-card combo | co-occurrence, PMI, lift, t-score, and the colour-conditioned t-score the report settles on | `cooccurrence.py` |
| Q1 | the play-rate metric, whose denominator is "decks that could *legally* run this card" | `playrate.py` |
| Q1 | inverse-probability weights correcting the harvest's commander imbalance | `deck_weights.py` |
| **Q2** Card packages | graph construction (top-k union sparsification, Jaccard gate, basic-land mask) and the Leiden partition | `communities.py` |
| Q2 | **the main result** — symmetric NMF over the colour-conditioned matrix, so a card can belong to several packages at once | `symnmf_packages.py` |
| Q2 | the Louvain hard partition the report compares against, run on the same graph | `notebooks/louvain_communities.ipynb` |
| **Q3** Reading the card | TF-IDF + SVD over oracle text, Word2Vec over decks-as-sentences, nearest-neighbour lookup | `embeddings.py` |
| Q3 | 18 functional roles per card, from mechanic tags plus a text-heuristic layer | `roles.py` |

Superseded but kept, because `notebooks/archive/communities_exploration.ipynb` documents the route
to the final method: `nmf_packages.py` (plain NMF over decks × cards), `symnmf_hierarchy.py`
(recursive refinement into a tree) and `theme_labels.py` (the label set used to score granularity).

The ingest side is one module per source in `edhcut/ingest/`, each runnable as
`python -m edhcut.ingest.<name>` and each logging its run to the `ingest_log` table; `§4 Option B`
gives the order they run in.
