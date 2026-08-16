# EDHCut

A data-driven recommender for **Magic: The Gathering — Commander (EDH)**: given a decklist, suggest
which card to cut. This repo holds the data pipeline, the analysis, and the exploration notebooks.

**If you're here to write the report, start at [§7](#7-writing-the-report)** — it maps our three
worked *directions* onto the material we have and proposes candidate research questions inside
each. [§8](#8-structuring-the-report) sketches the report's shape.

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
alternatives we rejected and why. **Those files are the primary source for the report** — this table
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
| Functional role classification (removal, ramp, draw, …) | **not started** | — |
| Oracle-text mining for power level | **not started** | — |
| **The cut recommender itself** | **not started** | — |
| API, frontend, deployment | **not started** | — |

In short: **the knowledge base is built and validated; the tool that would consume it is not.**
That's the honest framing for the report — we have a thorough analysis of a dataset, not a shipped
product, and the analysis is where all the interesting content is anyway.

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

Expect **405 passed, 3 skipped**.

---

## 4. Getting the data

`Project/EDHCut/data/` is gitignored — it's ~560 MB, mostly a SQLite database and derived matrices.

### Option A — get the files from Aviv (recommended)

Drop them so the tree looks like this:

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
| **`cooccurrence_metrics.ipynb`** | The association metrics we compare — raw co-occurrence, PMI, lift, t-score — with worked card examples. **This is the core methods notebook for research question 3.** |
| **`tscore_metric.ipynb`** | Why t-score specifically, and the colour-identity correction. Read alongside the one above. |
| **`precon_exploration.ipynb`** | The precon confounder: measuring how much of the co-occurrence signal is "people bought a product and didn't change it". |
| **`embeddings_exploration.ipynb`** | Card similarity from oracle text (TF-IDF + SVD, and Word2Vec). **Research question 2.** |
| **`symNMF_communities.ipynb`** | **The main result.** Soft/overlapping community detection — a card can belong to several synergy packages at once. |
| **`louvain_communities.ipynb`** | The hard-partition comparison. **Identical to the symNMF notebook through the entire pipeline** — same graph, same filters, same helper functions — so the two are directly comparable and the only difference is the clustering algorithm. |
| `my_kyler_deck.ipynb` | A worked example on one real deck. Nice for a concrete illustration in the report. |
| `notebooks/archive/` | A superseded approach (NMF topic modelling). Useful only for the "what we tried and abandoned" part of the report. |

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

## 7. Writing the report

### First: we have directions, not questions yet

The three areas below are **where the work went and where the material is**. None of them is a
research question — they are too broad to answer. Part of the report's job (and a graded part) is
choosing a sharp question inside each: one with a defensible answer and a figure to show.

Candidates are proposed per direction. They are starting points — pick, sharpen, or replace.

---

### Direction 1 — Community detection (Louvain, symNMF)

**Material**: `symNMF_communities.ipynb`, `louvain_communities.ipynb`;
`docs/devlog/6.3-synergy-communities.md`, `6.3c`, `6.3d`.

The strongest asset here: the system is given **no rules of Magic** — no card text, no types, no
notion of what a card does. Only "these cards appeared in the same deck". And communities are
auto-named from Scryfall mechanic tags **held out of the clustering entirely**, so the names are an
*independent* check rather than an input.

| candidate question | why it is answerable | figure |
|---|---|---|
| **"Can mechanically coherent card packages be recovered from co-occurrence alone, with no game knowledge?"** *(recommended)* | Held-out tags give an objective yes/no. Communities come out as `typal-goblin`, `lands-matter/landfall`, `equipment/quick-attach`. | the interactive graph; card-image rows per named community |
| "Do cards belong to one archetype or several — does soft clustering recover structure the hard partition destroys?" | Identical pipelines, only the algorithm differs. 41% of cards land in >1 community; Lion Sash = 39% Cats / 30% Equipment / 24% equipment-tutors. | side-by-side community tables; the multi-membership example |
| "At what granularity do communities stop being archetypes and start being noise?" | We have the level hierarchy, the resolution sweep, and package-size distributions across settings. | package-sized-community count vs. resolution |

The second is the most *novel*, the first the most *defensible*. They combine well: answer the
first, use the second as the interesting wrinkle.

---

### Direction 2 — Functionally similar cards (TF-IDF)

**Material**: `embeddings_exploration.ipynb`; `docs/devlog/6.2-embeddings.md`.

**This is the thinnest of the three.** It got far less attention than the others and may need new
work rather than write-up. Budget accordingly, or fold it into Direction 1 as a comparison.

| candidate question | why it is answerable | figure |
|---|---|---|
| **"Do text similarity and co-occurrence similarity find the same relationships, or complementary ones?"** *(recommended)* | Both signals exist for every card, and we have a measurement showing they come apart: boardwipes (a functional category) have median pairwise lift 2.8, landfall (a synergy package) 16.5. | scatter of text-similarity vs. co-occurrence-similarity per pair; the two rank-lists for one card |
| "Can oracle text alone predict a card's synergy community?" | Turns Direction 1's communities into labels and Direction 2's embeddings into features — a clean supervised task with a real accuracy number. | confusion matrix; per-community accuracy |

The first framing is the strong one: **text finds substitutes** (cards that do the same thing),
**co-occurrence finds complements** (cards that work together). That is a conceptual result, not
just a metric comparison.

**Standing validation to cite**: Cultivate's nearest neighbour is Kodama's Reach under PMI, t-score,
Word2Vec *and* TF-IDF independently — four methods, one answer, two near-identical cards.

---

### Direction 3 — Data biases and metric choice

**Material**: `cooccurrence_metrics.ipynb`, `tscore_metric.ipynb`, `precon_exploration.ipynb`;
`docs/devlog/6.1-cooccurrence.md`, `6.1b`, `5.3c`, `6.3d`.

The richest direction, and where most design effort went. Five confounders, each measured, each
handled differently:

1. **Colour legality** — cards co-occur because the game *permits* them together. Fixed by
   conditioning the null model on legality. Validated on 20 real pairs: 0–8% change for genuine
   synergy, 48–92% collapse for generic same-colour pairs, and exactly 0% change whenever one card
   is colourless — an algebraic self-check, not merely an empirical one.
2. **Precon shells** — people buy a product and change little, so its 100 cards co-occur massively.
3. **Commander oversampling** — the top 2 commanders are 24.6% of the corpus. Corrected by
   inverse-probability deck weighting, with a permanent guard re-measuring it on every run.
4. **Generically-good cards** — Sol Ring is in 82.5% of decks and correlates with everything.
5. **Lands vs. spells** — basic lands excluded (no mechanical identity); other lands kept, because
   some genuinely are synergy pieces.

| candidate question | why it is answerable | figure |
|---|---|---|
| **"Which association metric isolates mechanical synergy from structural artifacts — and what does each one fail at?"** *(recommended)* | Four metrics, worked examples, and *proofs* that particular fixes are impossible. | one card's top-10 neighbours under each metric, side by side |
| "How much of the observed co-occurrence is mechanical synergy vs. artifact?" | Each confounder has a before/after measurement — a decomposition question with real numbers. | stacked contribution chart per confounder |
| "Can a confounder be removed by correcting the null model, or must it be removed structurally?" | We have one clean example of each: colour legality yielded to a corrected null; the staples confound did not, and needed a structural fix. | before/after neighbour lists for both cases |

**Three negative results, the most citable things in the project** — negative results are hard to
come by and easy to defend:

- **PMI ties rare coincidences at a ceiling**, manufacturing spurious clusters. That is why we do
  not use it.
- **t-score is a *significance* measure, not an *effect size***. On large counts a tiny relative
  excess is many sigma, so a card in 8,682 decks scores like a real synergy piece. **Lift cannot fix
  this**: the ranges overlap *in the wrong direction* — pairs that must be kept span lift
  2.56–61.94, pairs that must be dropped span 1.24–3.06. It follows that **no statistic computed
  from `(co-occurrence, count_A, count_B, N)` can separate them.** That is a small impossibility
  argument, not just a failed experiment.
- The fix had to be **structural**: the Jaccard overlap of the two cards' *neighbourhoods*. Pairwise
  statistics were exhausted; graph structure supplied what they could not.

---

## 8. Structuring the report

The arc below starts from Aviv's outline; additions and disagreements are flagged.

| section | content | material |
|---|---|---|
| 1 | **The world of Magic** — enough rules to follow the analysis, framed as *why this dataset is statistically interesting*. Introduce colour identity and the land/spell split here, since they become confounders later. | §1 of this README |
| 2 | **Choosing the questions** — what we set out to build, and how the questions emerged | §7 above |
| 3 | **Getting the data** — four sources, why each, what the harvest cost, the QA pass | devlogs `5.1`–`5.7` |
| 4 | **Metrics and analysis** — co-occurrence → PMI → lift → t-score, each fixing the last one's failure | `6.1`, `6.1b` |
| 5 | **Problems and design shifts** — the confounders and the two big pivots | `5.7`, `6.3`, `6.3c`, `6.3d` |
| 6 | **Validation** — *added, see below* | throughout |
| 7 | **Results** — the answers, with figures | the notebooks |
| 8 | **Limitations** — *added, see below* | `6.3d` "Notes for future work" |
| 9 | **Future directions** | `6.3d`, plan tasks 6.4+ |

### Two sections worth adding

**Validation (§6).** How do you know an answer is *right*? Usually heavily weighted in a
data-science course, and we have unusually good material that would otherwise scatter across other
sections:

- **Held-out labels** — communities are named from Scryfall tags that never enter the clustering.
- **Null models** — colour purity is always reported against same-size random draws (~53% vs ~20%),
  so "more than chance?" has an actual answer.
- **Seed stability** — clusterings re-run across seeds, agreement measured.
- **A standing regression check** — Cultivate's top neighbour must be Kodama's Reach; holds across
  four independent methods.
- **A permanent guard** — commander-domination is re-measured on every notebook run, because that
  regression *did* happen once and was caught by eye from a rendered graph before any metric flagged
  it.

**Limitations (§8), distinct from future work.** Future work is optimistic; limitations are threats
to validity, and graders look for them. Ours are concrete:

- **Commanders are invisible to the model.** `deck_cards` contains 0 of 13,207 decks' commanders —
  the single most defining card of each deck is absent from the co-occurrence data.
- **The corpus is one website.** Archidekt users are not the whole player base.
- **Thin statistics.** The median card in the pool appears in **14 decks**; a card in 5 decks that
  happens to sit in 5 zombie decks is indistinguishable from a real zombie card.
- **A residual 3-colour bias**, measured but unfixed: 31% of 3-colour cards end up unassigned versus
  3–6% for other colour counts at equal popularity.

### On the order

The arc is sound and worth keeping. Three notes:

1. **§4 and §5 overlap heavily.** The problems *are* the analysis — each confounder is what forced
   the next metric. As separate sections you will either repeat yourself or leave §4 bloodless.
   Suggestion: let **§4 carry the metric-level story** (including the confounder that motivated each
   metric) and reserve **§5 for the big pivots** — abandoning NMF, the reset to Louvain, adopting
   soft clustering. Those are different in kind: project-level decisions, not metric ones.
2. **§2 before §3 is mildly ahistorical.** The data came first and the questions crystallised from
   it. Fine as narrative, but do not claim we chose questions and then went data-hunting — "here is
   what we set out to build, and here is how the questions emerged" is both truer and a better story.
3. **§1 should motivate, not just explain.** The hook is that this is a bipartite deck×card dataset
   in which several strong non-mechanical signals compete to explain the same co-occurrence. Put
   that on the page early and every later section has a reason to exist.

### What to cut if you are short

The abandoned approaches (`docs/devlog/archive/`) and rejected alternatives — Girvan–Newman at ~65
min per subgraph, k-clique percolation collapsing to one 15,479-node blob, degree normalisation that
improved every metric but produced worse communities — are good material, but they belong in §5 as
*brief* illustrations. Do not let them eat the results section.

---

## 9. Project layout

```
Project/EDHCut/
├── edhcut/
│   ├── ingest/        one module per data source (Scryfall, Archidekt, EDHREC, Tagger, precons)
│   ├── analysis/      co-occurrence metrics, communities, embeddings, play rates, deck weights
│   ├── config.py      paths + the 5-commander roster
│   └── db.py          SQLite schema and connection
├── notebooks/         all exploration, committed with outputs
├── docs/devlog/       one write-up per task — the report's primary source
├── tests/             405 tests
└── data/              gitignored; get it from Aviv (§4)
```

Most analysis modules also have a CLI, e.g.:

```bash
venv/Scripts/python -m edhcut.analysis.cooccurrence top "Cultivate"
```
