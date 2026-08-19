# Report planning (internal)

> Moved out of `README.md` on 2026-08-19. The README is the landing page a grader sees; this file
> is our own planning and should stay out of it.

## Decisions locked 2026-08-19

The report has **three chapters**, chosen from the directions below. Everything from
"Direction 1/2/3" onward is the brainstorm that produced them and is kept only for reference —
where it disagrees with this block, this block wins.

| # | title | subtitle | direction it came from |
|---|---|---|---|
| 1 | Two-card combo | measuring synergy between cards | Direction 3 (metrics & confounders) |
| 2 | "This would go great in my dragon deck!" | recovering card packages from co-occurrence | Direction 1 (community detection) |
| 3 | Reading the card explains the card | text similarity vs. co-occurrence | Direction 2 (embeddings) |

Report title: **A Needle in a Stack of Magic Cards**. Draft lives in `Project/Report/report.md`.

Deliberately excluded: the cut recommender (not built) and any chapter requiring new analysis.
Chapter 3 keeps the substitutes-vs-complements punchline as its result.

Open work items: symNMF **seed-stability check** (non-deterministic algorithm, needs a fixed seed
plus cross-seed agreement reported), verifying the per-source file sizes quoted in the report, and
a live interactive demo.

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

**Material**: `embeddings_exploration.ipynb`, `roles_exploration.ipynb`;
`docs/devlog/6.2-embeddings.md`, `6.4-functional-roles.md`.

**This is the thinnest of the three.** It got far less attention than the others and may need new
work rather than write-up. Budget accordingly, or fold it into Direction 1 as a comparison.

Task 6.4 added usable material here: it classifies every card into a functional role two
independent ways — from crowdsourced mechanic tags, and from regexes over the card's own oracle
text — and the two agree **74.9%** of the time on the 13,166 cards both can read. That is a direct,
quantified answer to "how much of what a card *does* is recoverable from its text alone", with the
1,521 cards only the text layer reaches as the concrete payoff.

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
