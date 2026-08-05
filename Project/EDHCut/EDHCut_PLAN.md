# EDHCut — Architecture & Phased Build Plan (v2)

A data-driven tool that helps Commander (EDH) players cut cards from their decks.
Given a decklist + a candidate card to add, it suggests which card to cut, surfaces other
interesting swaps, and produces deck stats + archetype classification — grounded in
co-occurrence data, functional similarity, synergy clusters, and archetype fit.

**Scope decisions (locked in):**
- Deploy target: **free-tier PaaS** (Fly.io / Render), FastAPI + prebuilt read-only artifacts.
- Frontend: **React + Vite SPA** talking to a FastAPI JSON API.
- Dual purpose: **course project + personal product**. Tasks marked 📊 produce artifacts that feed the final submission directly.
- **No contingency fallbacks are built in advance.** If a data source breaks, stop and report; we cross that bridge when we get to it.

**How to use this plan with Sonnet:** each sub-task below has a `Prompt` block. Start each
implementation session by telling Sonnet to read this file, then paste the prompt. Tasks are
ordered by dependency. **Every task implicitly ends with writing a devlog entry (§4) — this
is part of the task's definition of done, not optional.**

---

## 1. Target commanders (test roster)

Five deck slots, each chosen to stress a different part of the system. The user has real,
tried-and-tested lists for #2–#5 — these go in `data/fixtures/my_decks/` as golden fixtures
for parser tests, sanity checks, and evaluation.

| # | Commander(s) | Why it's in the roster |
|---|---|---|
| 1 | **Krenko, Mob Boss** | Popular, cheap, strongly synergy-driven goblin tribal — recommendations are easy to eyeball. The spike commander (§9). |
| 2 | **Kyler, Sigardian Emissary** | Humans tribal with mass-reanimation / power-2-or-less subthemes. Released as a precon face card → **precon-contamination test**: the corpus will be full of barely-modified precon lists, and the tool must not just parrot precon contents. |
| 3 | **Yoshimaru, Ever Faithful + Bruse Tarl, Boorish Herder** | Partner pair → tests that the whole pipeline (schema, Archidekt search, EDHREC pages, parser, color identity = union) handles two commanders correctly. |
| 4 | **Yenna, Redtooth Regent** | The user's list is a lower-power **fox-themed** deck, deliberately not a typical Yenna enchantment-combo list → tests that recommendations respect the deck in front of them instead of dragging everything toward the meta build. |
| 5 | **Orysa, Tide Choreographer** | Mono-U Merfolk Bard, released 2026-04 (Secrets of Strixhaven), EDHREC rank ~19,600 → **cold-start test**: few built decks exist. Per-commander stats will be thin; the tool must degrade gracefully to global signals rather than fabricate confidence. |

Config models a "commander slot" as a list of 1–2 card names so partners are first-class
throughout (DB already stores `partner_oracle_id`).

---

## 2. System architecture

### 2.1 Storage

| Store | What lives there | Why |
|---|---|---|
| **SQLite** (`data/edhcut.db`) | All relational/source data: cards, decks, deck_cards, tags, EDHREC stats | Zero-ops, single file, ships as a read-only artifact to the free-tier host, trivially inspectable with pandas/DB Browser |
| **Parquet / scipy `.npz`** (`data/kb/<version>/`) | Derived knowledge base: co-occurrence matrices, embeddings, clusters, text-mined features, archetype definitions | Columnar/sparse formats load fast; matches existing notebook conventions |
| **HTTP cache** (`data/http_cache.sqlite`) | Raw responses from remote sources via `requests-cache` | Free resumability; never re-hit a source for data we already have |

The web app never fetches remote data at runtime — it loads the SQLite file and KB
artifacts read-only. Data refresh is an offline pipeline run locally.

### 2.2 Canonical card identity

**`oracle_id` (Scryfall) is the single card key everywhere.** All sources resolve to it at
ingest time via the `card_names` table: full name, each face of double-faced/split/adventure
cards, punctuation/case-normalized variants. **This is the #1 silent-corruption risk in the
project** — an unresolved name means a card silently missing from co-occurrence counts.
Every ingest logs unresolved names; QA task 5.6 reports them.

### 2.3 Unified schema (SQLite)

```sql
cards(
  oracle_id TEXT PRIMARY KEY, name, mana_cost, cmc REAL, type_line,
  oracle_text, colors, color_identity, keywords,        -- JSON arrays as TEXT
  rarity, edhrec_rank INTEGER, price_usd REAL, game_changer BOOLEAN,
  legal_commander BOOLEAN, can_be_commander BOOLEAN,
  layout, produced_mana, is_land BOOLEAN
)
card_names(name_normalized TEXT PRIMARY KEY, oracle_id TEXT REFERENCES cards)
decks(
  deck_id INTEGER PRIMARY KEY, source TEXT, source_id TEXT, url TEXT,
  commander_oracle_id TEXT, partner_oracle_id TEXT,     -- NULL if no partner
  fetched_at TEXT, views INTEGER, updated_at TEXT,      -- meta for quality filtering
  precon_similarity REAL,                               -- filled by task 5.3 QA: overlap with known precon list
  UNIQUE(source, source_id)
)
deck_cards(deck_id INTEGER, oracle_id TEXT, qty INTEGER, PRIMARY KEY(deck_id, oracle_id))
card_tags(oracle_id TEXT, tag TEXT, source TEXT, PRIMARY KEY(oracle_id, tag, source))
  -- source ∈ {'tagger_bulk', 'textmine', 'manual'}
tag_aliases(alias_normalized TEXT PRIMARY KEY, tag TEXT)  -- e.g. "boardwipe" -> "sweeper"
edhrec_card_stats(
  commander_key TEXT,                                    -- single oracle_id or "id1+id2" for partners
  oracle_id TEXT, inclusion_rate REAL, synergy_score REAL,
  num_decks INTEGER, category TEXT,
  PRIMARY KEY(commander_key, oracle_id)
)
edhrec_themes(                                          -- GLOBAL tag popularity ranking: BOTH
  theme TEXT, kind TEXT, slug TEXT, num_decks INTEGER,    -- /tags/themes ('theme') AND
  fetched_at TEXT, PRIMARY KEY(kind, theme)               -- /tags/typal ('typal') — the fixed
)                                                          -- vocabulary for task 6.6's clustering
edhrec_themes_per_commander(                            -- per-slot tag counts, NOT comparable
  commander_key TEXT, theme TEXT, num_decks INTEGER,      -- across slots (corpus-size-dependent)
  PRIMARY KEY(commander_key, theme)
)
ingest_log(source TEXT, run_at TEXT, items INTEGER, unresolved INTEGER, notes TEXT)
```

### 2.4 Derived knowledge-base artifacts (phase 2 output, phases 3–4 input)

```
data/kb/<version>/
  manifest.json            # build date, source data stats, config used
  card_index.parquet       # row index ↔ oracle_id mapping for all matrices
  cooccur_global.npz       # sparse card×card co-occurrence counts (all decks)
  cooccur_<slot>.npz       # per-commander-slot conditional co-occurrence
  pmi_global.npz           # smoothed PMI / lift association scores
  embeddings.parquet       # card2vec vectors + text-vector columns
  clusters.parquet         # community assignments (global + per-slot)
  roles.parquet            # canonical functional role(s) per card + confidence
  text_features.parquet    # task 6.5: mechanic features mined from oracle text
  power_scores.parquet     # task 6.5: heuristic power-level score per card
  synergy_links.parquet    # task 6.5: explicit produce↔consume synergy pairs
  archetypes.parquet       # per-slot archetype definitions: centroids + role quotas
```

A single loader module (`edhcut/kb.py`) exposes these behind a `KnowledgeBase` class so
phase 3 code never touches file paths directly. Versioned directories make refreshes atomic.

### 2.5 Data flow

```
Scryfall bulk (cards) ──┐
Scryfall bulk (oracle_tags) ─┤→ ingest → SQLite ─→ build_kb.py ─→ kb/<version>/
Archidekt API ──────────┤   (phase 1)              (phase 2)          │
EDHREC JSON ────────────┘                                             ▼
                                                FastAPI (phase 3) ← KnowledgeBase
                                                      │
                                                React SPA (phase 4)
```

### 2.6 Repo layout

```
EDHCut/
  pyproject.toml
  edhcut/
    config.py            # commander slots, paths, rate limits
    http.py              # shared session: requests-cache + tenacity + User-Agent
    db.py                # schema DDL + connection helpers
    ingest/  scryfall.py archidekt.py edhrec.py tagger_bulk.py qa_report.py
    analysis/ cooccurrence.py embeddings.py communities.py roles.py
              textmine.py archetypes.py build_kb.py
    kb.py                # KnowledgeBase loader
    recommend/ parser.py deck_stats.py archetype_classify.py cutter.py suggester.py evaluate.py
    api/ main.py schemas.py
  frontend/              # React + Vite app (phase 4)
  notebooks/             # course EDA/evaluation notebooks 📊
  docs/
    devlog/              # §4 — one entry per completed task + TEMPLATE.md + figures/
    archidekt_api.md edhrec_api.md
  data/                  # gitignored: db, cache, kb artifacts
    fixtures/my_decks/   # the user's real decklists (Kyler, Yoshimaru+Bruse, Yenna, Orysa)
  tests/
```

---

## 3. External libraries & source-access decisions (investigated 2026-07-19)

| Source | Access method | Decision |
|---|---|---|
| **Scryfall cards** | Official bulk `oracle_cards` dump. | ✅ Settled; already prototyped in `Analysis/EDHDataAnalysis.ipynb`. |
| **Scryfall Tagger** | **Official `oracle_tags` bulk data file** (18.2 MB, updated daily, listed at `api.scryfall.com/bulk-data`, documented at scryfall.com/docs/api/tags). Tag objects carry a `taggings` array keyed by `oracle_id`; parent tags require traversing `child_ids`. | ✅ **No scraping needed at all.** The GraphQL/CSRF approach from plan v1 is obsolete — deleted. |
| **Archidekt** | [`mtg_parser`](https://pypi.org/project/mtg_parser/) (PyPI, actively maintained, last release 2026-07-25): multi-site decklist parser, Commander-focused, accepts any `requests`-compatible session — works against `archidekt.com` with no special handling (unlike Moxfield/Deckstats/MTGGoldfish, which need auth or Cloudflare-bypass). Superseded `pyrchidekt` (unmaintained since 2025-10, deck-fetch-only, no search) after investigation in task 5.3-A. | **Split use, not a single dependency for everything**: (1) for the **harvester** (5.3-B), fetch each deck's raw JSON ourselves via `GET /api/decks/{id}/` — `mtg_parser.parse_deck()` only returns cards (no views/updatedAt/format), and empirically excludes the commander card entirely from its output (the "Commander" category isn't marked `includedInDeck`), so we need the raw JSON anyway for metadata + commander identification. (2) `mtg_parser` earns its keep in **task 7.1** (decklist parser): `parse_deck(url, session)` lets that task accept pasted deck URLs from any of its 9 supported sites, not just raw text — worth revisiting 7.1's scope. Deck **search** (`archidekt.com/search/decks?commanderName=...&deckFormat=3&orderBy=-viewCount&page=N`) required reverse-engineering separately — see `docs/archidekt_api.md`; the "clean" `/api/decks/v3/` route is broken for direct external calls (confirmed internal-only — its own pagination `next` field leaks a private IP). |
| **EDHREC** | [`pyedhrec`](https://github.com/stainedhat/pyedhrec) (last commit **2024-02**, effectively unmaintained): scrapes the EDHREC homepage for the current Next.js build ID, then hits `/_next/data/<build_id>/commanders/<slug>.json` routes; also `json.edhrec.com/cards/<name>` for card details. In-memory cache only, no session injection. | **Reference implementation, not a dependency.** The build-ID discovery trick is the right technique and self-heals across EDHREC deploys — reimplement it (~30 lines) inside our own `requests-cache` session in task 5.4. Test pyedhrec live first in 5.4-A: if its parsing still matches today's page shape, copy its response-parsing logic too. |

---

## 4. Development documentation (cross-cutting, mandatory)

The final course submission (modeled on the attached "Reel Patterns" example project) is a
report with per-research-question chapters, each containing: **problem/hypothesis → data
(sources + volumes) → solution & method choices → evaluation criteria → setup → results
with figures → impediments**. We accumulate the raw material for that continuously: every
completed task gets a devlog entry, and the report is later assembled from devlog entries.

`docs/devlog/TEMPLATE.md` (created in task 5.1):

```markdown
# Task <id> — <title>
Date: <date> | Status: done

## Goal & hypothesis
What this step does and what we expected to find/achieve.

## Data
Sources touched; concrete volumes (rows, MB, deck counts, card-pool sizes) before/after;
filtering decisions and their justification.

## Method & design decisions
Approach chosen; alternatives considered and why rejected; parameters and how they were
picked (grids swept, seeds fixed, thresholds and their rationale).

## Evaluation & results
The criteria defined BEFORE looking at output; the numbers (metrics, coverage %,
modularity, accuracy vs spot-check set); determinism/stability checks.

## Interesting findings
Concrete examples worth showing in the final report (e.g. "the Krenko corpus cluster #3 is
recognizably 'sac outlets'"). Save figures to docs/devlog/figures/ and reference them.

## Impediments
What broke or surprised us, how it was resolved, what it cost. (Reel Patterns dedicates a
section to this per chapter — graders value it.)

## Notes for future work
```

Things the Reel Patterns example specifically tracks that we must too: **data volumes per
source** (they report "532 MB Kaggle, 8.12 GB IMDb, 254 KB Spotify"), **numeric quality
values for every claim** (modularity per partition), **justified statistical choices** (they
explain choosing 85% CIs over 95%), **seed/stability checks** for non-deterministic
algorithms, and **honest accounting of manual work and API limits** (their Spotify chapter).

**Standing instruction for every task prompt in this plan:** after the implementation works,
write `docs/devlog/<task-id>-<slug>.md` following the template. If a run produced numbers
(counts, metrics, timings), they go in the entry.

---

## 5. Legal / ToS notes (read before running ingest tasks)

| Source | Status | Notes |
|---|---|---|
| **Scryfall (cards + oracle_tags bulk)** | ✅ Explicitly permitted — bulk files exist precisely for this. | Identifying `User-Agent`, cache ≥24 h. Don't hotlink card images at scale; use their image CDN per guidelines when the frontend needs images. |
| **EDHREC** | ⚠️ No official API; Next.js data routes are undocumented. Check `edhrec.com/robots.txt` + ToS in task 5.4-A before fetching. | Volume is tiny at our scope: one page fetch per commander slot, ≥1 req/2s, cached two weeks, identifying UA. |
| **Archidekt** | ⚠️→✅ Investigated in task 5.3-A (2026-07-29): `robots.txt` permissive; the ToS's generic Acceptable Use Policy boilerplate literally bans "automated... requests, or queries," which *would* block this — **but** Archidekt's co-founder stated on their own forum (still-active account, 7 years ago through 22 months ago) that reading via the API is open/public, that EDHREC uses these exact same public endpoints with no special partnership, and that they only ask for attribution and restraint. Proceeding on that basis was an explicit user call, not an automatic green light — see `docs/archidekt_api.md` for the full quote and reasoning. | ≥1 req/2s (wider margin than the tolerated floor), checkpointed, decks re-cached every two weeks. A few hundred requests total at our scope. |
| **General** | Card names/text are WotC IP under the Fan Content Policy (unmonetized fan projects fine). Ship only *derived* artifacts (scores, clusters), never republished raw EDHREC/Archidekt dumps. | |

---

## 6. Phase 1 — Data acquisition

Riskiest parts: **undocumented Archidekt/EDHREC endpoints** (verify before building), and
**name→oracle_id resolution** (silent corruption if wrong).

### Task 5.1 — Project scaffold + DB schema + HTTP client
Depends on: nothing.

> **Prompt:** Read `EDHCut/EDHCut_PLAN.md` (§1, §2.3, §2.6, §4). Create the
> `EDHCut` package scaffold: `pyproject.toml` (deps: requests, requests-cache,
> tenacity, pandas, pyarrow, tqdm, pyrchidekt), `edhcut/config.py` (dataclass with:
> commander slots as lists of card names —
> `[["Krenko, Mob Boss"], ["Kyler, Sigardian Emissary"], ["Yoshimaru, Ever Faithful",
> "Bruse Tarl, Boorish Herder"], ["Yenna, Redtooth Regent"], ["Orysa, Tide Choreographer"]]`
> — data paths, per-source rate-limit settings, User-Agent
> "EDHCut/0.1 (avivg2001@gmail.com)"), `edhcut/db.py` (idempotent schema creation from
> §2.3, context-manager connection helper), and `edhcut/http.py` (a
> `get_session(source_name)` factory returning a requests-cache CachedSession with
> per-source cache expiry, tenacity retry with exponential backoff on 429/5xx, enforced
> minimum delay between requests, and the UA header). Create `docs/devlog/TEMPLATE.md`
> exactly as specified in plan §4, and `data/fixtures/my_decks/README.md` explaining the
> expected decklist file format. Smoke test: schema created in temp file, all tables exist.
> Finish with a devlog entry per plan §4.

### Task 5.2 — Scryfall ingest + name resolution table
Depends on: 5.1. 📊

> **Prompt:** Read `EDHCut/EDHCut_PLAN.md` (§2.2–2.3). Implement
> `edhcut/ingest/scryfall.py`: download the Scryfall bulk `oracle_cards` file (get the
> download URI from `api.scryfall.com/bulk-data`; port the approach from
> `Analysis/EDHDataAnalysis.ipynb` cell 2 but with plain requests, no scrython), then
> populate `cards` (fields per schema incl. `game_changer`; flatten prices/legalities like
> the notebook; `can_be_commander` = legendary creature or "can be your commander" text)
> and `card_names` with normalized aliases: full name, each `//` face name for multi-face
> layouts, lowercased/punctuation-stripped variants. Filter to commander-legal paper
> cards. Write an `ingest_log` row. Runnable as `python -m edhcut.ingest.scryfall`,
> idempotent. Devlog entry per plan §4 (record: card counts before/after filtering, alias
> counts, download size).

### Task 5.3 — Archidekt deck harvester
Depends on: 5.2. **Do step A before step B.**

> **Prompt (A — investigation):** ✅ Done 2026-07-29 — findings in `docs/archidekt_api.md`
> and devlog `5.3a-archidekt-investigation.md`. Summary: `robots.txt` permissive; ToS
> boilerplate technically restrictive but the founder's own forum statement (quoted in
> `docs/archidekt_api.md`) is the operative guidance, per explicit user decision. Deck-by-id
> (`GET /api/decks/{id}/`) works directly. The "clean" search API
> (`/api/decks/v3/`, `/api/decks/cards/`) is broken for external callers (internal-only —
> confirmed via a leaked private IP in its own pagination). Working search path: fetch the
> public SSR page `archidekt.com/search/decks?commanderName=<name>&deckFormat=3&orderBy=-viewCount&page=<n>`
> and parse the embedded `__NEXT_DATA__` JSON (`props.pageProps.deckResults.{results,count}`)
> — same technique task 5.4 needs for EDHREC. `card.card.oracleCard.uid` in the deck JSON
> **is** the Scryfall `oracle_id` (verified against our own `cards` table) — no fuzzy name
> matching needed for structured deck ingestion. Commander is identified by a card carrying
> the `"Commander"` entry in its `categories` list — confirmed on both single-commander and
> real partner-pair decks (both partners get their own `"Commander"`-tagged entry). No
> dedicated second-*commander* search param exists (tried 4 plausible names, all silently
> ignored), but adding `cardName=<other partner>` alongside `commanderName=<primary partner>`
> does filter server-side to decks containing that card (commander or not) — cuts the
> candidate pool ~23x (1000+ → 43 in testing) vs. `commanderName` alone. Still need a
> client-side check afterward: only ~12% of `cardName`-matched decks actually had that card
> `"Commander"`-tagged (the rest just maindeck it) — presence alone isn't proof of a declared
> partnership, confirmed with a real false-positive.

> **Prompt (B — harvester):** ✅ Done 2026-07-30 — findings in `docs/devlog/5.3b-archidekt-harvester.md`
> and `docs/devlog/5.3c-precon-similarity.md`. Summary: all 5 slots harvested at scale —
> 1,230 decks / 97,911 `deck_cards` rows total (Krenko 301, Kyler 300, Yenna 300, Orysa 29,
> Yoshimaru+Bruse Tarl 300). The maybeboard/sideboard board-membership rule needed 5
> iterations to get right (full postmortem in `docs/archidekt_api.md` §2.1) — final rule:
> only a card's *first* category decides membership, and only the literal `"Sideboard"`
> category is hardcoded-excluded regardless of its own `includedInDeck` flag. The
> Yoshimaru+Bruse Tarl partner slot needed a 3-pass fallback beyond what this prompt
> originally specified (exact pair yields only 4 decks) — added `colors=`-based
> single-partner search once the pair is exhausted, storing fallback decks' *real* commanders
> with a new `decks.slot_key` column to still group them into the pair's corpus; see
> `docs/devlog/5.3b-archidekt-harvester.md` for the full design and the discovery that this
> fallback also surfaced ~26 genuine pair decks the original `cardName=` search had been
> missing. `decks.precon_similarity` is populated via a new `precons`/`precon_cards`
> reference table (179 official Archidekt-curated precons) and generic
> declared-or-alternative-commander matching — this also picked up Krenko (an alternative
> commander in a Secret Lair precon) for free, beyond the Kyler case this prompt named.
>
> Read `EDHCut/EDHCut_PLAN.md` §1, §3 and
> `docs/archidekt_api.md`. Implement `edhcut/ingest/archidekt.py`: for each commander slot in
> config, page through `archidekt.com/search/decks?commanderName=<primary commander>&deckFormat=3&orderBy=-viewCount&page=N`
> — for partner slots, also add `cardName=<other partner>` to the same query to pre-filter
> server-side (cuts the candidate pool ~23x vs. `commanderName` alone) — (parsing
> `__NEXT_DATA__`, incrementing `page` ourselves — do not follow the response's `next` field,
> it's an internal-only URL) to collect candidate deck IDs. For partner slots, still discard
> candidates where the other expected partner isn't `"Commander"`-tagged once fetched (see
> 5.3-A notes — `cardName` only confirms the card is present, not that it's the declared
> partner; ~12% of `cardName` matches were genuine in testing). Keep collecting up to
> `config.decks_per_commander` (default 300; expect far fewer for Orysa, and expect to fetch
> roughly 8x as many partner-slot candidates as you keep — record how many actually
> exist/qualify for each slot). For each kept deck ID, fetch
> `GET /api/decks/{id}/` directly (not via `mtg_parser` — we need metadata
> `mtg_parser.parse_deck()` doesn't expose): pull `viewCount`/`updatedAt`/`createdAt` for the
> `decks` row, resolve cards to `oracle_id` via `card.card.oracleCard.uid` (direct match, not
> name lookup — log anything that doesn't resolve against our `cards` table, e.g. a card that
> failed our commander-legal/paper filter), identify commander(s) via the `"Commander"`
> category, exclude the commander(s) from `deck_cards`, store partner in `partner_oracle_id`,
> and apply the deck's own
> `includedInDeck` category flags to skip maybeboard/sideboard cards (same filtering logic
> `mtg_parser.archidekt.ArchidektDeckParser._parse_deck()` uses — replicate it inline rather
> than calling `mtg_parser` a second time on the same deck). Upsert into `decks` +
> `deck_cards`, checkpointed/resumable by `(source, source_id)`, runnable per-slot. After
> harvesting, compute `decks.precon_similarity` for slots whose commander shipped in a
> precon (Kyler): Jaccard overlap between each deck and the official precon list (fetch the
> precon decklist from Archidekt or store it as a fixture) — we need this for corpus QA and
> the precon test. Devlog entry (deck counts per slot, unresolved-oracle-id stats,
> precon-similarity distribution for Kyler).

### Task 5.4 — EDHREC commander pages
Depends on: 5.2. **Do step A before step B.**

> **Prompt (A — investigation):** ✅ Done 2026-07-31 — findings in `docs/edhrec_api.md`
>
> Read `EDHCut/EDHCut_PLAN.md` §3, §5. Check
> `https://edhrec.com/robots.txt` and ToS. Install pyedhrec in a scratch venv and test
> whether `get_commander_cards()` / `get_high_synergy_cards()` still work today for
> "Krenko, Mob Boss" and for the partner pair (EDHREC has combined pages for partner
> pairs — find the slug format). Read pyedhrec's source
> (`src/edhrec/pyedhrec.py` on GitHub) and document in `docs/edhrec_api.md`: the
> build-ID discovery mechanism (homepage → `/_next/data/<build_id>/...`), the exact
> routes for commander card stats and themes, response shape, and whether pyedhrec's
> parsing still matches. Devlog entry.

> **Prompt (B — fetcher):** ✅ Done 2026-07-31 — findings in `docs/devlog/5.4b-edhrec-fetcher.md`,
> `docs/devlog/5.4c-global-themes.md` (covers both the themes and typal global tables).
> Summary: all 5
> slots harvested — 1,177 `edhrec_card_stats` rows, 285 per-commander theme rows, 0 errors.
> Beyond what this prompt originally specified: per user feedback that per-commander tag
> counts alone aren't a usable input for task 6.6's clustering (not comparable across
> differently-sized commander corpora), also harvests EDHREC's *global*, site-wide tag
> popularity ranking — both of EDHREC's own tag categories, `edhrec.com/tags/themes` (270
> mechanical/archetype tags) **and** `edhrec.com/tags/typal` (131 tribal/creature-type tags,
> added on further user feedback right after the themes-only pass) — into one `edhrec_themes`
> table, keyed `(kind, theme)` and distinguished by a `kind` column. The per-commander table
> this prompt describes below was renamed to `edhrec_themes_per_commander` to free up the
> `edhrec_themes` name, via a data-preserving migration (`db._migrate_legacy_edhrec_themes_table`)
> that ran twice across the two schema changes without losing any already-harvested rows.
>
> Read `docs/edhrec_api.md` and plan §2.3. Implement
> `edhcut/ingest/edhrec.py` with our own session (do NOT depend on pyedhrec —
> reimplement build-ID discovery + the data routes inside our cached/rate-limited
> session, copying pyedhrec's parsing logic where it proved current in step A). For each
> commander slot (single and partner slugs), fetch card stats into `edhrec_card_stats`
> (commander_key = oracle_id or "id1+id2", inclusion rate, synergy score, num_decks,
> category label) and themes into `edhrec_themes`. Resolve names via `card_names`, log
> unresolved, cache 14 days. If the page shape differs from the docs, stop and report —
> do not improvise around it. Devlog entry (rows per slot, unresolved names, anything
> that had drifted from pyedhrec's assumptions).

### Task 5.5 — Scryfall Tagger tags (official bulk)
Depends on: 5.2.

> **Prompt:** ✅ Done 2026-07-31 — findings in `docs/devlog/5.5-tagger-bulk.md`. Summary:
> 376,800 `card_tags` rows, 4,505 tags / 230,624 taggings from the live 5.6 MB bulk file
> (smaller than this prompt's 18 MB estimate), 31,423/31,622 cards (99.4%) tagged. Two
> corrections to this prompt's field-name assumptions found by checking the live shape: tag
> objects use `label`/`slug`, not `name`; there's no `category` field (every object in this
> bulk file is already `type == "oracle"`, so the functional-tags-only filter is automatic).
> Ancestor tags use each tag's own `parent_ids` directly, no `child_ids` inversion needed.
> **Found via user manual spot-check after the initial build**: Scryfall's own `otag:` search
> resolves tag *aliases* (e.g. "boardwipe" → canonical tag "sweeper") — the original ingest
> never read the `aliases` field, so alias lookups silently didn't work. Added `tag_aliases`
> (813 rows, kept separate from `card_tags` so an alias isn't double-counted as a distinct tag
> downstream — user's explicit call).
>
> Read `EDHCut/EDHCut_PLAN.md` §3. Implement
> `edhcut/ingest/tagger_bulk.py`: download the official `oracle_tags` bulk file (find it
> by `type == "oracle_tags"` at `api.scryfall.com/bulk-data`; ~18 MB JSON; docs at
> scryfall.com/docs/api/tags). Each tag object has a name, category, optional `child_ids`,
> and a `taggings` array keyed by `oracle_id`. Insert into `card_tags` with
> source='tagger_bulk'. Handle tag hierarchy: also materialize ancestor tags for each
> tagging (walk parent relationships so a card tagged "boardwipe-creatures" also gets
> "removal"-style ancestors), keeping only functional/oracle categories. Idempotent,
> `ingest_log` row. Devlog entry (tag counts, coverage: % of our card pool with ≥1 tag,
> top-20 most common tags).

### Task 5.6 — QA & coverage report 📊
Depends on: 5.2–5.5.

> **Prompt:** ✅ Done 2026-07-31 — findings in `docs/devlog/5.6-qa-report.md`. Summary: 1,230
> decks, 0 genuine deck-size violations (2 explained edge cases with >2 Archidekt
> `"Commander"`-tagged cards), 376,800 `card_tags` rows, all 4 fixture decklists resolve.
> EDHREC-vs-Archidekt top-10 overlap ranges 2/10 (Yoshimaru+Bruse Tarl, expected given 5.3-B's
> partner-fallback corpus) to 10/10 (Krenko). Along the way, fixed two real bugs the naive
> version of this check itself introduced: a false-positive deck-size mismatch (23 decks,
> from not reconciling against each deck's own unresolved/banned-card count) and a
> comma-in-card-name parsing bug that fragmented names like "Orcrist, Goblin-cleaver" into
> fake half-entries. `notebooks/01_data_overview.ipynb` reuses the same query functions,
> executed end-to-end with real plotly output, 0 errors.
>
> Implement `edhcut/ingest/qa_report.py` producing a markdown report:
> per-slot deck counts and card-pool sizes; deck size sanity (flag ≠ 99/98 cards);
> top-20 unresolved names per source with counts; tag coverage per source; EDHREC vs
> Archidekt agreement spot-check (top-10 inclusion cards per slot side by side); Kyler
> precon-similarity histogram (how much of the corpus is near-precon?); Orysa corpus
> thinness stats. Parse the user's real decklists from `data/fixtures/my_decks/` and
> verify every card resolves. Also create `notebooks/01_data_overview.ipynb` with the
> same stats plus plotly charts — this doubles as the course EDA deliverable. Devlog
> entry.

---

## 7. Phase 2 — Analysis & knowledge base

Riskiest parts: **community-detection parameter sensitivity**, **small-corpus PMI noise**
(mitigate with count thresholds + smoothing), and **precon contamination** (near-identical
precon copies inflating co-occurrence — measured in 5.3/5.6, handled in 6.1).

### Task 6.1 — Co-occurrence & association scores
Depends on: 5.3, 5.6 passing sanity checks.

> ✅ Done 2026-08-02 — findings in `docs/devlog/6.1-cooccurrence.md`. Summary:
> `edhcut/analysis/cooccurrence.py`, 20 new tests (144 total in the package). Card pool:
> 3,095 cards (>= 3 decks) out of 31,622 total. Built global + all 5 per-slot co-occurrence/
> PMI/lift matrices under `data/kb/dev/` (`card_index.parquet` + `.npz` per scope). Precon
> down-weighting used a continuous per-deck novelty weight (full weight up to
> precon_similarity 0.9, linear decay to a 0.1 floor at 1.0) rather than explicit
> near-duplicate grouping — checked live that only 1/1,230 harvested decks actually exceeds
> the 0.9 threshold, so the scheme is real but not load-bearing at today's corpus size.
> **The literal smoothed-PMI formula alone produced useless top-lists** — a rare card that
> happens to co-occur with the query in 100% of its own (small) appearance count ties at a
> constant "ceiling" PMI value regardless of how coincidental that is, so dozens of unrelated
> low-count coincidences drowned out real associations (e.g. Purphoros, God of the Forge's
> raw top-10 was ten unrelated cards tied at the same score). Fixed with a standard Pantel/Lin
> low-count discounting factor applied to PMI (not lift, kept as a plain undiscounted ratio).
> After the fix, `Cultivate`'s #1 global association is `Kodama's Reach` — the exact example
> task 6.2 names as its own success criterion, found here already from co-occurrence alone —
> and Krenko-slot `Skirk Prospector` surfaces a recognizable sac-outlet package
> (Ashnod's Altar, Umbral Mantle, Goblin Sledder, Aggravated Assault).
>
> Read `EDHCut/EDHCut_PLAN.md` §2.4, §7. Implement
> `edhcut/analysis/cooccurrence.py`: build a card index (cards in ≥3 decks), then sparse
> co-occurrence count matrices — global and per-commander-slot — from `deck_cards`.
> **Down-weight near-duplicate decks**: for slots with precon contamination (Kyler),
> weight each deck by novelty (e.g. decks with precon_similarity > 0.9 collectively count
> as a few decks, not hundreds — pick and document a scheme). From counts compute
> smoothed PMI (add-k, zero pairs seen <3 times) and lift. Save `.npz` +
> `card_index.parquet` under `data/kb/dev/`. Sanity CLI: top-20 associated cards by PMI
> for a given card, globally and per-slot. Devlog entry (matrix dims/sparsity, the
> down-weighting scheme chosen and why, eyeball results for Krenko staples).

### Task 6.2 — Card embeddings + similarity index
Depends on: 6.1.

> **Prompt:** Implement `edhcut/analysis/embeddings.py`: gensim Word2Vec over
> decklists-as-sentences (deck = "sentence" of oracle_ids, shuffled a few times per epoch
> since order is meaningless; large window; vector_size≈64, min_count=3). Save to
> `embeddings.parquet`. Add complementary TF-IDF vectors over oracle_text + type_line
> reduced with TruncatedSVD to 64 dims, saved alongside with a column prefix — covers
> cards too rare in decks for good word2vec vectors (important for the Orysa slot).
> Sanity CLI: top-10 nearest neighbors per space ("Cultivate" should find "Kodama's
> Reach"). Devlog entry (params, vocab sizes, neighbor eyeball results).

### Task 6.3 — Synergy communities
Depends on: 6.1.

> **Prompt:** Implement `edhcut/analysis/communities.py`: weighted graph from the PMI
> matrix (edges above threshold), Leiden community detection (python-igraph + leidenalg)
> globally and per-slot. Sweep the resolution parameter over a small grid; report
> modularity + cluster count + size distribution per setting; fix the random seed and
> verify stability across seeds (report agreement). Default to best modularity. Save to
> `clusters.parquet`. Sanity CLI: print each cluster for a slot with top-10 members by
> EDHREC inclusion — clusters should read as recognizable packages (Krenko: "goblin
> payoffs", "sac outlets"). Devlog entry (grid results table, chosen resolution + why,
> seed-stability numbers, named example clusters — prime final-report material).

### Task 6.4 — Functional role classification
Depends on: 5.5.

> **Prompt:** Read `EDHCut/EDHCut_PLAN.md`. Implement `edhcut/analysis/roles.py`
> assigning each pool card one primary + optional secondary role from: `ramp, card_draw,
> spot_removal, board_wipe, counterspell, tutor, recursion, graveyard_hate, protection,
> evasion_enabler, wincon, stax_tax, land, synergy_piece, other`. Layered: (1) tagger_bulk
> tags mapped to roles via an explicit mapping dict; (2) complementary heuristic
> regex/keyword rules on oracle_text + type_line for cards the tags miss (e.g. "search
> your library for .* land" → ramp; "destroy target" on instant/sorcery →
> spot_removal; "destroy all" → board_wipe); (3) default synergy_piece/other. Save to
> `roles.parquet` with per-assignment `source`. Include a labeled spot-check: 60
> well-known cards with expected roles as a test — report accuracy and tagger-vs-heuristic
> agreement. 📊 Devlog entry (accuracy numbers, disagreement examples).

### Task 6.5 — Oracle-text mining: power level, extra tags, synergy detection *(new in v2)*
Depends on: 5.2, 6.4.

> **Prompt:** Read `EDHCut/EDHCut_PLAN.md` §2.4. Implement
> `edhcut/analysis/textmine.py`, our own oracle-text analysis layer with three outputs:
> **(1) Mechanic features** (`text_features.parquet`): parse oracle_text into structured
> produce/consume features — produces: tokens, treasures, counters (+1/+1, charge...),
> card draw, mana, creatures-entering triggers, death triggers, discard, mill, lifegain;
> consumes/rewards: sacrifice outlets, graveyard as resource, discard payoffs,
> token/counter payoffs, "whenever you draw", power-2-or-less references (Kyler!),
> tribal references (extract creature types mentioned). Regex/pattern-based over
> normalized oracle text (replace card's own name with ~, strip reminder text).
> **(2) Extra tags**: write the mined features into `card_tags` with source='textmine'.
> **(3) Power-level score** (`power_scores.parquet`): a documented heuristic 0–10 score
> combining mana efficiency (effect density per cmc), keyword count, tutor/extra-turn/
> free-spell patterns, Scryfall's `game_changer` flag, and edhrec_rank percentile —
> weights in config. Validate against anchors: game_changer cards should average well
> above precon filler; report the distribution and 20 sampled cards with scores for
> eyeballing. **(4) Synergy links** (`synergy_links.parquet`): explicit produce↔consume
> pairs (card A makes tokens, card B sacrifices tokens → link with type 'tokens') —
> distinct from co-occurrence because it can detect synergy for cards never played
> together (this is the cold-start signal for Orysa-like slots). Sanity CLI: given a
> card, print its features, score, and top synergy partners. 📊 Devlog entry (feature
> coverage %, score distribution + anchors check, example synergy links that
> co-occurrence would have missed).

### Task 6.6 — Archetype definitions & role quotas
Depends on: 6.1, 6.4, 6.5 (+5.4 for themes).

> **Prompt:** Implement `edhcut/analysis/archetypes.py`. Per commander slot: represent
> each deck as a feature vector (role counts + mean deck embedding + mechanic-feature
> counts from text_features), k-means (k via silhouette over 2–6) into sub-archetypes,
> label each by most-distinctive high-inclusion cards + closest EDHREC theme where
> available. Store per archetype: centroid, label, role quotas = IQR of role counts
> across its decks. For thin slots (Orysa) where k-means is meaningless, emit a single
> archetype with wide quotas and a `low_confidence` flag. Save `archetypes.parquet`.
> Sanity CLI: archetypes per slot with labels, deck counts, quota tables — check that
> the user's own Yenna fox list would NOT be forced into the meta Yenna archetype
> (classify it by hand with the 6.6 vectors and report). Devlog entry.

### Task 6.7 — KB build pipeline + loader
Depends on: 6.1–6.6.

> **Prompt:** Implement `edhcut/analysis/build_kb.py` — one command running 6.1–6.6 in
> order, writing versioned `data/kb/<YYYYMMDD-n>/` with `manifest.json` (build date, deck
> counts, config, per-step timings). Then `edhcut/kb.py`: a `KnowledgeBase` class
> loading a KB dir lazily, exposing: `similar_cards(oracle_id, space, k)`,
> `pmi(a, b, slot=None)`, `cluster_of(oracle_id, slot=None)`, `roles(oracle_id)`,
> `text_features(oracle_id)`, `power(oracle_id)`, `synergy_partners(oracle_id)`,
> `archetypes(slot)`, `edhrec_stats(slot, oracle_id)`, `card(name_or_id)`. All phase-3
> code consumes only this interface. Unit tests against a tiny fixture KB. Devlog entry.

---

## 8. Phase 3 — Recommendation tool

Riskiest part: **recommendation quality is hard to measure** — task 7.5's reconstruction
evaluation is the honest proxy. Decks whose commander isn't in our roster are rejected
politely with the supported list (v1 behavior).

### Task 7.1 — Decklist parser
Depends on: 5.2.

> **Prompt:** Implement `edhcut/recommend/parser.py`: raw decklist string →
> `{commanders: list[oracle_id] (1 or 2), cards: list[oracle_id], errors, warnings}`.
> Formats: plain "1 Card Name"/"1x"/bare names; Archidekt/Moxfield text exports (category
> headers, `*CMDR*`/"Commander:" markers); MTGO. **Partner pairs must parse** (two
> commander lines). Resolution via `card_names` with rapidfuzz fallback (≥0.9 → accept
> with warning, else error with near-matches). DFC face names. Validations as warnings:
> singleton, color identity vs commander(s) union, 100 cards. Pytest suite including all
> five fixture decks from `data/fixtures/my_decks/` — every card must resolve. Devlog
> entry.

### Task 7.2 — Deck stats & archetype classification
Depends on: 6.7, 7.1.

> **Prompt:** Implement `edhcut/recommend/deck_stats.py` and `archetype_classify.py`
> using only the `KnowledgeBase` interface. Stats: mana curve, color pips vs mana
> sources, role counts vs archetype quotas (surplus/deficit), mean power score +
> power-level histogram (from 6.5), average price, least-played inclusions. Classification:
> deck feature vector as in 6.6, nearest centroid + confidence, dominant synergy clusters
> (% of deck per cluster), and dominant mechanic themes from text_features (should
> identify "foxes/tokens" for the user's Yenna list, not generic Yenna-meta). Plain
> dataclasses ready for JSON. Test against all five fixture decks; record their computed
> stats in the devlog (great report material: "the tool's read on my own decks").

### Task 7.3 — Cut recommender (the core feature)
Depends on: 7.2.

> **Prompt:** Read `EDHCut/EDHCut_PLAN.md` §8. Implement
> `edhcut/recommend/cutter.py`: given parsed deck + candidate card, score every
> non-land existing card as a cut. Weighted components (weights in config):
> (a) **redundancy** — embedding similarity + shared role/mechanic-features with the
> candidate; (b) **low deck synergy** — mean PMI to the rest of the deck
> (slot-conditional where corpus allows) AND absence of explicit synergy_links to deck
> cards (the 6.5 signal — critical for thin slots like Orysa); (c) **low archetype
> fit** — EDHREC inclusion for this commander; (d) **power differential** — prefer
> cutting the lower-power card between near-equivalents; (e) **role surplus** — cutting
> from over-quota roles is cheap, under-quota expensive (unless the candidate fills that
> role). Top-N ranked cuts, each with a human-readable explanation naming dominant
> factors. Never suggest cutting commanders or basic lands. Devlog entry with worked
> examples from the fixture decks (candidate: pick a real card the user considered
> adding).

### Task 7.4 — Swap & addition suggester
Depends on: 7.3.

> **Prompt:** Implement `edhcut/recommend/suggester.py`: (1) **upgrade swaps** — for
> low-synergy deck cards, find in-color-identity commander-legal replacements with the
> same role, higher PMI-to-deck + synergy_links + EDHREC synergy, optional per-card
> budget cap, and a power-band option (suggest within ±2 power of the deck's mean —
> respects the low-power Yenna deck instead of pushing cEDH staples); (2) **missing
> staples** — high inclusion for this slot not in deck; (3) **cluster/mechanic
> completion** — if the deck holds most of a synergy cluster or a produce-without-consume
> imbalance (tokens made, nothing to exploit them), suggest completers. Grouped,
> explained, deduplicated. Devlog entry with fixture-deck examples.

### Task 7.5 — Evaluation harness 📊
Depends on: 7.3, 7.4.

> **Prompt:** Implement `edhcut/recommend/evaluate.py` (add a `holdout` flag in build_kb
> config first, hold out 20% of decks, rebuild KB). Metrics: (1) **leave-one-out
> recovery** — remove a random card from a held-out deck, report rank/hit@k of it among
> the suggester's additions; (2) **intruder detection** — inject an off-archetype card
> into a held-out deck, check the cutter ranks it #1 cut; (3) **precon test (Kyler)** —
> verify suggestions for a near-precon Kyler list are not simply the remaining precon
> cards, and that known precon weak cards get cut before staples; (4) **cold-start
> check (Orysa)** — suggestions must be legal, in-color, and mechanically relevant (via
> synergy_links) even with a thin corpus; (5) archetype classification stability.
> Baselines: random, EDHREC-inclusion-only. Results table + `notebooks/02_evaluation.ipynb`
> — the course evaluation deliverable. Devlog entry (all numbers; this IS the evaluation
> chapter's raw material).

### Task 7.6 — API layer + CLI
Depends on: 7.2–7.4.

> **Prompt:** Implement `edhcut/api/` with FastAPI: `POST /api/analyze`, `POST /api/cut`,
> `POST /api/suggest`, `GET /api/commanders` (supported slots incl. the partner pair),
> `GET /api/cards/search?q=` (autocomplete from card_names). Pydantic schemas, KB loaded
> once at startup, unsupported commander → clean 4xx listing supported slots. Thin CLI
> (`python -m edhcut analyze deck.txt`, `... cut deck.txt "Card Name"`). Integration
> tests using the fixture decks. Devlog entry.

---

## 9. Phase 4 — Deployment

Riskiest part: free-tier memory (512 MB typical) — trivial at 5-slot scale; revisit when
scope grows.

### Task 8.1 — Frontend
Depends on: 7.6.

> **Prompt:** Create `EDHCut/frontend/` — React 18 + Vite + TypeScript against
> the FastAPI OpenAPI contract (proxy to localhost:8000 in vite config). Pages: (1) deck
> input — textarea paste + commander autocomplete (`/api/cards/search`), parse warnings
> inline, partner-pair aware; (2) analysis — mana curve + role-vs-quota charts
> (recharts), archetype badge with confidence, cluster/mechanic composition, power-level
> histogram; (3) cut flow — candidate search → ranked cuts as cards with explanations
> and score-factor bars; (4) suggestions tab — grouped swaps/staples/completions with
> the power-band toggle. Card images via Scryfall image CDN per their guidelines. Clean,
> mobile-ok, React Query for API calls. Devlog entry (screenshots into
> docs/devlog/figures/).

### Task 8.2 — Containerize
Depends on: 8.1.

> **Prompt:** Multi-stage Dockerfile in `EDHCut/`: stage 1 builds the Vite
> frontend; stage 2 slim Python image installing the package, serving the SPA via
> FastAPI StaticFiles, bundling `data/edhcut.db` + latest `data/kb/<version>/`.
> `/api/health` reports KB version. Document local build/run in README. Image < ~1 GB.
> Devlog entry.

### Task 8.3 — Deploy + CI
Depends on: 8.2.

> **Prompt:** Fly.io deployment (fly.toml, 512 MB machine, auto-stop) — first check
> Fly's vs Render's current free allowances and pick the workable one. GitHub Actions:
> push → ruff + pytest; tag → build & deploy. Secrets documented in README. Devlog
> entry.

### Task 8.4 — Data refresh workflow
Depends on: 8.3.

> **Prompt:** Script the refresh loop (`refresh.py` or make target): re-run ingest
> (respecting caches), rebuild KB into a new version dir, run the evaluation harness,
> print a diff report vs previous manifest (deck counts, eval metrics) so regressions
> are visible before deploying. Basic API request logging (endpoint, slot, timing — no
> decklist contents). Manual/local for now; README notes how to move to a scheduled
> GitHub Action later. Devlog entry.

---

## 10. Recommended starting point (build this first)

**Vertical slice: Krenko, end to end, ugly.** Validates the two real risks: *can we get
Archidekt deck data reliably*, and *does co-occurrence signal alone produce sane cut
suggestions*.

Order (≈ tasks 5.1 → 5.2 → 5.3 → mini-6.1 → throwaway script):

1. Task 5.1 (scaffold) and 5.2 (Scryfall ingest — mostly porting the notebook).
2. Task 5.3-A (Archidekt + pyrchidekt investigation) — **the go/no-go moment.**
3. Task 5.3-B for Krenko only, ~300 decks.
4. Minimal global co-occurrence + PMI for that corpus (cut-down 6.1, no precon weighting).
5. Throwaway `scripts/spike_cut.py`: Krenko decklist + candidate card → rank cuts by
   (low mean PMI to deck) + (low corpus inclusion rate). Print top 10 with scores.

Judge the output against Magic intuition. Half-sensible → everything in phases 2–3 is
uplift on a working signal. Nonsense → debug the signal (deck quality filtering? more
decks? smoothing?) before building anything on top.

**Spike prompt to feed Sonnet:**

> Read `EDHCut/EDHCut_PLAN.md` §10 and implement the vertical slice exactly as ordered
> there, stopping after each numbered step to show me results. Steps 1–3 follow tasks
> 5.1, 5.2, 5.3 in the plan (single commander: Krenko, Mob Boss; 300 decks). Step 4:
> global co-occurrence counts + smoothed PMI for that corpus only, saved as npz. Step 5:
> `scripts/spike_cut.py` taking a decklist file + candidate card name, scoring each deck
> card by weighted (low mean PMI to rest of deck) + (low corpus inclusion rate), printing
> top-10 cut suggestions with both component scores. No API, no frontend, no polish.
> Write devlog entries per plan §4 as each step completes.

---

## Appendix A — Raw investigation notes (2026-07-19)

Backing detail for §3's verdicts, captured so implementation tasks don't need to
re-derive it from scratch. This is the working-session record; promote anything load-bearing
into the relevant task prompt if it turns out to matter more than expected.

**Scryfall bulk-data inventory** (`api.scryfall.com/bulk-data`, snapshot 2026-07-19,
updates continuously — re-check `updated_at` at ingest time, don't hardcode sizes):

| type | size | notes |
|---|---|---|
| `oracle_cards` | 179.8 MB | what task 5.2 downloads |
| `oracle_tags` | 18.2 MB | what task 5.5 downloads — functional Tagger tags, `taggings[].oracle_id` |
| `art_tags` | 40.6 MB | illustration tags, keyed by `illustration_id` — **not needed**, we only want functional tags |
| `rulings` | 25.9 MB | keyed by `oracle_id` — not currently used by any task, potential future feature (surface relevant rulings in card explanations) |
| `unique_artwork` | 264.4 MB | not used |
| `default_cards` | 557.6 MB | not used, `oracle_cards` is the right grain (one row per card, not per printing) |
| `all_cards` | 2572.0 MB | not used |

**pyedhrec internals** (github.com/stainedhat/pyedhrec, last commit 2024-02-03 "Add typing"):
- Source lives at `src/edhrec/pyedhrec.py` (note: package dir is `edhrec`, PyPI name is `pyedhrec`).
- Constants: `base_url = "https://edhrec.com"`, `_json_base_url = "https://json.edhrec.com/cards"`, `_api_base_url = f"{base_url}/api"`.
- Build-ID trick: `get_build_id()` fetches the homepage and scrapes the current Next.js
  build id out of it; `_build_nextjs_uri()` then constructs
  `{base_url}/_next/data/{build_id}/{endpoint}/{formatted_card_name}` for commander
  pages (this is the mechanism task 5.4-A must verify still works and 5.4-B reimplements).
- `get_card_details()` hits `{_json_base_url}/{formatted_card_name}` directly (no build id needed) — simpler path, worth using as-is for per-card lookups.
- Card name formatting (`format_card_name`) lowercases and slugifies — check its exact
  rules against a few of our multi-face/punctuation card names (e.g. "Yoshimaru, Ever
  Faithful") before trusting it blindly.
- Caching is in-memory only, 24h TTL, no way to inject a custom session — confirmed we
  cannot reuse its HTTP layer, only its URL-construction logic, matching the plan's
  "reference implementation, not a dependency" call.

**pyrchidekt internals** (github.com/linkian209/pyrchidekt, MIT, v2.2.0 released
2025-10-08, 28 commits, 19 stars):
- Entry point is `getDeckById()`, returning dataclasses with categories → cards →
  quantity + oracle card info.
- No search/listing endpoint in the package — confirmed gap that task 5.3-A's raw
  `requests` probe against `archidekt.com/api/decks/v3/?...` must fill.
- No documented rate limiting or auth requirements found in the repo; treat as
  unspecified and stay conservative (~1 req/s) per plan §5.

**Orysa identity check** — confirmed via live Scryfall search (`api.scryfall.com/cards/search?q=orysa`):
exactly one match, "Orysa, Tide Choreographer", Legendary Creature — Merfolk Bard, mono-U,
*Secrets of Strixhaven*, released 2026-04-24, `edhrec_rank` 19627 at query time (thin
corpus, as expected for the cold-start test case).

**Devlog template provenance** — §4's `docs/devlog/TEMPLATE.md` structure was reverse-
engineered from a previous-year course submission PDF ("Reel Patterns — A Deep Dive into
the Data Behind the Scenes", Kimhi/Forshmit/Tuval), which the user has locally in their
Downloads folder — **that PDF is not part of this repo and won't travel with git sync.**
Nothing from it is needed going forward beyond what's already distilled into §4 (the
per-chapter structure: hypothesis → data volumes → method + rejected alternatives →
pre-declared evaluation criteria → numeric results → impediments); if a future session
wants to re-examine the source example itself, it needs to be re-supplied on whatever
device that session runs on.

**Environment note (this device only, not portable via git):** the existing
`Analysis/EDHDataAnalysis.ipynb` notebook (pre-EDHCut Scryfall exploration) was run
against a local Jupyter server at `http://localhost:8888` with token `JupTokEDHCut`,
configured in a local `.mcp.json`. That server isn't part of the repo and needs to be
started fresh on any other device before that notebook can be driven interactively again.
