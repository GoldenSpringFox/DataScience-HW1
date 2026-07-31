# EDHREC API — investigation notes (task 5.4-A)

Investigated live 2026-07-31, in a scratch venv (`pip install pyedhrec`, not the shared repo
venv — this package is not a dependency of `edhcut`, task 5.4-B reimplements what it needs).
This is the working-session record backing plan §3/§5's EDHREC row and task 5.4-B's
implementation approach.

## 1. Legal basis — read this before touching task 5.4-B

**`robots.txt`** (`edhrec.com/robots.txt`): permissive.
```
User-agent: *
Disallow: /articles/preview/
Disallow: /articles/search/
Disallow: /deckpreview/
Disallow: /puzzlebookvegas/
```
None of these overlap with commander pages or the `/_next/data/.../commanders/...` route we
use.

**Terms of Service** (`edhrec.com/terms`, §3.1 Acceptable Use Policy): same generic
boilerplate shape as Archidekt's (see `docs/archidekt_api.md` §1) — states you agree not to:

> "use software or automated agents or scripts to...generate automated searches, requests, or
> queries to the Site."

Read literally, this prohibits exactly what the fetcher does. **Unlike Archidekt, no
equivalent resolving evidence exists here** — no team/founder statement anywhere (checked the
FAQ page too: it explains EDHREC *itself* pulls deck data from Archidekt/Moxfield/Scryfall,
but says nothing about third parties pulling from EDHREC). What does exist is a
long-running, openly documented ecosystem of unofficial tools hitting these same undocumented
endpoints — `pyedhrec` (the library this doc tests), plus several independent scrapers found
searching GitHub (`edhrec_scraper`, `EDHREC-Analysis`, `donaldpminer/edhrec`, `edhrec-mcp`,
etc.) — a tolerated norm, not a stated permission.

**Conclusion we're operating under** (explicit user decision after being presented with the
conflict and the Archidekt comparison, not an automatic override of the ToS text): proceed,
on the basis that our actual footprint is tiny and clearly distinguishable from the kind of
abuse the clause is aimed at —
- One page fetch per commander slot (5 slots; partner pairs are a single combined page, see
  §2), run offline as part of a periodic pipeline refresh — **not** a live/runtime scraper.
  The deployed app never calls EDHREC (plan §2.1: "The web app never fetches remote data at
  runtime").
- Cache 14 days (plan §5), so a re-run within that window makes zero new requests.
- Rate-limited, identifying `User-Agent` (reuse `edhcut/http.py`'s existing
  `RateLimitedSession` conventions, same as Archidekt) — **do not** copy `pyedhrec`'s own
  behavior of rotating through a static list of spoofed real-browser User-Agent strings
  (`pyedhrec/utils.py::get_random_ua`); that's a deliberate impersonation pattern we should
  not reproduce.
- Per plan §5's "General" row: ship only *derived* artifacts (our own computed scores/
  clusters) — never republish EDHREC's raw scraped numbers.
- If EDHREC's posture is ever stated less permissively (a takedown request, a documented
  block), that changes the calculus and should be revisited, not silently worked around —
  same standing rule as Archidekt's.

## 2. Confirmed endpoints

### 2.1 Build-ID discovery

```
GET https://edhrec.com/
```
Response HTML embeds `<script id="__NEXT_DATA__" type="application/json">{"buildId": "...", ...}</script>`
— parse with a regex, `json.loads` the captured group, read `.buildId`. Verified live;
current build id at test time was `K2qEK1aTLshG5gVnpr2lK` — **this drifts on every EDHREC
deploy, never hardcode it**, always re-fetch (or fall back to a cached last-known value, which
is what `pyedhrec` does with its `default_build_id` constant — worth keeping that fallback
pattern in 5.4-B).

### 2.2 Commander page data route

```
GET https://edhrec.com/_next/data/<build_id>/commanders/<slug>.json?commanderName=<slug>
```

**Slug format for a single commander**: lowercase, spaces → hyphens, strip apostrophes and
commas (`pyedhrec.format_card_name`). E.g. `"Krenko, Mob Boss"` → `krenko-mob-boss`.

**Slug format for a partner pair — not documented anywhere, worked out live**:
`format_card_name()` each partner's name **individually**, then join the two slugs directly
with a hyphen, **sorted alphabetically** (not `"+"`, not `"&"` — naively formatting a
`"name1 + name2"` string, which is what `pyedhrec.format_card_name` does with no special
casing, produces a literal `+` in the slug and 404s). Confirmed by fetching
`edhrec.com/partners/yoshimaru-ever-faithful`, which lists Bruse Tarl as a partner option
(222 decks) with an implied combined-page link, then testing the constructed URL directly:

```
GET https://edhrec.com/_next/data/<build_id>/commanders/bruse-tarl-boorish-herder-yoshimaru-ever-faithful.json?commanderName=bruse-tarl-boorish-herder-yoshimaru-ever-faithful
```
→ 200, full commander-page payload for the pair (`b` sorts before `y`, matching the observed
ordering). **5.4-B should build partner slugs as
`"-".join(sorted(format_card_name(n) for n in partner_names))`.**

### 2.3 Response shape

Unwrap with `response["pageProps"]["data"]` (`pyedhrec._get_nextjs_data`). Top-level keys on
the unwrapped payload: `creature`, `instant`, `sorcery`, `artifact`, `enchantment`, `battle`,
`planeswalker`, `land`, `basic`, `nonbasic`, `similar`, `bracket_counts`, `budget_counts`,
`tag_counts`, `savedate_counts`, `header`, `panels`, `description`, `container`.

**Card stats** live in `container.json_dict`:
- `container.json_dict.card.num_decks` — total decks EDHREC analyzed for this commander slot
  (Krenko: 42,669; Yoshimaru+Bruse Tarl: 222). This equals `potential_decks` on every card
  entry below — confirmed equal in both test cases.
- `container.json_dict.cardlists` — list of `{tag, header, cardviews: [...]}`. Confirmed tags
  present for Krenko: `newcards`, `highsynergycards`, `topcards`, `gamechangers`, `creatures`,
  `instants`, `sorceries`, `utilityartifacts`, `enchantments`, `utilitylands`,
  `manaartifacts`, `lands` (the partner pair additionally had `planeswalkers` — the tag set
  present depends on what card types actually appear at meaningful frequency for that slot,
  don't assume a fixed list).
- Each `cardviews` entry:
  ```json
  {
    "id": "5bac033c-dc4e-40a0-b103-4892e4b50249",
    "name": "Goblin Warchief",
    "sanitized": "goblin-warchief",
    "slug": "goblin-warchief",
    "url": "/cards/goblin-warchief",
    "synergy": 0.719099058427737,
    "num_decks": 37352,
    "potential_decks": 42669,
    "trend_zscore": 0.46240219327941384
  }
  ```
  Maps directly onto the plan's `edhrec_card_stats` schema: `synergy` → `synergy_score`;
  `num_decks / potential_decks` → `inclusion_rate`; `num_decks` → `num_decks`; the cardlist's
  `tag` (e.g. `"highsynergycards"`) → `category`.
- **No `oracle_id` anywhere on a cardview entry** — only `name`/`slug`/EDHREC's own `id`
  (a different UUID namespace from Scryfall's, spot-checked). Unlike Archidekt
  (`docs/archidekt_api.md` §2.1), **resolution must go through `card_names`** (task 5.2's
  alias table), same as any free-text source. Log unresolved names per the plan's instruction.
- **Duplicate-row risk for 5.4-B**: a single card commonly appears in more than one cardlist
  (e.g. a creature that's both `"topcards"` and `"creatures"`, or also `"highsynergycards"`).
  `edhrec_card_stats`'s primary key is `(commander_key, oracle_id)` — one row per card per
  slot — so 5.4-B needs an explicit precedence rule for which cardlist's `category`/`synergy`
  value wins when a card shows up more than once. **Not resolved here** — flagged as an open
  design decision for 5.4-B, don't improvise it silently; if it's non-obvious which list
  should win, ask.

**Theme stats** live in `panels.taglinks` (equivalently `tag_counts`, same data two ways):
```json
{"count": 6368, "slug": "goblins", "value": "Goblins"}
```
Maps onto `edhrec_themes_per_commander`: `value` → `theme`, `count` → `num_decks`. Confirmed
present and correctly shaped for both the single commander and the partner pair test. (Not to
be confused with the *global* theme ranking, §3 below — same field shape, different page and
different table.)

### 2.4 pyedhrec parsing status

Tested live against both `get_commander_data()` and the higher-level
`get_commander_cards()` / `get_high_synergy_cards()` wrappers for **"Krenko, Mob Boss"** and
the constructed partner-pair slug — **all still match the current site shape**, no drift
found. `get_high_synergy_cards()`'s tag filter (`"highsynergycards"`) is a genuine cardlist
tag, confirmed present in both test cases.

Other `pyedhrec` methods exist (`get_commanders_average_deck`, `get_commander_decks`,
`get_card_combos`, `get_card_details`, `get_card_list`) but aren't needed for the plan's
`edhrec_card_stats`/`edhrec_themes_per_commander` schema — not reimplemented in 5.4-B unless a
later task needs them.

## 3. Global tag popularity routes (tasks 5.4-C / 5.4-D)

EDHREC's own tag browser splits into two kinds of site-wide, popularity-ranked list,
independent of any one commander — both feed the single `edhrec_themes` table:
- `edhrec.com/tags/themes` — mechanical/archetype tags (e.g. "Aristocrats", "Tokens").
  Added after 5.4-B on user feedback: the per-commander tag counts alone (§2.3) aren't a
  usable, cross-comparable vocabulary for task 6.6's clustering — this route is.
- `edhrec.com/tags/typal` — tribal/creature-type tags (e.g. "Goblins", "Elves"). Added
  immediately after 5.4-C on further user feedback: EDHREC treats "themes" and "typal" as two
  distinct tag categories on its own site (confirmed by browsing both pages), and the first
  pass only covered one of them.

Same Next.js data-route mechanism as a commander page for both:
```
GET https://edhrec.com/_next/data/<build_id>/tags/themes.json
GET https://edhrec.com/_next/data/<build_id>/tags/typal.json
```
but the unwrapped payload (`response["pageProps"]["data"]`) has **no `panels` key** on either
— only `container` — so `fetch_commander_data`'s validation doesn't directly apply; a
kind-parameterized `fetch_global_tag_page(session, build_id, kind)` handles both.

`container.json_dict.cardlists` holds exactly **one** cardlist on each page, tagged
`"tagsbypopularitysort"`, already sorted descending by deck count — no pagination, the whole
ranking is one response per page. Verified live:
- **Themes**: 270 entries, `num_decks` ranging 196,845 ("Tokens") down to 1 ("Value Vintage").
- **Typal**: 131 entries, `num_decks` ranging 26,650 ("Dragons") down to 4 ("Nobles").
- **Zero name overlap** between the two lists (checked directly, set-intersection is empty) —
  but `edhrec_themes`'s primary key is still `(kind, theme)`, not `theme` alone, since nothing
  guarantees that stays true as EDHREC adds tags over time.

Each entry, same shape on both pages:
```json
{
  "id": "1d8b007b-3169-4ee3-80c7-781fc096fc7a",
  "name": "Tokens",
  "sanitized": "skullclamp",
  "slug": "skullclamp",
  "url": "/tags/tokens",
  "num_decks": 196845
}
```
**Trap**: `id`/`sanitized`/`slug` here describe a *representative card* EDHREC shows as that
tag's thumbnail (e.g. "Skullclamp" illustrates "Tokens") — not the tag itself, despite being
the same field names `extract_card_stats` reads for actual cards elsewhere on the site. The
tag's own identity is `name` (display name) and `url` (its own `/tags/<slug>` page, parsed
into the `edhrec_themes.slug` column — a stable identifier independent of `name`, which is
display text and less guaranteed to be a safe key).

## 4. Rate limiting / caching plan for 5.4-B/5.4-C/5.4-D

Reuse `edhcut/http.py`'s existing `RateLimitedSession` + `requests-cache` conventions
(same pattern as `edhcut/ingest/archidekt.py`), with an identifying `User-Agent` — **not**
`pyedhrec`'s rotating fake-browser UA list. Total live request volume for a full refresh: one
build-ID fetch (shared/cached across all slots) + two global-tag-page fetches (§3, themes and
typal, also shared/cached across slots — not scoped to any one of them) + one commander-page
fetch per slot (5 slots, partner pairs are a single combined request) — negligible, and zero
additional requests on any re-run within the 14-day cache window.
