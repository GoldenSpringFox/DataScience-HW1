# Archidekt API — investigation notes (task 5.3-A)

Investigated live 2026-07-29. This is the working-session record backing plan §3/§5's
Archidekt row and task 5.3-B's implementation approach. Re-verify against a real deck before
trusting any of this blindly in a future session — Archidekt's own public API surface has
already drifted once (see "Search" below) since the plan's original 2026-07-19 assumptions.

## 1. Legal basis — read this before touching task 5.3-B

**`robots.txt`** (`archidekt.com/robots.txt`): permissive.
```
User-agent: *
Disallow: /partialCompare
Disallow: /playtester-v2/
Disallow: /sandbox?
```
None of these overlap with deck pages or the search/deck-detail endpoints we use.

**Terms of Service** (`archidekt.com/terms`, effective 2018-09-07): generic boilerplate
Acceptable Use Policy (reads like a copy-pasted SaaS ToS template — e.g. "Like any other
website, uses 'cookies'" is a grammatical tell) states you agree not to:

> "use software or automated agents or scripts to produce multiple accounts on the Site, or
> to generate automated searches, requests, or queries to the Site"

Read literally, this prohibits exactly what the harvester does. **However**, Archidekt's
co-founder ("michael", tagged Alpha User) addressed this directly, unprompted, on their own
forum — [thread from 7 years ago](https://archidekt.com/forum/thread/40353), with the same
account still actively answering API questions as recently as
[22 months ago](https://archidekt.com/forum/thread/9428913):

> "our API is open and public (as far as reading is concerned)... **EDHRec already pulls out
> data into their site by using these requests** and we hope that others will as well to
> increase the amount of cool stuff available for Magic players... If you do pull our data
> we ask that you link back to Archidekt in some way as well if you're posting that data
> publicly... if it ever gets to a point where we're getting constantly hammered by requests
> that aren't ours and it's causing issues -- we'll have no choice but to lock down the API
> entirely for ourselves and people who we are officially affiliated with."

**Conclusions we're operating under** (this was an explicit user decision after being
presented with the conflict, not an automatic override of the ToS text):
- **No special EDHREC partnership exists.** EDHREC uses the same public, unauthenticated
  endpoints documented below. There is no allowlist or API key system to apply for.
- Read-only, respectful-volume access is explicitly welcomed by the site's own operator.
- We should: stay well under the "hammered" threshold (our config already runs Archidekt at
  2s/request, wider than any documented floor), and credit Archidekt somewhere in the
  eventual product (frontend attribution — flag for task 8.1).
- If Archidekt ever tightens this stance or the API genuinely locks down, that changes the
  calculus and should be revisited, not silently worked around.

## 2. Confirmed endpoints

### 2.1 Deck by ID — works directly, public

```
GET https://archidekt.com/api/decks/{deck_id}/
```
Returns the full deck JSON. No auth needed. Verified live against deck `24392195`
("Liesa, Shroud of Dusk").

Top-level fields we care about: `id`, `name`, `viewCount`, `createdAt`, `updatedAt`,
`deckFormat` (Commander = `3`), `categories` (list), `cards` (list). Also present but unused:
`owner`, `private`, `unlisted`, `theorycrafted`, `deckTags`, `edhBracket`, etc.

**`categories`** — one entry per user-defined or built-in category, e.g.:
```json
{"id": 310342581, "name": "Extort", "isPremier": false, "includedInDeck": true, "includedInPrice": true}
```
`includedInDeck: false` categories (e.g. "Upgrades" was `false` on the sample deck — deck
owners use categories for all kinds of personal organization, not just board membership) mark
content that's part of the deck's organizational scheme but not counted toward the actual
100-card list. A confirmed real deck's category list included both `"Maybeboard"` and
`"Sideboard"` as available category names, alongside a built-in `"Commander"` category.

**`cards`** — one entry per distinct printing/quantity group. Relevant shape:
```json
{
  "categories": ["Commander"],
  "quantity": 1,
  "card": {
    "collectorNumber": "CMR-286",
    "edition": {"editioncode": "plst", "editionname": "The List", ...},
    "oracleCard": {
      "uid": "2270daaf-252d-40c4-bed5-bd46ae15f041",
      "name": "Liesa, Shroud of Dusk",
      "cmc": 5, "colorIdentity": ["White","Black"], "edhrecRank": 5574,
      "legalities": {"commander": "legal", ...}
    }
  }
}
```

**Card identity — the important discovery**: `card.card.oracleCard.uid` is **exactly the
Scryfall `oracle_id`**. Verified directly: fetched the commander of deck `24392195`, got
`uid = "2270daaf-252d-40c4-bed5-bd46ae15f041"`, queried our own `cards` table with that exact
string as `oracle_id`, got back `Liesa, Shroud of Dusk` — an exact match. **Structured deck
ingestion from Archidekt needs zero fuzzy name matching** — just `oracle_id` lookup against
our `cards` table (name-based `card_names` resolution is still needed for task 7.1's
free-text decklist parser, which handles pasted text with no structured card IDs).

**Commander identification**: the card(s) carrying `"Commander"` in their `categories` list.
Confirmed on both a single-commander deck and a real partner-pair deck — **both partners get
their own `"Commander"`-tagged entry**, verified against
[archidekt.com/decks/7019846](https://archidekt.com/decks/7019846/50_the_ever_faithful_is_finaly_at_peace)
("50- The Ever Faithful is finaly at peace"): both `Yoshimaru, Ever Faithful` and
`Bruse Tarl, Boorish Herder` carry `categories: ["Commander"]`, and both `oracleCard.uid`s
resolve correctly against our `cards` table.

**Important**: a card matching a known commander's `oracle_id` is not by itself proof it's
*the* commander of that deck — Partner-having legendaries are legal ordinary 99-card
inclusions in unrelated decks too. Confirmed this empirically: deck `4443632` ("Jeskai
Yoshimaru Blink B4") contains `Bruse Tarl, Boorish Herder`, but tagged only
`categories: ["Creature"]` — a maindeck inclusion, not a declared partner. **Always check for
the `"Commander"` category specifically**, not just card presence.

**Board/maybeboard exclusion — went through five iterations before landing on the current
rule, documenting all of it because the reasoning matters for anyone tempted to "fix" this
again:**

1. **Started with**: a card belongs to the deck if it has no categories, or shares one with
   the deck's `includedInDeck: true` set — exactly what
   `mtg_parser.archidekt.ArchidektDeckParser._parse_deck()` does (see §3).
2. **Broke on token-tracking entries**: Archidekt lets a deck carry reference entries for
   tokens it can *produce* (e.g. a deck with "Agate Instigator" the creature also had a
   separate "Agate Instigator" *token* entry — confusingly matching name, different
   `oracle_id`). Confirmed real example: deck `2029118` had this token entry tagged
   `categories: ["Maybeboard", "Creature"]` — since `"Creature"` was `includedInDeck: true`,
   rule 1 wrongly counted it (and since the token's `oracle_id` is correctly
   `commander: "not_legal"`, it also showed as a false "illegal card"). **Fix**: check
   `card["card"]["oracleCard"]["layout"] == "token"` and exclude unconditionally — confirmed
   reliable (`layout: "token"`, `types: ["Creature", "Token"]`) regardless of category
   tagging.
3. **Broke on real decks with `size` mismatches even after the token fix.** Tried gating on
   the search-listing's own `size` field instead of reconstructing from categories at all —
   **this field can be stale**: for deck `18694948`, the search listing said `size: 101`
   while the deck's own page (fetched independently, same moment) displayed "Size: 100".
   Confirmed via a full recursive search of the deck page's `__NEXT_DATA__` payload that no
   literal `size`/count field exists anywhere in the reachable data — the page computes and
   displays it client-side from the same `cards`/`categories` data we already have, via JS
   logic not worth reverse-engineering from a minified bundle.
4. **Iteration 4, since superseded**: "any category whose name contains `maybeboard`/
   `sideboard` is always excluded, regardless of `includedInDeck` or co-tagging" — motivated
   by deck `14146865` where several Maybeboard-tagged cards also carried a legitimate
   category (e.g. `["Maybeboard", "Ramp"]`). Overcorrected: verified against 3 real
   "wrong_size"-flagged decks the user manually confirmed were valid 100-card decks,
   `18694948` came out exactly right (100) but `2029118` → 96 and `14146865` → 91, both
   *short*.
5. **Iteration 5, correct: only a card's *first* category decides board membership, and
   "Sideboard" — not "Maybeboard" — is the one hardcoded exception.** Root cause of
   iteration 4's shortfall, found by diffing deck `2029118`'s actual 99-card list (user
   pasted it directly) against our output: the 4 missing cards were all tagged like
   `["Goblin", "Maybeboard"]` — a real category *first*, "Maybeboard" only *second* — and
   Archidekt counts the card as in-deck anyway. Later categories are purely organizational
   tags a user layers on top; they don't affect board membership. Fixed to check only
   `categories[0]`, still hardcoding "maybeboard"/"sideboard" names out regardless of flag —
   re-verified against `2029118`, `14146865`, `18694948`, all exactly 99.
   - **Broke again on deck `3085756`** ("(WIP) no krenkombos" — user asked why it was
     flagged `wrong_size`, got 79/99). This deck has **both** "Maybeboard" and "Sideboard"
     marked `includedInDeck: true`, and uses "Maybeboard" as the *first* category for 20
     genuinely-included cards (its own displayed "Size: 100" and its "Quantity of
     Categories" stat widget both count them). Hardcoding "maybeboard" out unconditionally
     wrongly dropped all 20. Comparing against deck `18694948` (where `"Sideboard"` alone,
     as sole/first category on "Shared Animosity", was `includedInDeck: true` yet genuinely
     excluded from that deck's own stats widget) revealed the actual distinction: **only the
     literal built-in `"Sideboard"` category is hardcoded-excluded regardless of its flag**
     (Archidekt's API always seems to report `includedInDeck: true` for it, same as it
     always does for `"Commander"` — a metadata quirk, not the real membership signal).
     `"Maybeboard"` is *not* special-cased at all — it's an ordinary category like any other,
     and its own `includedInDeck` flag (owner-configurable, varies deck to deck) is
     authoritative. Custom sub-board names like `"Maybeboard - Goblin"` need no special
     handling either — they're excluded whenever their own flag is `false`, which is the
     normal case, via the same plain flag check.
   - **Final rule**: for each card, take `categories[0]`. If it's exactly `"Sideboard"`,
     exclude unconditionally. Otherwise include iff `categories[0]` is in the deck's
     `includedInDeck: true` set (or the card has no categories at all). Re-verified against
     all 4 previously-flagged decks (`2029118`, `14146865`, `18694948`, `3085756`) — all four
     now compute to exactly 99 non-commander cards (100 total), matching Archidekt's own
     displayed size. Implemented in `edhcut/ingest/archidekt.py::included_cards()`.

### 2.2 Deck search — the public REST route is broken; use the SSR page instead

The route the plan originally assumed (`archidekt.com/api/decks/v3/?...`, also seen written
as `/api/decks/cards/?...` in old forum posts) **does not work for direct external calls**:

```
GET https://archidekt.com/api/decks/cards/?orderBy=-createdAt&owner=Wildcard&ownerexact=true&pageSize=5
→ 404, body: "Client Unavailable: You are requesting client routes from the api. Use a
   react server with the archidekt-client to develop locally. If you are seeing this in
   production, the load balancer is not correctly routing."
```

This is not new breakage from our side — a forum user reported the identical error
[6 months ago](https://archidekt.com/forum/thread/16962481). Confirmed why: Archidekt's
search page is server-rendered (Next.js), and it calls this route from their own internal
network, not the public internet. Proof: the working workaround's paginated response literally
embeds the internal call it made —
```json
"next": "http://10.142.0.44/api/decks/v3/?commanderName=Krenko%2C+Mob+Boss&deckFormat=3&orderBy=-viewCount&page=2"
```
`10.142.0.44` is a private IP — unreachable from outside their network. **Don't try to hit
`/api/decks/v3/` or `/api/decks/cards/` directly; don't follow a response's `next` field.**

**Working approach**: fetch the public search page itself and parse its embedded Next.js data
payload.

```
GET https://archidekt.com/search/decks?commanderName=<url-encoded name>&deckFormat=3&orderBy=-viewCount&page=<n>
```

- `commanderName` — exact card name, URL-encoded (verified: `Krenko, Mob Boss` → returns
  exclusively mono-red goblin-tribal decks named things like "Krenko (Gobos)", "Gobs of
  Goblins" — correctly filtered).
- `deckFormat=3` — Commander format (matches the `deckFormat` field on real Commander decks
  fetched via §2.1).
- `orderBy=-viewCount` — descending by views. (`-createdAt`/`-updatedAt` presumably also
  work, matching Next.js/DRF-style ordering conventions seen elsewhere on the site, but only
  `-viewCount` was actually tested.)
- `page=<n>` — **paginate by incrementing this yourself** on the same public URL. Do not use
  the response's `next` field (see above).
- **No dedicated second-*commander* filter exists.** Tried four plausible param names
  (`commanderName2`, `partnerName`, `commanders`, `secondCommanderName`) — all silently
  ignored; result sets were byte-for-byte identical to searching `commanderName` alone
  (unrecognized params don't error, they're just dropped).
- **But `cardName` works, and narrows the field a lot.** Add
  `cardName=<other partner name>` alongside `commanderName=<primary partner name>` to filter
  to decks that contain *that card somewhere* (commander or not) in addition to the searched
  commander. Verified: `commanderName=Yoshimaru, Ever Faithful` alone → 1000+ (capped) decks;
  adding `cardName=Bruse Tarl, Boorish Herder` → an exact, non-capped **43** decks.
  `cardName` is not commander-specific, though — it still doesn't tell you whether that card
  is the *declared partner* or just an ordinary 99-card inclusion. Checked all 43: only
  **5 (≈12%)** were actually `"Commander"`-tagged for Bruse Tarl; the other 38 mostly
  maindeck him for his lifegain trigger (categories like `"Lifegain"`, `"Creature"`) in
  decks with a different actual partner or no partner at all.
  **Combined strategy for partner slots**: search with `commanderName=<partner A>` +
  `cardName=<partner B>` together (cuts the candidate pool by roughly 23x vs. `commanderName`
  alone in this test), then still apply the client-side check — keep a candidate only if
  partner B's card is tagged `categories: ["Commander"]` (see the commander-identification
  note above; presence alone isn't proof, confirmed with a concrete false-positive). The
  ~12% hit rate observed here means expect to fetch roughly 8x as many candidates as you keep
  for a partner slot — budget for that instead of treating a low yield as an error, same
  spirit as Orysa's expected-thin-corpus case.

The response HTML embeds a `<script id="__NEXT_DATA__" type="application/json">...</script>`
tag (same general technique the plan already anticipated for EDHREC via `pyedhrec` — see plan
§3/Appendix A). Extract with a regex or a proper HTML parser, then read:

```python
data = json.loads(next_data_script_text)
deck_results = data["props"]["pageProps"]["deckResults"]
deck_results["results"]  # list[dict], ~60 per page
deck_results["count"]    # capped/approximate — the UI shows "1000+" once this hits 1000,
                          # it is NOT a reliable exact total; don't treat it as ground truth
```

Each entry in `results` has deck-listing metadata (`id`, `name`, `size`, `viewCount`,
`updatedAt`, `createdAt`, `colors`, `owner`, ...) but **not** the commander or card list —
you still need §2.1's deck-by-id fetch per deck to get those.

`data["buildId"]` is present (Next.js build ID) — same mechanism `pyedhrec` uses for EDHREC.
Not needed for the approach above (fetching the rendered page works fine and is simpler than
hitting `/_next/data/<buildId>/...` directly), but worth knowing it's the same underlying
technique if the rendered-page approach ever breaks.

## 3. `mtg_parser` — what it's good for here, and what it isn't

[`mtg_parser`](https://pypi.org/project/mtg_parser/) (PyPI, actively maintained — latest
release 2026-07-25) parses decklists from 9 sites including Archidekt. Installed
(`mtg-parser` in `pyproject.toml`), verified end-to-end against our own `RateLimitedSession`
with zero special handling needed (unlike Moxfield, which needs an authorized custom
User-Agent, or Deckstats/MTGGoldfish/Aetherhub, which need a Cloudflare-bypass client) —
confirms Archidekt is the most straightforward of the sites it supports.

Its Archidekt parser (`src/mtg_parser/archidekt.py` on GitHub) does exactly what §2.1
documents: `GET /api/decks/{id}/`, filter cards by category membership. Source (for
reference — don't depend on this exact private method across versions):
```python
def _parse_deck(self, deck: dict) -> Iterable[Card]:
    categories = deck['categories']
    categories = filter(lambda c: c.get('includedInDeck', False), categories)
    categories = map(lambda c: c['name'], categories)
    categories = set(categories)
    for card in deck['cards']:
        if not card['categories'] or categories & set(card['categories']):
            yield Card(
                card['card']['oracleCard']['name'], card['quantity'],
                card['card']['edition']['editioncode'],
                card['card'].get('collectorNumber'), card['categories'],
            )
```

**Empirically confirmed limitation**: `parse_deck()` only returns `Card` objects (name,
quantity, extension, number, tags) — no deck-level metadata (views/format/dates), and on a
real deck (`3047743`, "Krenko, Mob Boss") **zero cards came back tagged `"Commander"`** —
the commander is excluded from its output entirely, consistent with `"Commander"` presumably
not being an `includedInDeck: true` category (the commander sits outside the 99-card count by
MTG rules, and Archidekt's data model reflects that).

**Decision**: don't use `mtg_parser.parse_deck()` as the core of the harvester (5.3-B) — we
need the raw deck JSON ourselves anyway for metadata and commander identification, and
calling `parse_deck()` on top of that would mean fetching/parsing the same deck twice through
two different code paths for no benefit. Instead, 5.3-B should fetch `/api/decks/{id}/`
directly and replicate the short category-filter logic above inline.

`mtg_parser` remains genuinely useful for **task 7.1** (decklist parser): its
`parse_deck(url, http_client)` signature means task 7.1 could accept a pasted *URL* (not just
raw text) from any of its 9 supported sites — Archidekt, TappedOut, MTGJSON, and Scryfall's
own decks feature all work with a plain session; Moxfield needs the authorized-UA flow.
Worth revisiting 7.1's scope when we get there.

## 4. Alternative decklist sources considered (context, not used)

Surveyed before deciding to stay with Archidekt — see conversation history for the full
comparison. Quick summary for future reference:
- **TappedOut**: also works with a plain session (per `mtg_parser`'s "known issues" list),
  but its `robots.txt` explicitly disallows `ClaudeBot` (and other named AI crawlers) while
  allowing generic user-agents — a different posture than Archidekt's explicit welcome.
  Not pursued.
- **Moxfield**: ToS explicitly prohibits scraping; has a real permission channel
  (`support@moxfield.com` for an authorized custom User-Agent) but nothing granted. Site
  also actively resisted automated/headless access during this investigation.
- **Deckstats.net / MTGGoldfish / Aetherhub**: all require `cloudscraper`
  (Cloudflare-bypass) per `mtg_parser`'s own docs — active anti-bot infrastructure, a
  meaningfully different (more adversarial) posture than Archidekt's.
- **MTGJSON**: official structured data project, but not a large corpus of user-submitted
  Commander decks — doesn't substitute for what we need.
