"""Task 6.4 — functional role classification: one primary + optional secondary role per card.

Plan `EDHCut_PLAN.md` task 6.4. Three layers, in the plan's own order:

1. **Tagger tags** (`card_tags`, source='tagger_bulk') mapped to roles through `TAG_RULES`, an
   explicit role -> {tag: weight} dict. This layer does nearly all the work: 98.7% of `cards`
   carry at least one tag, and task 5.5 already expanded every tagging up its ancestor chain,
   so a card tagged `sweeper-one-sided` also carries `sweeper` and `removal`.
2. **Oracle-text heuristics** (`TEXT_RULES`) — regex over normalized `oracle_text` + `type_line`,
   for cards the tags miss and for the roles Tagger has no good anchor for (goad has no usable
   tag at all).
3. **Default**: `other`. Deliberately the *only* default — an earlier build also had a
   `synergy_piece` bucket for "does something, just nothing on our list", dropped at the user's
   direction (2026-08-23) because a role vocabulary should say what a card does or admit it
   doesn't know, not hedge in between.

**Why weights instead of first-match-wins.** Real cards carry several role signals at once and
the right answer is usually "the strongest one, plus a real second". Path to Exile is tagged
`ramp`, `tutor` *and* `spot-removal`; Cyclonic Rift is tagged both `spot-removal` and `sweeper`.
Scoring each role and taking the argmax settles those the way a player would, and the runner-up
falls out for free as the secondary role.

**Anchor tags only, never the full expansion.** Because ancestors are expanded, a card like
Cyclonic Rift carries ten `removal-<permanent type>` tags. Summing all of them would let a
single wordy card out-score its own real role, so each role's rule set is a small number of
anchor tags (`sweeper` 4.0) plus low-weight corroborators (`sweeper-one-sided` +1.0), not every
tag in the subtree.

**Negative weights are a deliberate feature**, used where a tag means the *opposite* of what its
parent implies: `tutor-land-to-battlefield` cancels most of `tutor` (fetching a land is ramp, not
tutoring — Cultivate carries both), and `donate-rampant-growth` cancels `ramp` (the ramp goes to
an opponent — that is Path to Exile's drawback, not a mode you play it for).

**`land` is a type-line override, not a scored role.** Anything whose `type_line` contains "Land"
is `land` primary regardless of tags, because in deck terms it occupies a land slot; its
functional role becomes the *secondary* (Bojuka Bog -> land / graveyard_hate, Ancient Tomb ->
land / ramp). This is what makes role counts usable as deck quotas in task 6.6 — a land that
also ramps must not be double-counted out of the mana base.

**Vocabulary revisions, 2026-08-23** (two rounds of user review over the real output).

Renames: `card_draw`->`draw`, `board_wipe`->`boardwipe`, `evasion_enabler`->`evasion`,
`stax_tax`->`stax`, `counterspell`->`stack_interaction` (broadened past countering to copying,
redirecting, granting flash, and denying opponents the stack outright).

Added: `sacrifice_outlet`, `defensive` (keeping *you* alive, as opposed to `protection` keeping a
*permanent* alive), and `board_presence` (bodies that can block or attack, plus anything that just
adds a pile of stats — see `CREATURE_SCORE`). Anthems now score `wincon` rather than falling
through. Dropped: `synergy_piece`.

**Land destruction is deliberately *not* its own role**, after trying it. Single-target land
destruction (Stone Rain) is `spot_removal` — removal that happens to point at a land — and **mass
land destruction is `stax`** (Armageddon and Winter Orb do the same thing to a game). Both are
user decisions, and both turned out to be the easy direction: Tagger uses one tag,
`mass-land-denial`, for denial and destruction alike, so the separate-role version had to fight it
(25+ cards carry it with no `removal-land` at all). Routing MLD to stax means the tag needs no
disambiguation, which is why that rule set is now three lines instead of eight.

Output: `roles.parquet` (plan §2.4) with the primary/secondary roles, their scores, and a
per-assignment `source`, plus each layer's own independent verdict (`tagger_primary`,
`heuristic_primary`) so tagger-vs-heuristic agreement is auditable straight from the file.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from edhcut.config import CONFIG
from edhcut.db import connect
from edhcut.ingest.scryfall import normalize_name

KB_DEV_DIR = CONFIG.paths.kb_dir / "dev"

# The role vocabulary, in the tiebreak order used when two roles score exactly equal (rare, but
# the result must not depend on dict iteration order). Roughly "most specific / most
# load-bearing first". Two positions are load-bearing rather than aesthetic:
#   - `land` never actually competes (see LAND_SCORE) but outranks everything by construction.
#   - `board_presence` sits last before the default because it is the most generic role here:
#     nearly every creature has some, so anything more specific should win.
# `other` is the layer-3 default and wins only when nothing scored at all.
ROLES: tuple[str, ...] = (
    "land",
    # stax > boardwipe > graveyard_hate is a declared precedence (2026-08-23), set against
    # Decree of Annihilation: it blows up all lands, wipes the board, and exiles every graveyard,
    # and should read in that order.
    "stax",
    "boardwipe",
    "stack_interaction",
    "spot_removal",
    "sacrifice_outlet",
    "tutor",
    "mana_acceleration",
    "draw",
    "recursion",
    "graveyard_hate",
    "protection",
    "defensive",
    # `wincon` above `board_presence` so an evasion-granting or scaling lord beats a plain one
    # on a tie -- that pair of weights is deliberately set to tie (see TAG_RULES).
    "wincon",
    "evasion",
    "enhancer",
    "board_presence",
    "other",
)

ROLE_PRIORITY: dict[str, int] = {role: i for i, role in enumerate(ROLES)}

# A land's `land` score, high enough that no combination of tag/text evidence can displace it.
LAND_SCORE = 100.0

# `board_presence` for having a big body of its own (Ghalta). An earlier build gave *every*
# creature a floor score here and the role swallowed 10,028 cards; the user's rule is narrower —
# token makers, lords, counter distributors, and creatures that are or become genuinely large.
# A plain 2/2 gets nothing. Power/toughness are raw Scryfall strings ("*", "1+*"), so this is
# gated on the value parsing as an integer.
BIG_BODY_THRESHOLD = 5
BIG_BODY_SCORE = 2.5

# Minimum evidence for a role to be assignable at all. Corroborator weights are 0.5, so this
# says "an anchor, or two corroborating signals, or one text rule" — never a lone 0.5. Found by
# eyeballing the pool-wide distribution rather than by any test: without it, `power-boost-to-all`
# (0.5, and under no parent this mapping uses) made every anthem effect a `wincon`, inflating
# that role to 843 cards against the ~190 its real anchors cover.
MIN_ROLE_SCORE = 1.0

# A runner-up role needs this much evidence to be recorded as the secondary. Lowered from 1.5 to
# 1.0 (= `MIN_ROLE_SCORE`) at the user's direction 2026-08-23: **be liberal with secondaries.** A
# secondary is free information — Path to Exile really can ramp you if you point it at your own
# creature — and the cost of a slightly loose one is much lower than the cost of dropping a real
# mode. At 1.0 anything good enough to be *a* role at all is good enough to be a second one.
SECONDARY_MIN_SCORE = 1.0

# Roles that can never be a *secondary*: `land` is always primary when it applies, and the
# layer-3 default means "nothing else matched", which is never true when a primary exists.
NON_SECONDARY_ROLES = frozenset({"land", "other"})


# --------------------------------------------------------------------------------------
# Layer 1 — tagger_bulk tags
# --------------------------------------------------------------------------------------

# role -> {tag: weight}. Weights are on a deliberately coarse scale: ~3-4 for an anchor tag that
# alone justifies the role, ~0.5-1.0 for a corroborator, negative for an explicit correction.
# Every tag here is asserted to exist in the live vocabulary by `unknown_tags()` (and its test)
# — a typo'd slug would otherwise silently contribute nothing.
TAG_RULES: dict[str, dict[str, float]] = {
    "mana_acceleration": {
        # Named `mana_acceleration`, not `ramp`, because it covers rituals and cost reducers as
        # well as permanent mana sources -- "ramp" reads wrong for Dark Ritual.
        "ramp": 3.0,                      # parent of mana-rock / mana-dork / land-ramp / ritual
        "mana-rock": 0.5,
        "mana-dork": 0.5,
        "mana-producer": 0.5,
        "land-ramp": 0.5,
        "adds-multiple-mana": 0.5,
        "repeatable-treasures": 1.0,
        # Paying less for a spell accelerates you exactly like producing more mana does.
        # Ruby Medallion, Urza's Incubator, Goblin Warchief. `discount-self` is deliberately NOT
        # here -- "this card costs less" (Ghalta) accelerates nothing.
        "cost-reducer": 3.0,
        # NOTE: colour fixing is deliberately absent. `mana-fix` sits on Chromatic Lantern (which
        # also taps for mana, so `ramp` covers it) and on Prismatic Omen (which adds no mana at
        # all and correctly falls outside this role).
        # The searched land goes to an *opponent*, so this is not your acceleration -- but only
        # a partial cancel, because Path to Exile on your own creature really does ramp you.
        "donate-rampant-growth": -1.5,
    },
    "draw": {
        "draw": 3.0,                      # parent of pure-draw / cantrip / burst-draw / draw-engine
        "draw-engine": 1.0,
        "repeatable-draw": 0.5,
        "pure-draw": 0.5,
        "burst-draw": 0.5,
        # Impulse draw is a real anchor, not a corroborator: Tagger never tags Light Up the Stage
        # or Outpost Siege `draw`, because they never draw.
        "impulsive-draw": 3.0,
        "repeatable-impulsive-draw": 1.0,
        "long-term-impulsive-draw": 0.5,
        "impulse": 1.5,
        # Loot and rummage ARE draw (decision 2026-08-23): one card reaching several others, in
        # the right shell, is card advantage even though hand size does not go up. Faithless
        # Looting, Merfolk Looter, Frantic Search, Thrill of Possibility.
        "loot": 3.0,
        "repeatable-loot": 1.0,
        "rummage": 3.0,
        "repeatable-rummage": 1.0,
        "card-advantage": 0.5,
        "repeatable-card-advantage": 0.5,
        # Cantrips replace themselves and are card-NEUTRAL, so they are not card advantage
        # (decision 2026-08-23). `hand-neutral` is the general form of the same objection and is
        # what also removes Mind Stone and Commander's Sphere -- mana rocks that happen to cash
        # themselves in for a card. Both are large enough to fully cancel `draw` + its
        # corroborators; loot/rummage survive because their own anchors are worth more.
        "cantrip": -4.0,
        "hand-neutral": -4.0,
    },
    "spot_removal": {
        "spot-removal": 3.0,
        "removal-destroy": 0.5,
        "removal-exile": 0.5,
        "removal-bounce": 0.5,
        "removal-burn": 0.5,
        "removal-land": 0.5,
    },
    "boardwipe": {
        # 5.0, not 4.0, so a wipe that *also* hits graveyards (Farewell) reads as a board wipe
        # first -- the mode it is played for -- rather than as graveyard hate.
        "sweeper": 5.0,
        "sweeper-one-sided": 1.0,
        "multi-removal": 1.0,
        # `removal-creature` is deliberately NOT a weight here -- it sits on 5,219 cards, and once
        # SECONDARY_MIN_SCORE dropped to 1.0 it made every one of them "also a boardwipe" (2,597
        # spurious secondaries). It only ever existed to separate Jokulhaups from Armageddon, so
        # it lives in TAG_COMBOS paired with `sweeper` instead.
        # Mass land destruction is `stax`; this penalty keeps Armageddon (tagged `sweeper`, wipes
        # no board) out of boardwipe entirely, primary and secondary alike.
        "mass-land-denial": -4.5,
    },
    "stack_interaction": {
        # Broader than "counterspell": anything that fights over, copies, redirects, grants or
        # denies access to the stack.
        "counterspell": 4.0,
        "counterspell-exile": 0.5,
        "counterspell-sweeper": 0.5,
        "copy-spell": 3.0,
        "copy-instant": 0.5,
        "copy-sorcery": 0.5,
        "change-target": 3.0,             # Deflecting Swat, Bolt Bend
        "gives-flash": 3.0,               # Vedalken Orrery -- lets *you* use the stack
        "prevent-cast": 2.5,              # Grand Abolisher -- denies *them* the stack
        "prevent-activation": 2.0,
        # Making your own spells uncounterable is stack interaction too (Cavern of Souls).
        "gives-uncounterable": 3.0,
        "hate-counterspell": 2.0,
        "hate-instant": 1.0,
        "hate-flash": 1.0,
    },
    "tutor": {
        "tutor": 3.0,
        "tutor-card": 0.5,
        "tutor-to-hand": 0.5,
        "tutor-to-battlefield": 0.5,
        "tutor-creature": 0.5,
        # -3.0 cancels `tutor` (3.0) + `tutor-to-battlefield` (0.5): putting a land onto the
        # battlefield is mana base, full stop. Bare `tutor-land` is deliberately NOT penalised --
        # Expedition Map fetching a utility land is a genuine tutor.
        "tutor-land-basic": -1.5,
        "tutor-land-to-battlefield": -3.0,
    },
    "sacrifice_outlet": {
        "sacrifice-outlet": 4.0,
        "repeatable-sacrifice-outlet": 1.0,
        "free-sacrifice-outlet": 1.0,
        "sacrifice-outlet-creature": 0.5,
    },
    "recursion": {
        "recursion": 3.0,                 # parent of reanimate / regrowth
        "reanimate": 1.0,
        "regrowth": 0.5,
        "mass-reanimation": 1.0,
        # A card that only returns *itself* (Bloodghast) is a payoff, not a recursion spell.
        "recursion-self": -2.5,
        "reanimate-self": -1.0,
    },
    "graveyard_hate": {
        "hate-graveyard": 4.0,
        # Only a corroborator: `sweeper-graveyard` also sits on modal wipes whose main mode is
        # the board (Farewell), so it must not out-weigh `sweeper` on its own.
        "sweeper-graveyard": 1.0,
        # An MLD card's graveyard clause is a rider, not its point -- Decree of Annihilation
        # exiles all graveyards only when cycled. -4.0 rather than -3.0 so it also lands *below*
        # that card's boardwipe reading, giving the declared stax > boardwipe > graveyard_hate
        # order for exactly the card the order was declared for.
        "mass-land-denial": -4.0,
    },
    "protection": {
        # `protection` is NOT the anchor: Tagger puts it on anything that keeps *anyone* alive,
        # including Solitary Confinement (which protects the player, not a permanent) and every
        # fog. Protection here means specifically **your permanents**, so the `protects-*` tags
        # are the anchors and the two player-directed grants are penalties.
        "protection": 1.5,
        "protects-all": 2.0,
        "protects-creature": 2.0,
        "protects-permanent": 1.5,
        "gives-hexproof": 1.0,
        "gives-indestructible": 1.0,
        "gives-shroud": 1.0,
        "gives-ward": 1.0,
        "gives-protection": 1.0,
        "gives-player-shroud": -2.0,
        "gives-player-hexproof": -2.0,
    },
    "defensive": {
        # Keeping *yourself* alive, as opposed to `protection`, which keeps your permanents alive.
        "fog": 4.0,
        "pseudo-fog": 2.5,
        "fog-selective": 2.5,
        "damage-prevention": 1.5,
        "damage-prevention-you": 1.0,
        "damage-prevention-player": 1.0,
        # 4.0, above the `stax` reading these same cards get from "creatures can't attack you
        # unless..." — pillowfort is defensive first (decision 2026-08-23), with stax as the
        # secondary. Ghostly Prison and Propaganda are the cards this is set against.
        "pillowfort": 4.0,
        "prevent-attack": 2.0,
        "hate-attacker": 1.0,
        "repeatable-lifegain": 1.5,
        "lifegain": 0.5,
        "lifegain-increaser": 0.5,
        "set-life-total": 1.0,
        "gives-defender": 0.5,
    },
    "evasion": {
        # `gives-evasion` is the parent of gives-flying / gives-unblockable / gives-menace, but
        # NOT of gives-trample (checked live: 5.2% overlap), so trample is listed separately.
        # The bare `evasion` tag (4,976 cards) is deliberately excluded -- it means the card
        # *has* evasion (any 2/2 flier), not that it enables evasion for the deck.
        "gives-evasion": 3.0,
        "gives-unblockable": 1.0,
        "gives-trample": 1.0,
        "gives-flying": 0.5,
        "gives-menace": 0.5,
        "gives-landwalk": 0.5,
        "prevent-blocker": 1.0,
        "hate-blocker": 0.5,
    },
    "stax": {
        # `tax` is NOT the anchor it looks like. Tagger uses it (with `toll`) for "something
        # happens when an opponent does something", which covers real taxes but also Soul Warden
        # and Roaming Throne. The real anchors name an actual cost being imposed.
        "tax": 0.5,
        "cast-tax": 3.0,
        "cost-increaser": 3.0,
        "stasis": 3.0,
        "lockdown": 3.0,
        "mass-land-denial": 3.0,          # land denial AND mass land destruction both land here
        "hate-nonbasic-land": 2.0,
        "tax-attack": 1.5,
        "tax-block": 1.0,
        # **Land destruction is stax, not removal** (decision 2026-08-23), *unless* the card
        # replaces what it destroys or could have pointed at something else. Blowing up a land
        # with nothing in return attacks the opponent's ability to play the game, which is what
        # stax is; Ghost Quarter handing back a basic is a trade, which is removal.
        #   - `swap-removal` = it gives something back (Ghost Quarter, Demolition Field, Beast
        #     Within, Chaos Warp) -> not stax.
        #   - the other `removal-<type>` tags = it could have hit a nonland permanent instead
        #     (Acidic Slime, Decimate, Terastodon) -> ordinary multi-mode removal.
        # Stone Rain, Strip Mine, Wasteland and Molten Rain carry `removal-land` and none of the
        # escape hatches, so they land here.
        "removal-land": 4.5,
        "swap-removal": -3.0,
        "removal-creature": -1.5,
        "removal-artifact": -1.5,
        "removal-enchantment": -1.5,
        "removal-planeswalker": -1.5,
        "removal-nonland": -1.5,
        "removal-permanent": -3.0,
    },
    "wincon": {
        "alternate-win-condition": 5.0,
        "overrun": 4.0,
        "poison-opponents": 3.0,
        # Extra turns and extra combats are win conditions, not value (decision 2026-08-23).
        "extra-turn": 3.0,
        "extra-combat-phase": 3.5,
        # **Damage aimed at players closes games.** Impact Tremors, Goblin War Strike, Purphoros.
        # `burn-creature` is the discriminator that keeps Lightning Bolt in spot_removal: a burn
        # spell that can point at a creature is removal that happens to also hit players.
        "burn-player": 4.0,
        "burn-player-each": 1.0,
        "group-slug": 1.0,
        "opponent-loses-life": 1.5,
        "drain-life": 1.0,
        "burn-creature": -3.0,
        # **A lord is `board_presence` by default, and only a `wincon` when the buff is massive
        # or comes with evasion** (decision 2026-08-23). `anthem` alone scores less here than in
        # board_presence; paired with `gives-evasion` (Eldrazi Monument, Lord of the Accursed,
        # Jetmir) or with `quadratic` scaling (Coat of Arms, Shared Animosity) it wins.
        # `anthem` on its own is worth less here than in board_presence, so a plain lord is board
        # presence. What promotes it is a *conjunction* — see TAG_COMBOS. Bare `gives-evasion` is
        # deliberately absent: on its own it means the `evasion` role, not a win condition.
        "anthem": 2.0,
        "quadratic": 2.5,
    },
    "enhancer": {
        # Cards whose job is to amplify something else rather than do it: doublers, increasers,
        # extra phases, counter manipulation. This is the group that was left over in `other`
        # after every other role was assigned -- Doubling Season, Panharmonicon, Hardened Scales,
        # The Ozolith, Thousand-Year Elixir -- and it turned out to be coherent enough to name.
        "trigger-doubler": 4.0,           # Panharmonicon
        "counter-doubler": 3.5,           # Doubling Season, Branching Evolution
        "counter-increaser": 3.5,         # Hardened Scales
        "token-doubler": 3.5,
        "token-increaser": 3.5,
        "power-doubler": 3.0,
        "damage-multiplier": 3.0,
        "life-doubler": 2.0,
        "move-counters": 3.0,             # The Ozolith
        "counter-preservation": 2.0,
        "pseudo-proliferate": 2.0,
        "extra-upkeep": 3.5,              # Paradox Haze -- extra phases that are not combat
        "extra-draw-step": 3.5,
        "play-additional-land": 3.0,
        "extra-land": 1.0,
        "extra-untap": 1.0,
        # Thousand-Year Elixir: letting an activated ability be reused is amplification.
        "synergy-activated-ability": 3.0,
    },
    "board_presence": {
        # Bodies that can block or attack, and cards that build a board. Deliberately NOT every
        # creature -- an earlier build gave every creature a floor score and swallowed 10,028
        # cards. A plain 2/2 is not what this role is for.
        "repeatable-creature-tokens": 3.0,
        "repeatable-token-generator": 0.5,
        # Artifact/Treasure tokens are deliberately excluded -- a Treasure cannot block.
        # Lords: `anthem` scores *higher* here than in `wincon`, so a plain lord (Imperious
        # Perfect, Patchwork Banner) is board presence and only an evasion-granting or scaling
        # one is a win condition.
        "anthem": 3.5,
        "keyword-anthem": 0.5,
        "power-boost-to-all": 0.5,
        "toughness-boost-to-all": 0.5,
        "gives-pp-counters": 2.5,
        "gives-pp-counters-to-all": 1.0,
        "repeatable-pp-counters": 1.0,
        "gains-pp-counters": 2.0,         # Taurean Mauler: a body that grows
        "leaves-body-behind": 1.0,
    },
}


# Conjunctions: a bonus that applies only when a card carries *every* tag in the set. Additive
# weights cannot express "A and B together mean something neither means alone", and this role set
# needs exactly that once: **a lord is `board_presence`, but a lord that also grants evasion or
# scales is a `wincon`** (Eldrazi Monument, Lord of the Accursed, Goblin King, Coat of Arms). Doing
# it with plain weights would mean giving `gives-evasion` a large `wincon` weight, which would then
# fire on every evasion granter in the pool.
TAG_COMBOS: tuple[tuple[str, frozenset[str], float], ...] = (
    ("wincon", frozenset({"anthem", "gives-evasion"}), 5.0),
    ("wincon", frozenset({"anthem", "quadratic"}), 4.0),
    # Jokulhaups wipes the board as well as the lands; Armageddon does not. `removal-creature`
    # separates them, but only in the presence of `sweeper` -- on its own it means nothing here.
    ("boardwipe", frozenset({"sweeper", "removal-creature"}), 1.0),
)


def _tag_components(tags: set[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Split one card's tag evidence into `(credits, penalties)` — the positive weights summed per
    role, and the negative ones summed separately.

    They have to be separable because they mean different things. A credit is *this layer's*
    reading of the card, to be compared against the text layer's. A penalty is a statement about
    the **card**, true no matter which layer noticed the role, so `classify` applies penalties
    after the layers are combined. Without that split, Ponder's `cantrip`/`hand-neutral` penalties
    cancelled the tag layer's `draw` while the text layer's plain "Draw a card." sailed through
    untouched — the card came out as draw anyway."""
    credits: dict[str, float] = {}
    penalties: dict[str, float] = {}
    for role, rules in TAG_RULES.items():
        credit = sum(w for tag, w in rules.items() if tag in tags and w > 0)
        penalty = sum(w for tag, w in rules.items() if tag in tags and w < 0)
        if credit:
            credits[role] = credit
        if penalty:
            penalties[role] = penalty
    for role, combo, bonus in TAG_COMBOS:
        if combo <= tags:
            credits[role] = credits.get(role, 0.0) + bonus
    return credits, penalties


def score_tags(tags: Iterable[str]) -> dict[str, float]:
    """Layer 1: `TAG_RULES` (plus `TAG_COMBOS`) applied to one card's tag set -> {role: score},
    net of penalties, with non-positive roles dropped — a score of zero or less means "explicitly
    not this role". This is the tag layer's own standalone verdict, which is what
    `RoleAssignment.tagger_primary` and the agreement metric use; `classify` recombines the
    components itself (see `_tag_components`)."""
    tags = set(tags)
    credits, penalties = _tag_components(tags)
    scores = {
        role: credits.get(role, 0.0) + penalties.get(role, 0.0)
        for role in set(credits) | set(penalties)
    }
    return {role: score for role, score in scores.items() if score > 0}


# --------------------------------------------------------------------------------------
# Layer 2 — oracle-text heuristics
# --------------------------------------------------------------------------------------

_REMINDER_TEXT = re.compile(r"\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")


def normalize_oracle_text(name: str, oracle_text: str | None) -> str:
    """Lowercased oracle text with reminder text stripped and every occurrence of the card's own
    name replaced by `~`, so a rule can't fire on a card that merely *names* an effect (a card
    called "Counterspell" shouldn't heuristically read as one, and "Krenko, Mob Boss" shouldn't
    match a rule looking for the word "boss"). Both the full name and each face of a
    `A // B` multi-faced name are replaced — Scryfall writes each face's own oracle text using
    that face's short name, not the combined one."""
    text = (oracle_text or "").lower()
    text = _REMINDER_TEXT.sub(" ", text)
    faces = [name] + [face.strip() for face in name.split(" // ")] if name else []
    for face in sorted({f for f in faces if f}, key=len, reverse=True):
        text = text.replace(face.lower(), "~")
    return _WHITESPACE.sub(" ", text).strip()


# The land-fetch target — shared by the `ramp` rule (which wants it) and the `tutor` rule (which
# must exclude it), so the two can never disagree about what counts as fetching a land. Basic
# land *type names* have to be spelled out: Farseek and Nature's Lore never use the word "land"
# ("search your library for a Plains, Island, Swamp, or Mountain card"), and without this they
# read as generic tutors on text alone.
_LAND_TARGET = (
    r"(?:(?:basic |snow )*\w* ?lands? card"
    r"|(?:plains|island|swamp|mountain|forest)[\w\s,]*card)"
)

# role -> ((pattern, weight), ...) over `normalize_oracle_text` output. Weight scale matches
# `TAG_RULES` so the two layers are directly comparable. Kept deliberately conservative: layer 1
# already covers 98.7% of cards, so these exist to catch the untagged tail and to give the roles
# Tagger has no broad anchor for (goad has no usable tag at all; `mass_land_destruction` has
# exactly one) real coverage.
TEXT_RULES: dict[str, tuple[tuple[re.Pattern[str], float], ...]] = {
    "mana_acceleration": (
        (re.compile(rf"search your library for (?:a |an |up to \w+ )?{_LAND_TARGET}"), 3.0),
        (re.compile(r"add \{[wubrgc]\}"), 2.0),
        (re.compile(r"add (one|two|three) mana"), 2.0),
        (re.compile(r"create (a|two|three) .{0,20}treasure token"), 2.0),
    ),
    "draw": (
        # Second-person and third-person are separate patterns on purpose: an earlier `draws?`
        # spelling made both fire on the same clause and double-scored every draw spell.
        (re.compile(r"draw (a card|\w+ cards)"), 3.0),
        (re.compile(r"draws (a card|\w+ cards)"), 1.0),
        # Impulse draw: exile off the top and get permission to play it.
        (re.compile(r"exile the top .{0,30}of your library.{0,80}you may (play|cast)"), 3.0),
    ),
    "spot_removal": (
        # The two lookaheads carry the merged land-destruction cases and are the same principle
        # as graveyard_hate's `(?!your\b)`: hitting your *own* resource is a cost, not removal.
        # `you control` excludes Extraplanar Lens's imprint ("exile target land you control"),
        # `from a graveyard` excludes Deathrite Shaman ("exile target land card from a
        # graveyard"), which is ramp plus graveyard hate.
        (re.compile(
            r"(destroy|exile) target (?!player\b|opponent\b)"
            r"(?![\w\s'-]{0,30}you control)"
            r"(?![\w\s'-]{0,30}from (?:a|your|target player's) graveyard)"
        ), 3.0),
        (re.compile(r"deals? \d+ damage to target creature"), 3.0),
        (re.compile(r"return target .{0,30}to (its owner's|their owner's) hand"), 2.0),
    ),
    "boardwipe": (
        (re.compile(r"(destroy|exile) (all|each) (other )?(creature|permanent|nonland)"), 4.0),
        (re.compile(r"all creatures get -\d+/-\d+"), 4.0),
        (re.compile(r"each player sacrifices? (all|\w+) creature"), 3.0),
        (re.compile(r"return all .{0,40}to (?:their|its) owners?'? hands?"), 4.0),
    ),
    "stack_interaction": (
        (re.compile(r"counter target (?!.{0,20}ability\b)"), 4.0),
        (re.compile(r"copy target (instant|sorcery|spell)"), 4.0),
        (re.compile(r"change the target"), 3.0),
        (re.compile(r"as though (it|they) had flash"), 3.0),
        (re.compile(r"(players|opponents|your opponents) can't cast spells"), 3.0),
    ),
    "tutor": (
        (re.compile(
            rf"search your library for (?:a|an|up to \w+) (?!{_LAND_TARGET})[\w\s,-]{{0,40}}card"
        ), 3.0),
    ),
    "sacrifice_outlet": (
        (re.compile(r"sacrifice (a|another) (creature|permanent|artifact|token)[^:]{0,20}:"), 4.0),
    ),
    "recursion": (
        (re.compile(r"return .{0,40}from your graveyard to (the battlefield|your hand)"), 3.0),
        (re.compile(r"return target .{0,40}card from (a|your) graveyard"), 2.0),
    ),
    # The `(?!your\b)` is load-bearing: exiling cards from *your own* graveyard is a cost or a
    # fuel engine, not hate. Without it Necropotence ("whenever you discard a card, exile that
    # card from your graveyard") classified as graveyard_hate on text alone.
    "graveyard_hate": (
        (re.compile(r"exile [\w\s,'-]{0,30}from (?!your\b)[\w\s',]{0,25}graveyards?"), 3.0),
        (re.compile(r"exile (target player's|each player's|that player's|all) graveyards?"), 3.0),
        (re.compile(r"would be put into .{0,30}graveyard.{0,25}exile it instead"), 3.0),
    ),
    # `gains?` already covers both "gains hexproof" and "creatures you control gain hexproof";
    # an earlier second pattern spelled `gain (...)` made both fire on the same clause, which
    # doubled the score and pushed Craterhoof Behemoth from `wincon` to `evasion`.
    "protection": (
        (re.compile(r"gains? (hexproof|indestructible|shroud|protection from)"), 3.0),
    ),
    "defensive": (
        (re.compile(r"prevent all (combat )?damage"), 4.0),
        (re.compile(r"you gain \d+ life"), 1.5),
        (re.compile(r"goads?\b"), 2.0),          # no usable Tagger tag exists for goad
        (re.compile(r"creatures? can't attack you"), 3.0),
    ),
    "evasion": (
        (re.compile(r"gains? (flying|menace|trample|fear|intimidate)"), 2.0),
        (re.compile(r"can't be blocked"), 3.0),
        # `(?<!this )` is load-bearing: "this creature can't block" is a *drawback on itself*, not
        # evasion granted to your team. Carrion Feeder read as an evasion enabler without it.
        (re.compile(r"(?<!this )creatures? can't block"), 3.0),
    ),
    "wincon": (
        (re.compile(r"you win the game"), 5.0),
        (re.compile(r"(target player|each opponent|that player) loses the game"), 5.0),
        (re.compile(r"take an extra turn"), 4.0),
        (re.compile(r"(untap all creatures you control|there is an additional combat phase)"), 3.5),
        (re.compile(r"each opponent loses \d+ or more life"), 3.0),
        # A *massive* team buff is a win condition; an ordinary lord's +1/+1 is board presence.
        # +2/+2 is the cutoff, which puts Beastmaster Ascension and Overrun here and leaves
        # Imperious Perfect and Patchwork Banner in board_presence. 4.5 is set to tie
        # board_presence's own anthem score, with `wincon` winning the tie via ROLES order.
        (re.compile(r"creatures you control get \+([2-9]|\d\d)/"), 4.5),
    ),
    "stax": (
        (re.compile(r"cost \{\d\}( or more)? more to cast"), 3.0),
        (re.compile(r"(players|opponents|creatures) can't (attack|block|untap|search)"), 3.0),
        (re.compile(r"don't untap during (their|your) untap step"), 3.0),
        (re.compile(r"unless (that player|they) pays? \{"), 2.0),
        # Mass land destruction, classed as stax by decision. `\blands?\b` (not `lands?`) so
        # "destroy all nonland permanents" cannot match -- there is no word boundary inside
        # "nonland". The `[\w\s,]` run spans the list in "destroy all artifacts, creatures, and
        # lands" (Jokulhaups) but cannot cross a sentence boundary into an unrelated clause.
        # The highest weight in the whole mapping, deliberately. Blowing up everyone's lands is
        # the defining thing one of these cards does, and they reliably carry a lot of incidental
        # other text that scores elsewhere -- Decree of Annihilation's cycling clause exiles
        # graveyards, giving it `hate-graveyard` + `sweeper-graveyard` = 5.0 of graveyard hate.
        (re.compile(r"(destroy|exile) (all|each) [\w\s,]{0,40}\blands?\b"), 6.0),
        (re.compile(r"each player sacrifices? (all|\w+) \blands?\b"), 4.0),
    ),
    "board_presence": (
        # Catches one-shot token makers (Call the Coppercoats, Krenko's Command) that carry no
        # `repeatable-creature-tokens` tag. No `gets +N/+N` rule: that fired on every generic
        # equipment, and Bonesplitter is not board presence.
        (re.compile(r"create \w+ .{0,40}creature tokens?"), 3.0),
        (re.compile(r"put \w+ \+1/\+1 counters?"), 2.0),
    ),
    "enhancer": (
        (re.compile(r"(twice that many|double the number|that many plus)"), 3.0),
        (re.compile(r"one or more .{0,30}would (enter|be put|be created)"), 2.0),
        (re.compile(r"additional \+1/\+1 counter"), 3.0),
        (re.compile(r"(additional|extra) (upkeep|end step|draw step|combat phase)"), 3.0),
        (re.compile(r"triggers? an additional time"), 3.5),
    ),
}


def score_text(name: str, type_line: str | None, oracle_text: str | None) -> dict[str, float]:
    """Layer 2: `TEXT_RULES` applied to one card -> {role: score}. `type_line` gates the `ramp`
    rules: `search your library for a land` on a Land card is that land's own fetch ability, not
    a ramp spell in a nonland slot.

    Type-line signals (`LAND_SCORE`, `CREATURE_SCORE`) deliberately do NOT live here — they are
    applied in `classify` instead, so `heuristic_primary` stays a verdict about the card's actual
    *text* and the tagger-vs-heuristic agreement metric measures what it claims to. Folding
    `CREATURE_SCORE` in here dropped that agreement from 80% to 63% in one step, purely by making
    the "heuristic" layer fire on all 17,751 creatures in the pool."""
    text = normalize_oracle_text(name, oracle_text)
    if not text:
        return {}
    type_line = type_line or ""
    is_land = "Land" in type_line

    scores: dict[str, float] = {}
    for role, rules in TEXT_RULES.items():
        if is_land and role == "mana_acceleration":
            # A fetchland's own ability shouldn't read as a ramp *spell*; the land override
            # makes it `land` primary anyway, and this keeps ramp from becoming its secondary
            # on every fetch/karoo in the pool.
            continue
        score = sum(weight for pattern, weight in rules if pattern.search(text))
        if score > 0:
            scores[role] = score
    return scores


# --------------------------------------------------------------------------------------
# Combining the layers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleAssignment:
    """One card's classification. `tagger_primary`/`heuristic_primary` are each layer's own
    independent argmax (before the land override and before the layers are summed) — kept so
    the tagger-vs-heuristic agreement the plan asks for is measurable per card, not just in
    aggregate."""

    primary: str
    primary_score: float
    primary_source: str
    secondary: str | None
    secondary_score: float
    secondary_source: str | None
    tagger_primary: str | None
    heuristic_primary: str | None


def _argmax_role(scores: Mapping[str, float]) -> str | None:
    """Highest-scoring role that clears `MIN_ROLE_SCORE`, ties broken by `ROLE_PRIORITY` so the
    result is deterministic."""
    eligible = {role: score for role, score in scores.items() if score >= MIN_ROLE_SCORE}
    if not eligible:
        return None
    return min(eligible, key=lambda role: (-eligible[role], ROLE_PRIORITY[role]))


def _source_for(role: str, tag_scores: Mapping[str, float], text_scores: Mapping[str, float]) -> str:
    """Which layer earned this role: 'tagger_bulk', 'heuristic', 'both' when each contributed at
    least a corroborator's worth (1.0) on its own, or 'type_line' when neither did — which means
    the score came from a type-line signal (`LAND_SCORE`, or `CREATURE_SCORE` on a card whose
    only claim to `board_presence` is being a creature)."""
    tag_score = tag_scores.get(role, 0.0)
    text_score = text_scores.get(role, 0.0)
    if tag_score < 1.0 and text_score < 1.0:
        return "type_line"
    if tag_score >= 1.0 and text_score >= 1.0:
        return "both"
    return "tagger_bulk" if tag_score >= text_score else "heuristic"


def _is_big_body(power: str | None, toughness: str | None) -> bool:
    """Is this creature large enough that its own body is the point (Ghalta)? Power/toughness are
    raw Scryfall strings and are frequently non-numeric (`*`, `1+*`, `?`), so a value that doesn't
    parse as an integer simply doesn't count — a `*`-power creature's size is a property of the
    board, not of the card."""
    for value in (power, toughness):
        try:
            if int(str(value)) >= BIG_BODY_THRESHOLD:
                return True
        except (TypeError, ValueError):
            continue
    return False


def classify(
    *,
    name: str,
    type_line: str | None,
    oracle_text: str | None,
    tags: Iterable[str],
    power: str | None = None,
    toughness: str | None = None,
) -> RoleAssignment:
    """The full three-layer classification for one card. Pure — no DB access — so the mapping
    can be unit-tested and eyeballed on hand-written cards. `power`/`toughness` are optional and
    only feed `board_presence`'s big-body signal; omitting them costs nothing else."""
    tags = list(tags)
    tag_scores = score_tags(tags)
    text_scores = score_text(name, type_line, oracle_text)
    credits, penalties = _tag_components(set(tags))

    # Max of the two layers' *credits*, then tag penalties applied on top.
    #
    # Max, not sum, because the two layers are independent *readings of the same card*: a draw
    # spell tagged `draw` whose text also says "draw a card" is one piece of evidence seen twice.
    # Summing let the text layer tip any card whose tags already covered it — Mind Stone (a mana
    # rock you can sacrifice to draw) scored mana 4.0+2.0 against draw 4.0+3.0 and came out a
    # draw spell.
    #
    # Penalties are applied *after* the max, not folded into the tag layer, because they are
    # claims about the card rather than about a layer. Ponder is the case that proves it: its
    # `cantrip`/`hand-neutral` penalties cancel the tag layer's `draw`, but the text layer reads
    # a plain "Draw a card." and would otherwise carry the role through untouched.
    combined: dict[str, float] = {
        role: max(credits.get(role, 0.0), text_scores.get(role, 0.0)) + penalties.get(role, 0.0)
        for role in set(credits) | set(text_scores) | set(penalties)
    }

    tagger_primary = _argmax_role(tag_scores)
    heuristic_primary = _argmax_role(text_scores)

    # Type-line signals, applied after the two layers are combined so they never pollute either
    # layer's own verdict (see `score_text`'s docstring). `land` is an override; a big body is
    # ordinary evidence that anything more specific outscores.
    if "Land" in (type_line or ""):
        combined["land"] = LAND_SCORE
    elif "Creature" in (type_line or "") and _is_big_body(power, toughness):
        combined["board_presence"] = max(combined.get("board_presence", 0.0), BIG_BODY_SCORE)

    primary = _argmax_role(combined)
    if primary is None:
        # Layer 3, and the only default: `other` means "nothing on our role list fits". An
        # earlier build split this into `synergy_piece` (has some functional tag) vs `other`
        # (has none); dropped 2026-08-23 -- a role vocabulary should name what a card does or
        # admit it doesn't know, and `synergy_piece` was doing neither for 12,841 cards.
        return RoleAssignment(
            primary="other",
            primary_score=0.0,
            primary_source="default",
            secondary=None,
            secondary_score=0.0,
            secondary_source=None,
            tagger_primary=None,
            heuristic_primary=None,
        )

    primary_source = _source_for(primary, tag_scores, text_scores)

    runners_up = {
        role: score
        for role, score in combined.items()
        if role != primary and role not in NON_SECONDARY_ROLES and score >= SECONDARY_MIN_SCORE
    }
    secondary = _argmax_role(runners_up)

    return RoleAssignment(
        primary=primary,
        primary_score=float(combined[primary]),
        primary_source=primary_source,
        secondary=secondary,
        secondary_score=float(runners_up[secondary]) if secondary else 0.0,
        secondary_source=_source_for(secondary, tag_scores, text_scores) if secondary else None,
        tagger_primary=tagger_primary,
        heuristic_primary=heuristic_primary,
    )


# --------------------------------------------------------------------------------------
# Building the table
# --------------------------------------------------------------------------------------

ROLES_COLUMNS: tuple[str, ...] = (
    "oracle_id",
    "name",
    "type_line",
    "primary_role",
    "primary_score",
    "primary_source",
    "secondary_role",
    "secondary_score",
    "secondary_source",
    "tagger_primary",
    "heuristic_primary",
    "n_tags",
)


def load_card_tags(conn: sqlite3.Connection, *, source: str = "tagger_bulk") -> dict[str, list[str]]:
    """`oracle_id -> [tag, ...]` in one scan. Scoped to one `card_tags.source` so a future
    `source='textmine'` write (task 6.5) can't feed its own derived tags back into layer 1 and
    make the two tasks circular."""
    tags: dict[str, list[str]] = {}
    for oracle_id, tag in conn.execute(
        "SELECT oracle_id, tag FROM card_tags WHERE source = ?", (source,)
    ):
        tags.setdefault(oracle_id, []).append(tag)
    return tags


def assign_roles(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per `cards` row — the whole commander-legal pool, not just cards seen in a
    harvested deck. The recommender has to answer for whatever a user pastes in, and a role is
    well-defined for a card in zero decks (same reasoning as `playrate.build_card_play_rates`)."""
    tags_by_card = load_card_tags(conn)
    rows = conn.execute(
        "SELECT oracle_id, name, type_line, oracle_text, power, toughness FROM cards"
    ).fetchall()

    records = []
    for oracle_id, name, type_line, oracle_text, power, toughness in rows:
        tags = tags_by_card.get(oracle_id, [])
        assignment = classify(
            name=name, type_line=type_line, oracle_text=oracle_text, tags=tags,
            power=power, toughness=toughness,
        )
        records.append(
            {
                "oracle_id": oracle_id,
                "name": name,
                "type_line": type_line,
                "primary_role": assignment.primary,
                "primary_score": assignment.primary_score,
                "primary_source": assignment.primary_source,
                "secondary_role": assignment.secondary,
                "secondary_score": assignment.secondary_score,
                "secondary_source": assignment.secondary_source,
                "tagger_primary": assignment.tagger_primary,
                "heuristic_primary": assignment.heuristic_primary,
                "n_tags": len(tags),
            }
        )
    return pd.DataFrame(records, columns=list(ROLES_COLUMNS))


def role_distribution(roles: pd.DataFrame) -> pd.DataFrame:
    """Per-role card counts: as primary, as secondary, and by which layer assigned the primary."""
    primary = roles["primary_role"].value_counts()
    secondary = roles["secondary_role"].value_counts()
    by_source = roles.pivot_table(
        index="primary_role", columns="primary_source", values="oracle_id", aggfunc="count"
    ).fillna(0).astype(int)

    table = pd.DataFrame({"role": list(ROLES)})
    table["as_primary"] = table["role"].map(primary).fillna(0).astype(int)
    table["as_secondary"] = table["role"].map(secondary).fillna(0).astype(int)
    for source in ("tagger_bulk", "heuristic", "both", "type_line", "default"):
        column = by_source[source] if source in by_source.columns else None
        table[f"primary_from_{source}"] = (
            table["role"].map(column).fillna(0).astype(int) if column is not None else 0
        )
    return table


def layer_agreement(roles: pd.DataFrame) -> dict[str, float | int]:
    """Tagger-vs-heuristic agreement, over the cards where *both* layers produced a verdict —
    the only population where "do they agree?" is a meaningful question. Reported alongside how
    often each layer fires alone, since that (not the agreement rate) is what says how much the
    heuristic layer is actually contributing."""
    both = roles.dropna(subset=["tagger_primary", "heuristic_primary"])
    agree = int((both["tagger_primary"] == both["heuristic_primary"]).sum())
    return {
        "n_cards": len(roles),
        "n_tagger_only": int(roles["tagger_primary"].notna().sum() - len(both)),
        "n_heuristic_only": int(roles["heuristic_primary"].notna().sum() - len(both)),
        "n_neither": int(roles["tagger_primary"].isna().sum() - (roles["heuristic_primary"].notna().sum() - len(both))),
        "n_both": len(both),
        "n_agree": agree,
        "agreement_rate": agree / len(both) if len(both) else 0.0,
    }


def unknown_tags(conn: sqlite3.Connection) -> set[str]:
    """Tags named in `TAG_RULES` that don't exist in `card_tags`.
    A typo'd slug contributes silently nothing, which is exactly the kind of bug the 6.3
    devlogs' "unit tests on mechanics don't substitute for checking real output" lesson is
    about — so it gets its own check, run by the CLI and by a test against the live DB."""
    live = {tag for (tag,) in conn.execute("SELECT DISTINCT tag FROM card_tags")}
    named = {tag for rules in TAG_RULES.values() for tag in rules}
    return named - live


def build_and_save(conn: sqlite3.Connection, *, out_dir: Path = KB_DEV_DIR) -> dict[str, object]:
    """Classify every commander-legal card and write `roles.parquet`. The returned summary is the
    health check worth reading after a rebuild: how many cards got a secondary role, how many fell
    through to the `default` role (nothing matched), and how often the two layers agreed where
    both had an opinion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = assign_roles(conn)
    roles.to_parquet(out_dir / "roles.parquet", index=False)
    return {
        "n_cards": len(roles),
        "n_with_secondary": int(roles["secondary_role"].notna().sum()),
        "n_default": int((roles["primary_source"] == "default").sum()),
        **{k: v for k, v in layer_agreement(roles).items() if k in {"n_both", "agreement_rate"}},
    }


def load_roles(out_dir: Path = KB_DEV_DIR) -> pd.DataFrame:
    """`roles.parquet`: one row per commander-legal card with its primary/secondary role and the layer
    that assigned each."""
    return pd.read_parquet(out_dir / "roles.parquet")


# --------------------------------------------------------------------------------------
# Labeled spot-check (plan task 6.4's accuracy requirement)
# --------------------------------------------------------------------------------------

# Well-known Commander cards with the primary role a player would name for them, plus an
# expected secondary where the card genuinely has one.
#
# `calibrated=True` marks the cards whose tag sets were read while *writing* the rules -- the
# mapping was tuned to get these right, so they are not independent evidence. Accuracy is
# reported both overall and held-out, and **the held-out number is the real one**.
#
# Lands are expected to be `land` primary by design (see the module docstring); their functional
# role is checked as the secondary instead.
SpotCheckRow = tuple[str, str, str | None, bool]

SPOT_CHECK: tuple[SpotCheckRow, ...] = (
    # (name, expected_primary, expected_secondary, calibrated)
    # -- mana_acceleration (renamed from `ramp`: it covers rituals and cost reducers too)
    ("Sol Ring", "mana_acceleration", None, True),
    ("Arcane Signet", "mana_acceleration", None, True),
    ("Cultivate", "mana_acceleration", None, True),
    ("Llanowar Elves", "mana_acceleration", None, True),
    ("Kodama's Reach", "mana_acceleration", None, False),
    ("Rampant Growth", "mana_acceleration", None, False),
    ("Birds of Paradise", "mana_acceleration", None, False),
    ("Farseek", "mana_acceleration", None, False),
    ("Nature's Lore", "mana_acceleration", None, False),
    ("Dark Ritual", "mana_acceleration", None, False),
    ("Chromatic Lantern", "mana_acceleration", None, True),
    ("Mind Stone", "mana_acceleration", None, True),          # NOT draw: hand-neutral
    ("Commander's Sphere", "mana_acceleration", None, True),  # NOT draw: hand-neutral
    # cost reducers accelerate exactly like mana sources do
    ("Ruby Medallion", "mana_acceleration", None, True),
    ("Urza's Incubator", "mana_acceleration", None, True),
    ("Goblin Warchief", "mana_acceleration", None, True),
    # -- draw
    ("Rhystic Study", "draw", None, True),
    ("Phyrexian Arena", "draw", None, False),
    ("Harmonize", "draw", None, False),
    ("Night's Whisper", "draw", None, False),
    ("Mystic Remora", "draw", None, False),
    ("Guardian Project", "draw", None, False),
    ("Beast Whisperer", "draw", None, False),
    ("Necropotence", "draw", None, True),
    ("Light Up the Stage", "draw", None, False),    # impulse draw -- no `draw` tag exists
    ("Outpost Siege", "draw", None, False),
    ("Skullclamp", "draw", None, True),
    # loot / rummage count as draw: one card reaching several others
    ("Faithless Looting", "draw", None, True),
    ("Merfolk Looter", "draw", None, True),
    ("Frantic Search", "draw", None, True),
    ("Thrill of Possibility", "draw", None, False),
    # -- NOT draw: cantrips replace themselves and are card-neutral
    ("Ponder", "other", None, True),
    ("Brainstorm", "other", None, False),
    ("Preordain", "other", None, False),
    # -- spot_removal
    ("Swords to Plowshares", "spot_removal", None, True),
    ("Path to Exile", "spot_removal", "mana_acceleration", True),
    ("Beast Within", "spot_removal", None, True),
    ("Chaos Warp", "spot_removal", None, True),
    ("Generous Gift", "spot_removal", None, False),
    ("Go for the Throat", "spot_removal", None, False),
    ("Anguished Unmaking", "spot_removal", None, False),
    ("Krosan Grip", "spot_removal", None, False),
    ("Lightning Bolt", "spot_removal", None, True),
    ("Acidic Slime", "spot_removal", None, True),     # hits land, but also artifact/enchantment
    # -- boardwipe
    ("Wrath of God", "boardwipe", None, True),
    ("Blasphemous Act", "boardwipe", None, True),
    ("Cyclonic Rift", "boardwipe", "spot_removal", True),
    ("Damnation", "boardwipe", None, False),
    ("Toxic Deluge", "boardwipe", None, False),
    ("Austere Command", "boardwipe", None, False),
    ("Farewell", "boardwipe", "graveyard_hate", False),
    # -- stax, including mass land destruction AND land destruction with nothing given back
    ("Winter Orb", "stax", None, True),
    ("Blood Moon", "stax", None, False),
    ("Thalia, Guardian of Thraben", "stax", None, False),
    ("Sphere of Resistance", "stax", None, False),
    ("Armageddon", "stax", None, True),
    ("Ravages of War", "stax", None, True),
    ("Jokulhaups", "stax", "boardwipe", True),
    ("Catastrophe", "stax", None, False),
    ("Decree of Annihilation", "stax", "boardwipe", True),
    ("Stone Rain", "stax", None, True),          # destroys a land, gives nothing back
    ("Molten Rain", "stax", None, True),
    # ...but land destruction that REPLACES the land is removal, not stax
    ("Demolition Field", "land", "spot_removal", True),
    ("Ghost Quarter", "land", "spot_removal", True),
    # -- stack_interaction (broader than countering)
    ("Counterspell", "stack_interaction", None, True),
    ("Swan Song", "stack_interaction", None, False),
    ("Negate", "stack_interaction", None, False),
    ("Arcane Denial", "stack_interaction", None, False),
    ("Dovin's Veto", "stack_interaction", None, False),
    ("Fork", "stack_interaction", None, True),
    ("Reiterate", "stack_interaction", None, False),
    ("Deflecting Swat", "stack_interaction", None, True),
    ("Vedalken Orrery", "stack_interaction", None, True),
    ("Grand Abolisher", "stack_interaction", None, True),
    # -- tutor
    ("Demonic Tutor", "tutor", None, True),
    ("Vampiric Tutor", "tutor", None, False),
    ("Enlightened Tutor", "tutor", None, False),
    ("Worldly Tutor", "tutor", None, False),
    ("Idyllic Tutor", "tutor", None, False),
    ("Expedition Map", "tutor", None, False),
    ("Entomb", "tutor", None, False),
    # -- sacrifice_outlet
    ("Ashnod's Altar", "sacrifice_outlet", None, True),
    ("Viscera Seer", "sacrifice_outlet", None, True),
    ("Phyrexian Altar", "sacrifice_outlet", None, True),
    ("Altar of Dementia", "sacrifice_outlet", None, False),
    ("Carrion Feeder", "sacrifice_outlet", "board_presence", True),   # NOT evasion
    # -- recursion
    ("Eternal Witness", "recursion", None, True),
    ("Regrowth", "recursion", None, False),
    ("Reanimate", "recursion", None, False),
    ("Animate Dead", "recursion", None, False),
    ("Sun Titan", "recursion", None, False),
    # -- graveyard_hate
    ("Relic of Progenitus", "graveyard_hate", None, False),
    ("Rest in Peace", "graveyard_hate", None, False),
    ("Soul-Guide Lantern", "graveyard_hate", None, False),
    # -- protection: only your PERMANENTS
    ("Heroic Intervention", "protection", None, True),
    ("Lightning Greaves", "protection", None, True),
    ("Swiftfoot Boots", "protection", None, False),
    # -- defensive: keeping YOU alive
    ("Fog", "defensive", None, True),
    ("Constant Mists", "defensive", None, True),
    ("Ghostly Prison", "defensive", "stax", True),
    ("Propaganda", "defensive", "stax", True),
    ("Moment's Peace", "defensive", None, False),
    ("Darkness", "defensive", None, False),
    ("Solitary Confinement", "defensive", None, True),   # NOT protection: protects the player
    ("Soul Warden", "defensive", None, True),            # NOT stax: `tax` is not a cost
    # accepted as 50/50 by the user 2026-08-23 -- both readings are defensible, so these are
    # labeled to the tie-break's own answer and flagged calibrated rather than counted as misses
    ("Teferi's Protection", "defensive", None, True),
    ("Whispersilk Cloak", "protection", "evasion", True),
    # -- evasion
    ("Rogue Class", "evasion", None, False),
    ("Rogue's Passage", "land", "evasion", True),
    # -- wincon
    ("Thassa's Oracle", "wincon", None, True),
    ("Craterhoof Behemoth", "wincon", None, True),
    ("Approach of the Second Sun", "wincon", None, False),
    ("Triumph of the Hordes", "wincon", None, False),
    ("Overrun", "wincon", None, True),                # one-shot board buff, not board_presence
    # damage aimed at players closes games
    ("Impact Tremors", "wincon", None, True),
    ("Goblin War Strike", "wincon", None, True),
    ("Purphoros, God of the Forge", "wincon", None, True),
    # extra turns and extra combats
    ("Time Warp", "wincon", None, False),
    ("Aggravated Assault", "wincon", None, True),
    ("Great Train Heist", "wincon", None, True),
    # lords are wincons only with a massive buff or evasion
    ("Coat of Arms", "wincon", None, True),
    ("Eldrazi Monument", "wincon", None, True),
    ("Lord of the Accursed", "wincon", None, True),
    ("Shared Animosity", "wincon", None, True),
    ("Beastmaster Ascension", "wincon", None, True),
    # -- board_presence: token makers, lords, counter distributors, big bodies
    ("Krenko, Mob Boss", "board_presence", None, True),
    ("Adeline, Resplendent Cathar", "board_presence", None, True),
    ("Rampaging Baloths", "board_presence", None, True),
    ("Call the Coppercoats", "board_presence", None, False),
    ("Imperious Perfect", "board_presence", None, True),   # a plain lord -- NOT a wincon
    ("Patchwork Banner", "board_presence", None, True),
    ("Ghalta, Primal Hunger", "board_presence", None, True),   # a big body of its own
    ("Taurean Mauler", "board_presence", None, True),          # a body that grows
    ("Thalia's Lieutenant", "board_presence", None, True),
    # -- enhancer: doublers, increasers, extra phases, counter manipulation
    ("Doubling Season", "enhancer", None, True),
    ("Panharmonicon", "enhancer", None, True),
    ("Hardened Scales", "enhancer", None, True),
    ("Branching Evolution", "enhancer", None, True),
    ("The Ozolith", "enhancer", None, True),
    ("Thousand-Year Elixir", "enhancer", None, True),
    ("Paradox Haze", "enhancer", None, True),
    # -- land (type-line override; the functional role is checked as the secondary)
    ("Command Tower", "land", None, True),
    ("Bojuka Bog", "land", "graveyard_hate", True),
    ("Ancient Tomb", "land", "mana_acceleration", False),
    ("Reliquary Tower", "land", None, False),
    ("Strip Mine", "land", "stax", True),
    ("Wasteland", "land", "stax", True),
    ("Cavern of Souls", "land", "stack_interaction", True),
    # -- other (the only default; nothing on the role list fits)
    ("Bonesplitter", "other", None, True),          # generic equipment is NOT board_presence
    ("Prismatic Omen", "other", None, True),        # pure colour fixing is not acceleration
)


def resolve_card(conn: sqlite3.Connection, name: str) -> str | None:
    """`oracle_id` for a card name (punctuation- and case-insensitive, via `card_names`), or None."""
    row = conn.execute(
        "SELECT oracle_id FROM card_names WHERE name_normalized = ?", (normalize_name(name),)
    ).fetchone()
    return row[0] if row else None


def evaluate_spot_check(conn: sqlite3.Connection, roles: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per `SPOT_CHECK` card: expected vs. actual primary/secondary, `primary_ok`,
    `secondary_ok`, and the `calibrated` flag. `secondary_ok` is only meaningful where an
    expected secondary was declared — it is `NA` elsewhere, rather than silently counting a
    card with no expectation as a pass."""
    if roles is None:
        roles = assign_roles(conn)
    by_id = roles.set_index("oracle_id")

    records = []
    for name, expected_primary, expected_secondary, calibrated in SPOT_CHECK:
        oracle_id = resolve_card(conn, name)
        row = by_id.loc[oracle_id] if oracle_id in by_id.index else None
        actual_primary = row["primary_role"] if row is not None else None
        actual_secondary = row["secondary_role"] if row is not None else None
        records.append(
            {
                "name": name,
                "calibrated": calibrated,
                "expected_primary": expected_primary,
                "actual_primary": actual_primary,
                "primary_ok": actual_primary == expected_primary,
                "expected_secondary": expected_secondary,
                "actual_secondary": actual_secondary,
                "secondary_ok": (
                    pd.NA if expected_secondary is None else actual_secondary == expected_secondary
                ),
                "primary_source": row["primary_source"] if row is not None else None,
                "resolved": oracle_id is not None,
            }
        )
    return pd.DataFrame(records)


def spot_check_accuracy(results: pd.DataFrame) -> dict[str, float | int]:
    """Summarize `evaluate_spot_check`'s per-card table. The honest number is `held_out_accuracy`:
    accuracy over only the cards *not* used to calibrate the rules — overall `primary_accuracy`
    includes cards the rules were tuned on and reads optimistically."""
    held_out = results.loc[~results["calibrated"]]
    with_secondary = results.loc[results["expected_secondary"].notna()]
    return {
        "n": len(results),
        "n_unresolved": int((~results["resolved"]).sum()),
        "primary_correct": int(results["primary_ok"].sum()),
        "primary_accuracy": float(results["primary_ok"].mean()),
        "n_held_out": len(held_out),
        "held_out_correct": int(held_out["primary_ok"].sum()),
        "held_out_accuracy": float(held_out["primary_ok"].mean()) if len(held_out) else 0.0,
        "n_expected_secondary": len(with_secondary),
        "secondary_correct": int(with_secondary["secondary_ok"].sum()),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        missing = unknown_tags(conn)
        if missing:
            print(f"WARNING: {len(missing)} mapped tag(s) not in card_tags: {sorted(missing)}")
        stats = build_and_save(conn)
    print(f"Wrote roles.parquet to {KB_DEV_DIR}:")
    print(json.dumps(stats, indent=2, default=str))


def _cmd_dist(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        roles = assign_roles(conn)
    print(role_distribution(roles).to_string(index=False))
    print()
    print("Layer agreement (tagger vs heuristic):")
    print(json.dumps(layer_agreement(roles), indent=2, default=str))


def _cmd_spotcheck(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        roles = assign_roles(conn)
        results = evaluate_spot_check(conn, roles)
    summary = spot_check_accuracy(results)
    wrong = results.loc[~results["primary_ok"]]
    if not wrong.empty:
        print("Misclassified:")
        print(
            wrong[["name", "calibrated", "expected_primary", "actual_primary", "primary_source"]]
            .to_string(index=False)
        )
        print()
    bad_secondary = results.loc[results["secondary_ok"] == False]  # noqa: E712 -- NA-safe
    if not bad_secondary.empty:
        print("Wrong secondary:")
        print(
            bad_secondary[["name", "expected_secondary", "actual_secondary"]].to_string(index=False)
        )
        print()
    print(json.dumps(summary, indent=2, default=str))


def _cmd_show(args: argparse.Namespace) -> None:
    with connect(CONFIG.paths.db_path) as conn:
        oracle_id = resolve_card(conn, args.card)
        if oracle_id is None:
            raise SystemExit(f"Card {args.card!r} not found in card_names.")
        name, type_line, oracle_text, power, toughness = conn.execute(
            "SELECT name, type_line, oracle_text, power, toughness FROM cards WHERE oracle_id = ?",
            (oracle_id,),
        ).fetchone()
        tags = [t for (t,) in conn.execute(
            "SELECT tag FROM card_tags WHERE oracle_id = ? AND source = 'tagger_bulk' ORDER BY tag",
            (oracle_id,),
        )]

    assignment = classify(name=name, type_line=type_line, oracle_text=oracle_text, tags=tags,
                          power=power, toughness=toughness)
    print(f"{name}  [{type_line}]" + (f"  {power}/{toughness}" if power else ""))
    print(f"  primary   : {assignment.primary} ({assignment.primary_score:.1f}, {assignment.primary_source})")
    if assignment.secondary:
        print(
            f"  secondary : {assignment.secondary} "
            f"({assignment.secondary_score:.1f}, {assignment.secondary_source})"
        )
    print(f"  tag scores : {score_tags(tags)}")
    print(f"  text scores: {score_text(name, type_line, oracle_text)}")
    print(f"  tags ({len(tags)}): {', '.join(tags)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build & save roles.parquet to data/kb/dev/")
    build_p.set_defaults(func=_cmd_build)

    dist_p = sub.add_parser("dist", help="Role distribution + tagger/heuristic agreement")
    dist_p.set_defaults(func=_cmd_dist)

    spot_p = sub.add_parser("spotcheck", help="Run the 60-card labeled spot-check")
    spot_p.set_defaults(func=_cmd_spotcheck)

    show_p = sub.add_parser("show", help="Explain one card's classification")
    show_p.add_argument("card", help="Card name")
    show_p.set_defaults(func=_cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
