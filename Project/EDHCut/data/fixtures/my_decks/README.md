# `data/fixtures/my_decks/` — real decklist fixtures

One plain-text file per commander slot with a real user list (plan §1), used as golden
fixtures for the parser (task 7.1), QA report (task 5.6), and evaluation harness (task 7.5).

## Expected files

| File | Commander slot |
|---|---|
| `kyler.txt` | Kyler, Sigardian Emissary |
| `yoshimaru_bruse.txt` | Yoshimaru, Ever Faithful + Bruse Tarl, Boorish Herder (partner pair) |
| `yenna.txt` | Yenna, Redtooth Regent (fox-themed build) |
| `orysa.txt` | Orysa, Tide Choreographer |

Slot 1 (Krenko, Mob Boss) has no fixture here — it's the spike commander (plan §10), no real
user list exists for it.

## File format

Plain text, one card per line. The parser (task 7.1) is expected to accept:

- Bare names: `Sol Ring`
- Quantity-prefixed: `1 Sol Ring` or `1x Sol Ring`
- Archidekt/Moxfield text-export category headers (e.g. `Commander`, `Creature`, `Land`) —
  ignored as structure, cards still parsed line by line
- Commander markers: a line containing `*CMDR*` or a `Commander:` prefix marks the
  following card(s) as commander(s) rather than deck cards — required for partner pairs
  (two marked lines)
- MTGO-style exports

Comment lines starting with `#` and blank lines are ignored.

## Notes

- Every card in these files must resolve to an `oracle_id` via `card_names` — the QA report
  (task 5.6) and parser tests (task 7.1) fail loudly if one doesn't.
- These are the user's real, tried-and-tested lists — do not "clean them up" or replace them
  with meta lists. Yenna in particular is intentionally an off-meta fox build (plan §1).
