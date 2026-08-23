"""Chapter 2 card-image rows: the three dragon packages, and one card's multiple memberships."""
import sys, pickle
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
from cards import card, grid, _font, CARD_W, PAD, BG, OUT
from edhcut.db import connect
from edhcut.config import CONFIG

D = pickle.load(open(SP / "ch2.pkl", "rb"))
cards, share, floored, names = D["cards"], D["share"], D["floored"], D["names"]

WIDE = 250

# Written from the members, not from the tag labels -- the tags say "dragon" three times over,
# which is exactly the thing the figure needs to get past.
DRAGON_BLURB = {
    22:  "aggressive red dragons\nthat deal damage the\nturn they land",
    128: "big green/red/blue\ndragons that ramp into\nenormous bodies",
    70:  "blue/black dragons that\ncontrol the board and\nrecur themselves",
}


def top_cards(cid, n=5, exclude=()):
    """The `n` most-played members of community `cid`, by deck count."""
    m = cards[(cards["community"] == cid) & (~cards["name"].isin(exclude))]
    return m.nlargest(n, "deck_count")["name"].tolist()


def dragons():
    """**Figure 10.** The most-played members of each of the three dragon packages — the same tag
    ('dragon') three times over, split by what the decks actually do with them."""
    ids = [c for c, nm in names.items() if "dragon" in nm]
    ids.sort(key=lambda c: -(cards["community"] == c).sum())
    rows = [top_cards(c, 5) for c in ids]
    labels = [f"{DRAGON_BLURB.get(c, names[c])}\n\n({int((cards['community'] == c).sum())} cards)"
              for c in ids]
    with connect(CONFIG.paths.db_path) as conn:
        img = grid(conn, rows, labels,
                   title="Dragons are not one package but three, split by how the deck plays",
                   label_w=WIDE)
    img.save(OUT / "cards3_dragons.png")
    print("wrote cards3_dragons.png")


def multi_membership(query="Mondrak, Glory Dominus", blurbs=None, out="cards4_multi_membership.png"):
    """**Figure 11.** One card and every package it belongs to, with its share of membership in each —
    the overlapping membership a hard partition cannot express."""
    i = cards.index[cards["name"] == query][0]
    tops = [t for t in np.argsort(-share[i])[:5] if floored[i, t]]
    rows, labels = [], []
    for t in tops:
        rows.append(top_cards(t, 4, exclude=(query,)))
        blurb = (blurbs or {}).get(t, names.get(t, "(generic)").split("/")[0])
        labels.append(f"{share[i, t]:.0%}\n{blurb}")
    with connect(CONFIG.paths.db_path) as conn:
        body = grid(conn, rows, labels, title=None, label_w=WIDE)
        head = card(conn, query)
        title_h, gap = 36, 14
        canvas = Image.new("RGB", (max(body.width, WIDE + CARD_W + PAD),
                                   title_h + head.height + gap + body.height), BG)
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 10), "One card, three packages - one for each thing the card does",
               fill=(20, 20, 20), font=_font(17))
        canvas.paste(head, (WIDE, title_h))
        d.text((PAD, title_h + head.height // 2 - 9), query.split(",")[0],
               fill=(40, 40, 40), font=_font(15))
        canvas.paste(body, (0, title_h + head.height + gap))
    canvas.save(OUT / out)
    print(f"wrote {out}")
    for l, r in zip(labels, rows):
        print("   ", l.replace(chr(10), " | "), "->", ", ".join(r))


if __name__ == "__main__":
    dragons()
    i = cards.index[cards["name"] == "Mondrak, Glory Dominus"][0]
    ts = [t for t in np.argsort(-share[i])[:5] if floored[i, t]]
    multi_membership(blurbs={
        ts[0]: "token payoffs\n(it doubles your tokens)",
        ts[1]: "powerful staples\n(it shields itself)",
        ts[2]: "Phyrexian typal\n(it is a Phyrexian)",
    })
