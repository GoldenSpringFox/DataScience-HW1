"""Card-image rows for the report."""
import sys
from pathlib import Path
import numpy as np, scipy.sparse as sparse
from PIL import Image, ImageDraw, ImageFont

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb
from edhcut.db import connect
from edhcut.config import CONFIG
from edhcut.images import get_card_image

OUT = Path(r"C:/Aviv/University/Semester 8/Data Science/Homework - Group/Project/Report/Images")
CARD_W, PAD, LABEL_W = 200, 10, 150
BG = (255, 255, 255)


def _font(size):
    for name in ("seguisb.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def card(conn, name, width=CARD_W):
    img = get_card_image(conn, name).convert("RGB")
    h = int(img.height * width / img.width)
    return img.resize((width, h), Image.LANCZOS)


def grid(conn, rows, row_labels, title=None, label_w=LABEL_W):
    """rows: list of lists of card names. Renders a labelled grid."""
    imgs = [[card(conn, n) for n in r] for r in rows]
    ch = imgs[0][0].height
    ncol = max(len(r) for r in imgs)
    w = label_w + ncol * (CARD_W + PAD) + PAD
    title_h = 34 if title else 0
    h = title_h + len(imgs) * (ch + PAD) + PAD
    canvas = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((PAD, 9), title, fill=(20, 20, 20), font=_font(17))
    for ri, row in enumerate(imgs):
        y = title_h + PAD + ri * (ch + PAD)
        d.text((PAD, y + ch // 2 - 9), row_labels[ri], fill=(40, 40, 40), font=_font(15))
        for cix, im in enumerate(row):
            canvas.paste(im, (label_w + cix * (CARD_W + PAD), y))
    return canvas


def metric_rows(query="Goblin Sharpshooter", out="cards1_metric_comparison.png"):
    ci, tscore_w, lift, n2r, r2n, dc = kb.load()
    r = n2r[query]
    co = sparse.load_npz("data/kb/dev/cooccur_global.npz").tocsr()
    cr = co.getrow(r).toarray().ravel()
    lr = lift.getrow(r).toarray().ravel()
    tr = tscore_w.getrow(r).toarray().ravel()
    comb = tr * np.log1p(np.maximum(lr, 0))
    rows, labels = [], []
    for label, vec in [("co-occurrence", cr), ("lift", lr), ("t-score", tr),
                       ("synergy score", comb)]:
        top = np.argsort(-vec)[:5]
        rows.append([r2n[i] for i in top])
        labels.append(label)
    with connect(CONFIG.paths.db_path) as conn:
        body = grid(conn, rows, labels, title=None)
        head = card(conn, query, width=CARD_W)
        w = max(body.width, LABEL_W + CARD_W + PAD)
        title_h, gap = 34, 16
        canvas = Image.new("RGB", (w, title_h + head.height + gap + body.height), BG)
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 9), f"{query}: five strongest partners under each metric",
               fill=(20, 20, 20), font=_font(17))
        canvas.paste(head, (LABEL_W, title_h))
        d.text((PAD, title_h + head.height // 2 - 9), "the card", fill=(40, 40, 40), font=_font(15))
        canvas.paste(body, (0, title_h + head.height + gap))
    canvas.save(OUT / out)
    print(f"wrote {out}")
    for l, r_ in zip(labels, rows):
        print(f"  {l:16s} {', '.join(r_)}")


def keep_drop_examples():
    rows = [
        ["Sanguine Bond", "Exquisite Blood"],
        ["Viscera Seer", "Zulaport Cutthroat"],
        ["Evolving Wilds", "Scute Swarm"],
        ["Abrade", "Lightning Bolt"],
        ["Command Tower", "Beast Within"],
    ]
    labels = ["synergy", "synergy", "synergy", "substitutes", "no relation"]
    with connect(CONFIG.paths.db_path) as conn:
        img = grid(conn, rows, labels,
                   title="Hand-labelled pairs: what counts as synergy, and what does not")
    img.save(OUT / "cards2_keep_drop.png")
    print("wrote cards2_keep_drop.png")


if __name__ == "__main__":
    metric_rows()
    keep_drop_examples()
