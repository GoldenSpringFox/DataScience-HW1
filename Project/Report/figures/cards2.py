"""Card-image figures: metric comparison for one card, and top-pair panels for lift / t-score."""
import sys
from pathlib import Path
import numpy as np, scipy.sparse as sparse
from PIL import Image, ImageDraw

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb
from cards import card, grid, _font, CARD_W, PAD, LABEL_W, BG, OUT
from edhcut.db import connect
from edhcut.config import CONFIG

ci, tscore, lift, n2r, r2n, dc = kb.load()
counts = ci.set_index("row")["deck_count"].sort_index().values
co = sparse.load_npz("data/kb/dev/cooccur_global.npz").tocsr()


def _top_pairs(matrix, n, exclude_names=()):
    excl = {n2r[x] for x in exclude_names if x in n2r}
    m = sparse.triu(matrix, k=1).tocoo()
    out = []
    for i in np.argsort(-m.data):
        u, v = m.row[i], m.col[i]
        if u in excl or v in excl:
            continue
        out.append((u, v, m.data[i]))
        if len(out) == n:
            return out
    return out


def pair_panel(rows, title, out, note_fmt):
    """rows: [(name_a, name_b, note_args...)] rendered as one pair per line."""
    with connect(CONFIG.paths.db_path) as conn:
        imgs = [(card(conn, a), card(conn, b)) for a, b, *_ in rows]
        ch = imgs[0][0].height
        note_w = 250
        w = PAD + 2 * (CARD_W + PAD) + note_w
        title_h = 36
        h = title_h + len(rows) * (ch + PAD) + PAD
        canvas = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 10), title, fill=(20, 20, 20), font=_font(17))
        for k, ((ia, ib), r) in enumerate(zip(imgs, rows)):
            y = title_h + PAD + k * (ch + PAD)
            canvas.paste(ia, (PAD, y))
            canvas.paste(ib, (PAD + CARD_W + PAD, y))
            for li, line in enumerate(note_fmt(r)):
                d.text((PAD + 2 * (CARD_W + PAD) + 6, y + ch // 2 - 20 + li * 20),
                       line, fill=(45, 45, 45), font=_font(15))
    canvas.save(OUT / out)
    print("wrote", out)


def lift_panel():
    top = _top_pairs(lift, 4)
    rows = [(r2n[i], r2n[j], float(v), int(co[i, j]), int(counts[i]), int(counts[j]))
            for i, j, v in top]
    pair_panel(rows,
               "Lift's highest-scoring pairs in all 13,207 decks",
               "fig2_lift_fails.png",
               lambda r: [f"lift = {r[2]:,.0f}", f"but only {r[3]} decks play both",
                          f"({r[4]} and {r[5]} decks each)"])
    for r in rows:
        print("   ", r)


def tscore_panel():
    top = _top_pairs(tscore, 4, exclude_names=kb.BASICS)
    rows = [(r2n[i], r2n[j], float(v), int(co[i, j]), int(counts[i]), int(counts[j]))
            for i, j, v in top]
    pair_panel(rows,
               "t-score's highest-scoring pairs in all 13,207 decks",
               "fig3_tscore_fails.png",
               lambda r: [f"t-score = {r[2]:,.1f}", f"{r[3]:,} decks play both",
                          f"({r[4]:,} and {r[5]:,} decks each)"])
    for r in rows:
        print("   ", r)


def metric_rows(query="Basalt Monolith", out="cards1_metric_comparison.png"):
    r = n2r[query]
    cr = co.getrow(r).toarray().ravel()
    lr = lift.getrow(r).toarray().ravel()
    tr = tscore.getrow(r).toarray().ravel()
    comb = tr * np.log1p(np.maximum(lr, 0))
    verdict = {
        "co-occurrence": "just the generic\nstaples",
        "lift": "far too specific -\ncards nobody plays",
        "t-score": "related, but still\ntoo generic",
        "synergy score": "actually relevant\nresults",
    }
    rows, labels = [], []
    for label, vec in [("co-occurrence", cr), ("lift", lr), ("t-score", tr),
                       ("synergy score", comb)]:
        rows.append([r2n[i] for i in np.argsort(-vec)[:5]])
        labels.append(f"{label}\n\n{verdict[label]}")
    with connect(CONFIG.paths.db_path) as conn:
        body = grid(conn, rows, labels, title=None, label_w=210)
        head = card(conn, query)
        title_h, gap = 36, 16
        canvas = Image.new("RGB", (max(body.width, 210 + CARD_W + PAD),
                                   title_h + head.height + gap + body.height), BG)
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 10), f"{query}: five strongest partners under each metric",
               fill=(20, 20, 20), font=_font(17))
        canvas.paste(head, (210, title_h))
        d.text((PAD, title_h + head.height // 2 - 9), "the card", fill=(40, 40, 40), font=_font(15))
        canvas.paste(body, (0, title_h + head.height + gap))
    canvas.save(OUT / out)
    print("wrote", out)
    for l, r_ in zip(labels, rows):
        print(f"  {l:16s} {', '.join(r_)}")


if __name__ == "__main__":
    metric_rows(); lift_panel(); tscore_panel()
