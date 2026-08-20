"""One card's neighbours under text similarity and under co-occurrence, side by side."""
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
import kb
from cards import card, grid, _font, CARD_W, PAD, BG, OUT
from edhcut.db import connect
from edhcut.config import CONFIG

QUERY = "Skullclamp"
WIDE = 250


def neighbours():
    ci, tscore, lift, n2r, r2n, dc = kb.load()
    emb = pd.read_parquet("data/kb/dev/embeddings.parquet")
    text_cols = [c for c in emb.columns if c.startswith(("tfidf_", "types_", "struct_"))]
    emb = emb[emb["oracle_id"].isin(set(ci["oracle_id"]))].reset_index(drop=True)
    vec = np.array(emb[text_cols].to_numpy(dtype=np.float32), copy=True)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)

    oid_of_name = dict(zip(ci["name"], ci["oracle_id"]))
    name_of_oid = dict(zip(emb["oracle_id"], emb["name"]))
    is_land = dict(zip(emb["oracle_id"], emb["is_land"]))
    erow = {o: k for k, o in enumerate(emb["oracle_id"])}

    q_oid = oid_of_name[QUERY]
    sims = vec @ vec[erow[q_oid]]
    order = np.argsort(-sims)
    text_list = []
    for k in order:
        o = emb["oracle_id"].iloc[k]
        if o == q_oid or is_land.get(o):
            continue
        text_list.append(name_of_oid[o])
        if len(text_list) == 4:
            break

    r = n2r[QUERY]
    tr = tscore.getrow(r).toarray().ravel()
    lr = lift.getrow(r).toarray().ravel()
    comb = tr * np.log1p(np.maximum(lr, 0))
    land_row = dict(zip(ci["row"], ci["is_land"]))
    co_list = []
    for i in np.argsort(-comb):
        if i == r or land_row.get(i):
            continue
        co_list.append(r2n[i])
        if len(co_list) == 4:
            break
    return text_list, co_list


if __name__ == "__main__":
    text_list, co_list = neighbours()
    print("text        :", ", ".join(text_list))
    print("co-occurrence:", ", ".join(co_list))
    rows = [text_list, co_list]
    labels = ["what else\ndoes this?\n\n(text similarity)",
              "what goes\nwith this?\n\n(co-occurrence)"]
    with connect(CONFIG.paths.db_path) as conn:
        body = grid(conn, rows, labels, title=None, label_w=WIDE)
        head = card(conn, QUERY)
        title_h, gap = 36, 14
        canvas = Image.new("RGB", (max(body.width, WIDE + CARD_W + PAD),
                                   title_h + head.height + gap + body.height), BG)
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 10), "Two different questions, two completely different answers",
               fill=(20, 20, 20), font=_font(17))
        canvas.paste(head, (WIDE, title_h))
        d.text((PAD, title_h + head.height // 2 - 9), QUERY, fill=(40, 40, 40), font=_font(15))
        canvas.paste(body, (0, title_h + head.height + gap))
    canvas.save(OUT / "cards5_substitutes_vs_complements.png")
    print("wrote cards5_substitutes_vs_complements.png")
