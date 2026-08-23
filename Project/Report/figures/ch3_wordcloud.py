"""Word cloud over every card's rules text - Magic writes in its own dialect.

The point for the report: standard English stopword lists remove "the" and "of", but they have
nothing to say about "target", "creature" or "battlefield", which appear on a large share of all
cards and therefore carry almost no signal about what any one card does.
"""
import sys, re, sqlite3, collections
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
from edhcut.config import CONFIG

OUT = Path(__file__).resolve().parents[1] / "Images"   # Project/Report/Images

con = sqlite3.connect(CONFIG.paths.db_path)
texts = [t for (t,) in con.execute(
    "SELECT oracle_text FROM cards WHERE legal_commander = 1 AND oracle_text IS NOT NULL "
    "AND oracle_text != ''")]
print(f"{len(texts):,} cards with rules text")

TOKEN = re.compile(r"[a-z][a-z'-]+")
counts = collections.Counter()
doc_freq = collections.Counter()
for t in texts:
    words = TOKEN.findall(t.lower())
    counts.update(words)
    doc_freq.update(set(words))

drop = set(STOPWORDS) | {"s", "t", "you", "your", "it", "its", "this", "that", "may", "if",
                         "when", "whenever", "each", "any", "all", "other", "another", "than",
                         "up", "one", "two", "three", "end", "until", "put", "get", "gets",
                         "have", "has", "had", "can", "cannot", "as", "for", "with", "the"}
freq = {w: c for w, c in counts.items() if w not in drop and len(w) > 2 and c >= 40}

n = len(texts)
print("\nmost widespread words, by share of cards carrying them:")
for w, d in sorted(doc_freq.items(), key=lambda kv: -kv[1])[:60]:
    if w in freq:
        print(f"  {w:16s} {d / n:6.1%} of cards")

wc = WordCloud(width=1500, height=780, background_color="white", colormap="viridis",
               prefer_horizontal=0.92, max_words=110, random_state=1,
               relative_scaling=0.45).generate_from_frequencies(freq)
fig, ax = plt.subplots(figsize=(9.4, 4.9), dpi=150)
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Magic writes in its own dialect, and its commonest words are not English stopwords",
             fontsize=11.5, pad=10)
fig.savefig(OUT / "fig19_oracle_wordcloud.png", bbox_inches="tight", facecolor="white")
print("\nwrote fig19_oracle_wordcloud.png")
