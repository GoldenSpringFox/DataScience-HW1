from pathlib import Path
from collections import Counter
import re

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import sent_tokenize, word_tokenize, pos_tag
from wordcloud import WordCloud


# ----------------------------
# Paths and constants
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TEXT_PATH = DATA_DIR / "alice.txt"

BOOK_NAME = "Alice's Adventures in Wonderland"
AUTHOR = "Lewis Carroll"

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"


# ----------------------------
# Setup
# ----------------------------

def ensure_nltk_resources() -> None:
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]

    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name)


# ----------------------------
# Utility functions
# ----------------------------

def load_book_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text.split(START_MARKER)[1].split(END_MARKER)[0].strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+(?:['’][a-zA-Z]+)?\b", text.lower())


def print_basic_stats(title: str, tokens: list[str], counts: Counter) -> None:
    print(f"\n{title}")
    print("Total tokens:", len(tokens))
    print("Unique tokens:", len(counts))


def print_top_20(title: str, counts: Counter) -> None:
    print(f"\n{title}")
    for token, count in counts.most_common(20):
        print(f"{token}: {count}")


def plot_log_rank_frequency(counts: Counter, title: str, output_filename: str) -> None:
    frequencies = sorted(counts.values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)

    plt.figure()
    plt.scatter(ranks, frequencies, s=8, alpha=0.7)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Log Rank")
    plt.ylabel("Log Frequency")
    plt.title(title)

    plt.grid(True)

    plt.savefig(OUTPUT_DIR / output_filename, dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------
# Assignment sections
# ----------------------------

def original_text_analysis(tokens: list[str]) -> Counter:
    counts = Counter(tokens)

    print_basic_stats("Original text", tokens, counts)

    plot_log_rank_frequency(
        counts,
        "Token Frequency Distribution - Original Text",
        "original_log_rank_frequency.png"
    )

    print_top_20("Top 20 tokens:", counts)

    return counts


def stopwords_analysis(tokens: list[str]) -> tuple[list[str], Counter]:
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [
        token for token in tokens
        if token not in stop_words
    ]

    counts = Counter(filtered_tokens)

    print_basic_stats("After removing stopwords", filtered_tokens, counts)

    plot_log_rank_frequency(
        counts,
        "Token Frequency Distribution - Stopwords Removed",
        "stopwords_removed_log_rank_frequency.png"
    )

    print_top_20("Top 20 tokens after removing stopwords:", counts)

    return filtered_tokens, counts


def stemming_analysis(tokens_without_stopwords: list[str]) -> tuple[list[str], Counter]:
    stemmer = PorterStemmer()

    stemmed_tokens = [
        stemmer.stem(token)
        for token in tokens_without_stopwords
    ]

    counts = Counter(stemmed_tokens)

    print_basic_stats("After stemming", stemmed_tokens, counts)

    plot_log_rank_frequency(
        counts,
        "Token Frequency Distribution - Stemmed Text",
        "stemmed_log_rank_frequency.png"
    )

    print_top_20("Top 20 stems:", counts)

    return stemmed_tokens, counts


def pos_tagging_examples(text: str) -> None:
    sentences = sent_tokenize(text)

    print("\nPOS tagging examples:")
    print("---------------------")

    for sentence in sentences[:30]:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)

        print("\nSentence:")
        print(sentence)

        print("Tagged:")
        print(tagged_words)


def proper_noun_wordcloud(text: str) -> Counter:
    sentences = sent_tokenize(text)

    proper_nouns = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)

        for word, tag in tagged_words:
            if tag in {"NNP", "NNPS"}:
                clean_word = re.sub(r"[^a-zA-Z]", "", word)

                if clean_word:
                    proper_nouns.append(clean_word)

    counts = Counter(proper_nouns)

    print_top_20("Top 20 proper nouns:", counts)

    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color="white"
    ).generate_from_frequencies(counts)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Proper Noun Word Cloud - Alice's Adventures in Wonderland")

    plt.savefig(
        OUTPUT_DIR / "proper_nouns_wordcloud.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return counts

def repeated_words_regex(text: str) -> list[str]:
    # Two consecutive repeated words, allowing punctuation/whitespace between them
    pattern = r"\b([A-Za-z]{2,})\b[\s,.;:!?-]+\1\b"

    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    print("\nRepeated consecutive words:")
    print("---------------------------")

    if not matches:
        print("No matches found.")
        return []

    found = []

    for match in matches:
        print(match.group(0))
        found.append(match.group(0))

    return found

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_nltk_resources()

    print(f"Book: {BOOK_NAME}")
    print(f"Author: {AUTHOR}")

    text = load_book_text(TEXT_PATH)
    tokens = tokenize(text)

    original_text_analysis(tokens)

    tokens_without_stopwords, _ = stopwords_analysis(tokens)

    stemming_analysis(tokens_without_stopwords)

    pos_tagging_examples(text)

    proper_noun_wordcloud(text)

    repeated_words_regex(text)


if __name__ == "__main__":
    main()