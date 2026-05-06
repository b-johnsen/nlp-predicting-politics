"""
Executive Orders NLP Analysis
==============================
Reads all executive orders from `all_executive_orders_txt_clean/`,
then produces:
  - N-gram frequency analysis (unigrams, bigrams, trigrams)
  - Word cloud images per president and per political party
  - Visual distributions of key phrases (bar charts, heatmaps)

Output goes into  ./eo_analysis_output/
"""

import os
import re
import json
import string
from pathlib import Path
from collections import Counter, defaultdict

import nltk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from wordcloud import WordCloud
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Download required NLTK data ──────────────────────────────────────────────
for pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(pkg, quiet=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR   = Path("nlp-predicting-politics/eo_data/all_executive_orders_txt_clean")
OUTPUT_DIR = Path("nlp-predicting-politics/word analysis/eo_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Map every president folder name → party
PARTY_MAP = {
    # ── Democrat ──────────────────────────────────────────────────────────────
    "Andrew_Jackson":           "Democrat",       # D  1829–1837
    "Martin_van_Buren":         "Democrat",       # D  1837–1841
    "James_K_Polk":             "Democrat",       # D  1845–1849
    "James_Buchanan":           "Democrat",       # D  1857–1861
    "Andrew_Johnson":           "Democrat",       # D (ran on National Union ticket, lifelong Democrat)
    "Grover_Cleveland":         "Democrat",       # D  1885–1889, 1893–1897
    "Woodrow_Wilson":           "Democrat",       # D  1913–1921
    "Franklin_D_Roosevelt":     "Democrat",       # D  1933–1945
    "Harry_S_Truman":           "Democrat",       # D  1945–1953
    "John_F_Kennedy":           "Democrat",       # D  1961–1963
    "Lyndon_B_Johnson":         "Democrat",       # D  1963–1969
    "Jimmy_Carter":             "Democrat",       # D  1977–1981
    "William_J_Clinton":        "Democrat",       # D  1993–2001
    "Barack_Obama":             "Democrat",       # D  2009–2017
    "Joseph_R_Biden_Jr":        "Democrat",       # D  2021–2025

    # ── Republican ────────────────────────────────────────────────────────────
    "Abraham_Lincoln":          "Republican",     # R  1861–1865
    "Ulysses_S_Grant":          "Republican",     # R  1869–1877
    "Rutherford_B_Hayes":       "Republican",     # R  1877–1881
    "James_A_Garfield":         "Republican",     # R  1881
    "Chester_A_Arthur":         "Republican",     # R  1881–1885
    "Benjamin_Harrison":        "Republican",     # R  1889–1893
    "William_McKinley":         "Republican",     # R  1897–1901
    "Theodore_Roosevelt":       "Republican",     # R  1901–1909
    "William_Howard_Taft":      "Republican",     # R  1909–1913
    "Warren_G_Harding":         "Republican",     # R  1921–1923
    "Calvin_Coolidge":          "Republican",     # R  1923–1929
    "Herbert_Hoover":           "Republican",     # R  1929–1933
    "Dwight_D_Eisenhower":      "Republican",     # R  1953–1961
    "Richard_Nixon":            "Republican",     # R  1969–1974
    "Gerald_R_Ford":            "Republican",     # R  1974–1977
    "Ronald_Reagan":            "Republican",     # R  1981–1989
    "George_Bush":              "Republican",     # R  1989–1993  (George H.W. Bush)
    "George_W_Bush":            "Republican",     # R  2001–2009
    "Donald_J_Trump_(1st_Term)":"Republican",     # R  2017–2021
    "Donald_J_Trump_(2nd_Term)":"Republican",     # R  2025–

    # ── Whig ──────────────────────────────────────────────────────────────────
    "John_Tyler":               "Whig",           # Whig (later expelled; ran as independent)
    "Zachary_Taylor":           "Whig",           # Whig 1849–1850
    "Millard_Fillmore":         "Whig",           # Whig 1850–1853
    "Franklin_Pierce":          "Democrat",       # D  1853–1857  (listed here for completeness)

    # ── Democratic-Republican / no-party era ──────────────────────────────────
    "John_Quincy_Adams":        "Democratic-Republican",  # 1825–1829
}

# Party colour palettes
PARTY_COLORS = {
    "Republican": "#C0392B",
    "Democrat":   "#2471A3",
}

# Custom stop-words to prune EO boilerplate
CUSTOM_STOPS = {
    "shall", "section", "pursuant", "thereof", "hereby",
    "whereas", "executive", "order", "united", "states",
    "federal", "government", "authority", "act", "law",
    "title", "part", "accordance", "including", "such",
    "may", "within", "upon", "any", "also", "set",
    "forth", "thereunder", "provided", "therefor", "said",
    "us", "president", "america", "american",  "section", "sec", "subsection", "order", "shall", "may", "must",
    "hereby", "thereof", "therein", "whereas", "therefore",
    "president", "executive", "federal", "government", "agency", "agencies",
    "united", "states", "state", "department",
    "act", "law", "provision", "title", "chapter", "paragraph",
    "provide", "provides", "provided", "including", "include",
    "said", "secretary", "ordered", "approved", "director", "stat", "authority", "pursuant",
    "vested", "amended", "public", "usc", "service", "virtue", "policy", "within"
}


STOP_WORDS = stopwords.words("english")
STOP_WORDS = set(STOP_WORDS) | CUSTOM_STOPS

TOP_N_NGRAMS  = 20   # bars shown per n-gram chart
TOP_N_PHRASES = 15   # phrases shown in heatmap


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(raw: str) -> str:
    """Lowercase, strip punctuation and digits."""
    text = raw.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text)
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]


def get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(" ".join(g) for g in ngrams(tokens, n))


def load_president_texts(data_dir: Path) -> dict[str, str]:
    """Return {president_slug: full_concatenated_text}."""
    president_texts: dict[str, str] = {}
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            "Place this script next to the all_executive_orders_txt_clean/ folder."
        )
    for pdir in sorted(data_dir.iterdir()):
        if not pdir.is_dir():
            continue
        texts = []
        for fpath in sorted(pdir.rglob("*.txt")):
            try:
                texts.append(fpath.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        if texts:
            president_texts[pdir.name] = "\n".join(texts)
            print(f"  Loaded {len(texts):>4} orders  →  {pdir.name}")
    return president_texts


def display_name(slug: str) -> str:
    return slug.replace("-", " ").title()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ngrams(counter: Counter, title: str, color: str, out_path: Path, top_n=TOP_N_NGRAMS):
    items = counter.most_common(top_n)
    if not items:
        return
    labels, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels[::-1], counts[::-1], color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    # value labels
    for bar in bars:
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}", va="center", fontsize=8, color="#444")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_wordcloud(text: str, color: str, out_path: Path, title: str):
    wc = WordCloud(
        width=1400, height=700,
        background_color="white",
        colormap=None,
        color_func=lambda *a, **kw: color,
        stopwords=STOP_WORDS,
        max_words=200,
        prefer_horizontal=0.85,
        collocations=False,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phrase_heatmap(phrase_matrix: pd.DataFrame, title: str, out_path: Path, color: str):
    """Heatmap of top phrases × presidents or parties."""
    fig_h = max(6, len(phrase_matrix) * 0.45)
    fig_w = max(8, len(phrase_matrix.columns) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = sns.light_palette(color, as_cmap=True)
    sns.heatmap(phrase_matrix, ax=ax, cmap=cmap, linewidths=0.4,
                linecolor="#ddd", fmt=".0f", annot=True, annot_kws={"size": 8})
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Phrase", fontsize=10)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("\nLoading executive orders …")
    president_texts = load_president_texts(DATA_DIR)

    # ── tokenise every president ──────────────────────────────────────────────
    print("\nTokenising …")
    president_tokens: dict[str, list[str]] = {}
    for slug, raw in president_texts.items():
        president_tokens[slug] = tokenize(clean_text(raw))

    # ── group by party ────────────────────────────────────────────────────────
    party_tokens: dict[str, list[str]] = defaultdict(list)
    party_raw:    dict[str, str]        = defaultdict(str)
    for slug, tokens in president_tokens.items():
        party = PARTY_MAP.get(slug, "Unknown")
        party_tokens[party].extend(tokens)
        party_raw[party] += " " + president_texts[slug]

    # PER-PRESIDENT ANALYSIS
    print("\nGenerating per-president charts …")
    bigram_all: dict[str, Counter] = {}

    for slug, tokens in president_tokens.items():
        name   = display_name(slug)
        party  = PARTY_MAP.get(slug, "Unknown")
        color  = PARTY_COLORS.get(party, "#555555")
        pout   = OUTPUT_DIR / "presidents" / slug
        pout.mkdir(parents=True, exist_ok=True)

        uni  = get_ngrams(tokens, 1)
        bi   = get_ngrams(tokens, 2)
        tri  = get_ngrams(tokens, 3)
        bigram_all[slug] = bi

        # N-gram bar charts
        plot_ngrams(uni, f"{name}  —  Top Unigrams",  color, pout / "unigrams.png")
        plot_ngrams(bi,  f"{name}  —  Top Bigrams",   color, pout / "bigrams.png")
        plot_ngrams(tri, f"{name}  —  Top Trigrams",  color, pout / "trigrams.png")

        # Word cloud
        make_wordcloud(president_texts[slug], color, pout / "wordcloud.png",
                       f"{name}  —  Word Cloud")

        # Save top phrases as JSON
        phrase_data = {
            "president": name,
            "party": party,
            "top_unigrams": uni.most_common(50),
            "top_bigrams":  bi.most_common(50),
            "top_trigrams": tri.most_common(30),
        }
        (pout / "phrases.json").write_text(json.dumps(phrase_data, indent=2))
        print(f"  ✓ {name}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHRASE HEATMAP  —  top bigrams across all presidents
    # ─────────────────────────────────────────────────────────────────────────
    print("\nBuilding cross-president bigram heatmap …")
    # collect global top bigrams
    global_bigrams: Counter = Counter()
    for c in bigram_all.values():
        global_bigrams.update(c)
    top_phrases = [p for p, _ in global_bigrams.most_common(TOP_N_PHRASES)]

    heat_data: dict[str, dict[str, int]] = {}
    for slug, bi in bigram_all.items():
        heat_data[display_name(slug)] = {p: bi.get(p, 0) for p in top_phrases}

    heat_df = pd.DataFrame(heat_data, index=top_phrases).T
    plot_phrase_heatmap(heat_df.T, "Top Bigram Frequency  ×  President",
                        OUTPUT_DIR / "presidents_bigram_heatmap.png", "#2C3E50")

    # ─────────────────────────────────────────────────────────────────────────
    # PER-PARTY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🔵🔴  Generating per-party charts …")
    party_out = OUTPUT_DIR / "parties"
    party_out.mkdir(exist_ok=True)

    party_bigrams: dict[str, Counter] = {}
    for party, tokens in party_tokens.items():
        if party == "Unknown":
            continue
        color = PARTY_COLORS.get(party, "#555")
        pout  = party_out / party.lower()
        pout.mkdir(exist_ok=True)

        uni = get_ngrams(tokens, 1)
        bi  = get_ngrams(tokens, 2)
        tri = get_ngrams(tokens, 3)
        party_bigrams[party] = bi

        plot_ngrams(uni, f"{party}  —  Top Unigrams",  color, pout / "unigrams.png")
        plot_ngrams(bi,  f"{party}  —  Top Bigrams",   color, pout / "bigrams.png")
        plot_ngrams(tri, f"{party}  —  Top Trigrams",  color, pout / "trigrams.png")
        make_wordcloud(party_raw[party], color, pout / "wordcloud.png",
                       f"{party}  —  Word Cloud")

        phrase_data = {
            "party": party,
            "top_unigrams": uni.most_common(50),
            "top_bigrams":  bi.most_common(50),
            "top_trigrams": tri.most_common(30),
        }
        (pout / "phrases.json").write_text(json.dumps(phrase_data, indent=2))
        print(f"  ✓ {party}")

    # ── side-by-side party comparison bar chart ───────────────────────────────
    print("\n📊  Building party comparison chart …")
    parties = [p for p in party_bigrams if p in PARTY_COLORS]
    if len(parties) == 2:
        shared_top = [p for p, _ in
                      (party_bigrams[parties[0]] + party_bigrams[parties[1]]).most_common(TOP_N_NGRAMS)]

        x   = np.arange(len(shared_top))
        w   = 0.38
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, party in enumerate(parties):
            vals = [party_bigrams[party].get(ph, 0) for ph in shared_top]
            ax.bar(x + (i - 0.5) * w, vals, w, label=party,
                   color=PARTY_COLORS[party], alpha=0.88, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(shared_top, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Frequency")
        ax.set_title("Republican vs Democrat  —  Top Shared Bigrams", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(party_out / "party_comparison_bigrams.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── party heatmap ─────────────────────────────────────────────────────────
    if len(parties) >= 2:
        global_party_bi: Counter = Counter()
        for c in party_bigrams.values():
            global_party_bi.update(c)
        top_party_phrases = [p for p, _ in global_party_bi.most_common(TOP_N_PHRASES)]
        heat_party = {party: {ph: party_bigrams[party].get(ph, 0) for ph in top_party_phrases}
                      for party in parties}
        heat_party_df = pd.DataFrame(heat_party, index=top_party_phrases)
        plot_phrase_heatmap(heat_party_df, "Top Bigrams  ×  Party",
                            party_out / "party_bigram_heatmap.png", "#2C3E50")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📝  Writing summary …")
    lines = ["# Executive Orders NLP Analysis — Summary\n"]
    for slug, tokens in president_tokens.items():
        party  = PARTY_MAP.get(slug, "Unknown")
        bi     = bigram_all.get(slug, Counter())
        top5   = ", ".join(p for p, _ in bi.most_common(5))
        lines.append(f"## {display_name(slug)}  ({party})")
        lines.append(f"- Total tokens: {len(tokens):,}")
        lines.append(f"- Top bigrams: {top5}\n")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))

    print(f"Results saved to  {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()