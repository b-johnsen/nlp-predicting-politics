import re
import pandas as pd


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from collections import defaultdict
from sklearn.feature_extraction import text

from collections import Counter

# download once if needed
# nltk.download("punkt")
# nltk.download("stopwords")

############################################
# 1. LOAD DATA
############################################

import os
import pandas as pd
from dateutil import parser


OUTPUT_DIR = "nlp-predicting-politics/word analysis/results_over_time"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def extract_date_from_filename(path):
    """
    Extracts a human-readable date like:
    'April 13, 1866__executive-order.txt'
    """

    fname = os.path.basename(path)

    # remove file extension and everything after "__"
    cleaned = fname.split("__")[0]

    try:
        return parser.parse(cleaned)
    except Exception:
        return pd.NaT

custom_stopwords = {
    "section", "sec", "subsection", "order", "shall", "may", "must",
    "hereby", "thereof", "therein", "whereas", "therefore",
    "president", "executive", "federal", "government", "agency", "agencies",
    "united", "states", "state", "department",
    "act", "law", "provision", "title", "chapter", "paragraph",
    "provide", "provides", "provided", "including", "include",
    "said", "secretary", "ordered", "approved", "director", "stat", "authority", "pursuant",
    "vested", "amended"
}

# include pursuant authority vested? frequent
from nltk.corpus import stopwords

base_stopwords = set(stopwords.words("english"))

STOPWORDS = base_stopwords.union(custom_stopwords)

def custom_tokenizer(text):
    """
    Tokenizer optimized for executive orders / policy text.
    """

    # 1. Normalize text
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d+", " ", text)  # remove numbers
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation

    # 2. Tokenize
    tokens = nltk.word_tokenize(text)

    # 3. Filter tokens
    cleaned_tokens = []
    for tok in tokens:
        if (
            len(tok) > 2 and              # remove tiny tokens
            tok not in STOPWORDS and      # remove stopwords
            tok.isalpha()                 # keep only words
        ):
            cleaned_tokens.append(tok)

    return cleaned_tokens

def load_text_dataset(data_dirs):
    """
    data_dirs: list of root directories (e.g., ["train", "test"])
    
    Returns:
        DataFrame with columns:
        - text
        - label (0/1)
        - party
        - president
        - filepath
        - split (train/test)
    """
    
    data = []

    # map party to label
    party_to_label = {
        "democrat": 0,
        "republican": 1
    }

    for split_dir in data_dirs:
        split_name = os.path.basename(split_dir)  # "train" or "test"

        for party in os.listdir(split_dir):
            party_path = os.path.join(split_dir, party)

            if not os.path.isdir(party_path):
                continue

            party_lower = party.lower()
            label = party_to_label.get(party_lower, None)

            for president in os.listdir(party_path):
                pres_path = os.path.join(party_path, president)

                if not os.path.isdir(pres_path):
                    continue

                for root, _, files in os.walk(pres_path):
                    for fname in files:
                        if fname.endswith(".txt"):
                            path = os.path.join(root, fname)

                            try:
                                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                    text = f.read()

                                data.append({
                                    "text": text,
                                    "label": label,
                                    "party": party_lower,
                                    "president": president.lower(),
                                    "filepath": path,
                                    "date": extract_date_from_filename(path)
                                })

                            except Exception as e:
                                print(f"Skipping {path}: {e}")

    return pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text

def word_over_time(word, freq_df):
    if word in freq_df.index:
        plt.plot(freq_df.columns, freq_df.loc[word])
        plt.title(f"Frequency Over Time: {word}")
        plt.xlabel("Year")
        plt.ylabel("Relative Frequency")
        plt.show()

def main():
    df = load_text_dataset([
        Path("nlp-predicting-politics/eo_data/clean_eo_split/train"),
        Path("nlp-predicting-politics/eo_data/clean_eo_split/test")
    ])

    df = df.dropna(subset=["text", "president", "party"])

    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M")

    df["text"] = df["text"].apply(clean_text)

    year_groups = df.groupby("year")["text"].apply(lambda x: " ".join(x))

    year_tokens = {
        year: custom_tokenizer(text)
        for year, text in year_groups.items()
    }


    year_counts = {
        year: Counter(tokens)
        for year, tokens in year_tokens.items()
    }

    freq_df = pd.DataFrame(year_counts).fillna(0)
    freq_df = freq_df / freq_df.sum(axis=0)  # normalize
    freq_df = freq_df.sort_index(axis=1)
    freq_df.to_csv(os.path.join(OUTPUT_DIR, "word_freq_over_time.csv"))

    word_over_time("immigration", freq_df)
    word_over_time("climate", freq_df)
    word_over_time("security", freq_df)

    for party in df["party"].unique():
        sub = df[df["party"] == party]

        year_groups = sub.groupby("year")["text"].apply(lambda x: " ".join(x))

        print(f"{party}: {len(year_groups)} years")


if __name__ == "__main__":
    main()

