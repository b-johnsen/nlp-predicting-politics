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

# download once if needed
# nltk.download("punkt")
# nltk.download("stopwords")


import os
import pandas as pd


OUTPUT_DIR = "nlp-predicting-politics/word analysis/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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
                                    "split": split_name
                                })

                            except Exception as e:
                                print(f"Skipping {path}: {e}")

    return pd.DataFrame(data)

from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text

base_stopwords = set(stopwords.words("english"))

# stopwords
custom_stopwords = {
    "section", "sec", "subsection", "order", "shall", "may", "must",
    "hereby", "thereof", "therein", "whereas", "therefore",
    "president", "executive", "federal", "government", "agency", "agencies",
    "united", "states", "state", "department",
    "act", "law", "provision", "title", "chapter", "paragraph",
    "provide", "provides", "provided", "including", "include",
    "said", "secretary", "ordered", "approved", "director", "stat", "authority", "pursuant",
    "vested", "amended", "public", "usc", "service", "virtue", "thence", "consistent", "provisions"
}

# include pursuant authority vested? frequent

STOPWORDS = base_stopwords.union(custom_stopwords)

def custom_tokenizer(text):

    text = clean_text(text)

    tokens = nltk.word_tokenize(text)

    cleaned_tokens = []
    for tok in tokens:
        if (
            len(tok) > 2 and              # remove tiny tokens
            tok not in STOPWORDS and      # remove stopwords
            tok.isalpha()                 # keep only words
        ):
            cleaned_tokens.append(tok)

    return cleaned_tokens


def get_top_terms(matrix, labels, group_name, feature_names, top_n=20):
    """
    matrix: TF-IDF matrix
    labels: group labels (party or president)
    group_name: specific group to filter
    """
    idx = np.where(labels == group_name)[0]
    
    if len(idx) == 0:
        return []
    
    submatrix = matrix[idx]
    
    # average TF-IDF score across documents
    mean_scores = np.asarray(submatrix.mean(axis=0)).flatten()
    
    top_indices = mean_scores.argsort()[::-1][:top_n]
    
    return list(zip(feature_names[top_indices], mean_scores[top_indices]))


def make_wordcloud(term_list, title, filename):
    freq_dict = {term: score for term, score in term_list}
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(freq_dict)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title(title)
    #plt.show()
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches="tight", dpi = 200)
    plt.close() 

def save_top_terms(terms, name):
    df_terms = pd.DataFrame(terms, columns=["term", "score"])
    df_terms.to_csv(os.path.join(TABLES_DIR, f"{name}_top_terms.csv"), index=False)

def save_barplot(term_list, title, filename):
    terms = [t for t, _ in term_list]
    scores = [s for _, s in term_list]
    
    plt.figure(figsize=(10, 6))
    plt.barh(terms[::-1], scores[::-1])
    plt.title(title)
    
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches="tight", dpi = 200)
    plt.close()


def main():
    df = load_text_dataset([
        Path("nlp-predicting-politics/eo_data/clean_eo_split/train"),
        Path("nlp-predicting-politics/eo_data/clean_eo_split/test")
    ])

    df = df.dropna(subset=["text", "president", "party"])


    df["text"] = df["text"].apply(clean_text)


    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=10000,
        min_df=5
    )

    X = vectorizer.fit_transform(df["text"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    parties = df["party"].values
    unique_parties = df["party"].unique()

    print("\n=== TOP TERMS BY PARTY ===\n")

    party_top_terms = {}

    for party in unique_parties:
        top_terms = get_top_terms(X, parties, party, feature_names, top_n=25)
        party_top_terms[party] = top_terms

        print(f"\n--- {party} ---")
        for term, score in top_terms:
            print(f"{term:20s} {score:.4f}")

        save_top_terms(top_terms, f"party_{party}")

    for party, terms in party_top_terms.items():
        if len(terms) > 0:
            save_barplot(
                terms[:15],
                f"Top Terms - {party}",
                f"barplot_party_{party}.png"
            )


    presidents = df["president"].values
    unique_presidents = df["president"].unique()

    print("\n=== TOP TERMS BY PRESIDENT ===\n")

    pres_top_terms = {}

    for pres in unique_presidents:
        top_terms = get_top_terms(X, presidents, pres, feature_names, top_n=25)
        pres_top_terms[pres] = top_terms

        print(f"\n--- {pres} ---")
        for term, score in top_terms:
            print(f"{term:20s} {score:.4f}")

        save_top_terms(top_terms, f"president_{pres}")

    for pres, terms in pres_top_terms.items():
        if len(terms) > 0:
            save_barplot(
                terms[:15],
                f"Top Terms - {pres}",
                f"barplot_president_{pres}.png"
            )

    for party, terms in party_top_terms.items():
        if len(terms) > 0:
            make_wordcloud(
                terms,
                f"Top Terms - {party}",
                f"wordcloud_party_{party}.png"
            )

    for pres, terms in pres_top_terms.items():
        if len(terms) > 0:
            make_wordcloud(
                terms,
                f"Top Terms - {pres}",
                f"wordcloud_president_{pres}.png"
            )
    



if __name__ == "__main__":
    main()
