import numpy as np
import matplotlib.pyplot as plt
import gensim
import re
import nltk
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer 

def load_text_dataset(folder_0, folder_1):
    data = []

    def process_folder(root_folder, label):
        for root, _, files in os.walk(root_folder):
            for fname in files:
                if fname.endswith(".txt"):
                    path = os.path.join(root, fname)
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                        data.append({
                            "text": text,
                            "label": label,
                            # "filepath": path # using to keep track of file?
                        })
                    except Exception as e:
                        print(f"Skipping {path}: {e}")

    # Process both folders
    process_folder(folder_0, label=0)
    process_folder(folder_1, label=1)

    df = pd.DataFrame(data)

    return df

from nltk.corpus import wordnet

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def normalize_for_bow(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words("english"))

    def preprocess(text):
        # lowercase
        text = text.lower()

        # remove URLs
        text = re.sub(r"http\S+", "", text)

        # remove digits, punctuation
        text = re.sub(r"[^(a-z)\s]", " ", text)

        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # remove stopwords + lemmatize
        tokens = nltk.word_tokenize(text)

        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words and len(word) > 2
        ]

        return " ".join(tokens)

    df = df.copy()
    df["text"] = df["text"].apply(preprocess)

    return df

def build_bow_features(train_texts, test_texts, max_features=5000, ngram_range=(1,3)):
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=None
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer

def score_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    # Apply model to test set using predict()
    y_pred = model.predict(X_test)

    print(metrics.classification_report(y_test, y_pred))

    results = {
        "model": model_name if model_name else type(model).__name__,
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "f1": metrics.f1_score(y_test, y_pred),
    }

    return results

def run_bow_experiment(folder, normalize = True):
    train_path = Path("nlp-predicting-politics/data/eo_data/clean_eo_split/train")
    test_path = Path("nlp-predicting-politics/data/eo_data/clean_eo_split/test")

    train_df = load_text_dataset(
        train_path / "democrat",
        train_path / "republican"
    )

    test_df = load_text_dataset(
        test_path / "democrat",
        test_path / "republican"
    )

    if normalize:
        train_df = normalize_for_bow(train_df)
        test_df = normalize_for_bow(test_df)

    X_train, X_test, vectorizer = build_bow_features(
        train_df["text"],
        test_df["text"]
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    results = []

    models = [
        (GaussianNB(), "GaussianNB"),  # needs dense
        (LogisticRegression(max_iter=1000), "LogReg"),
        (LinearSVC(max_iter=1000), "LinearSVC"),
        (DecisionTreeClassifier(), "DecisionTree"),
    ]

    bow_type = ''
    if normalize:
        bow_type = 'norm_bow'
    else:
        bow_type = 'unnorm_bow'

    for model, name in models:

        if name == "GaussianNB":
            X_tr = X_train.toarray()
            X_te = X_test.toarray()
        else:
            X_tr = X_train
            X_te = X_test

        res = score_model(model, f"{bow_type}_{name}", X_tr, y_train, X_te, y_test)
        results.append(res)

    return pd.DataFrame(results)

def main():
    folder = Path("nlp-predicting-politics/statistical modeling/bow models/bow results")

    bow_not_normalized_results = run_bow_experiment(folder, normalize=False)
    bow_normalized_results = run_bow_experiment(folder, normalize=True)

    all_bow_results = pd.concat([bow_not_normalized_results, bow_normalized_results], ignore_index=True)

    all_bow_results.to_csv(folder / "modern_bow_comparison_results.csv", index=False)

if __name__ == "__main__":
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger_eng')
  main()