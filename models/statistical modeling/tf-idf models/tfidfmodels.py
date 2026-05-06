import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk


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

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

custom_stopwords = {
    "section", "sec", "subsection", "order", "shall", "may", "must",
    "hereby", "thereof", "therein", "whereas", "therefore",
    "president", "executive", "federal", "government", "agency", "agencies",
    "united", "states", "state", "department",
    "act", "law", "provision", "title", "chapter", "paragraph",
    "provide", "provides", "provided", "including", "include",
    "said", "secretary", "ordered", "approved", "director", "stat", "authority", "pursuant",
    "vested", "amended", "public", "usc", "service", "virtue", "thence", "consistent", "provisions",
    "policy", "within"
}

# include pursuant authority vested? frequent

STOPWORDS = stop_words.union(custom_stopwords)

def tfidf_tokenizer(text):
    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # keep letters only
    text = re.sub(r"[^a-z\s]", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # tokenize
    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOPWORDS
    ]

    return tokens


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


def build_tfidf_features(train_texts, test_texts, max_features=5000, ngram_range=(1,2), normalize = True):
    vectorizer = TfidfVectorizer(
        tokenizer=tfidf_tokenizer,
        max_features=10000,
        ngram_range=(1,3),
        sublinear_tf=True,
        min_df=2,
        max_df=0.9,
        norm='l2'
    )
    if not normalize:
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            norm='l2'
        )


    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer

def run_tfidf_experiment(folder, normalize = True):

    # currently not on modern
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

    X_train, X_test, vectorizer = build_tfidf_features(
        train_df["text"],
        test_df["text"],
        normalize=normalize
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    results = []

    models = [
        (GaussianNB(), "GaussianNB"),  # needs dense input, consider something else?
        (LogisticRegression(max_iter=1000), "LogReg"),
        (LinearSVC(), "LinearSVC"),
        (DecisionTreeClassifier(), "DecisionTree")
    ]

    for model, name in models:
        # GaussianNB requires dense matrix
        if name == "GaussianNB":
            X_tr = X_train.toarray()
            X_te = X_test.toarray()
        else:
            X_tr = X_train
            X_te = X_test

        res = score_model(model, f"tfidf_{name}", X_tr, y_train, X_te, y_test)
        results.append(res)

    return pd.DataFrame(results)
    

def main():
    folder = Path("nlp-predicting-politics/statistical modeling/tf-idf models")


    # tf-idf experiments
    not_norm_tfidf_results = run_tfidf_experiment(folder, normalize = False)
    norm_tfidf_results = run_tfidf_experiment(folder, normalize = True)

    all_results = pd.concat([
        not_norm_tfidf_results.assign(method="notnorm_tfidf"),
        norm_tfidf_results.assign(method="norm_tfidf")
    ])

    print(all_results)
    all_results.to_csv(folder / "tf-idf results" / "tfidfcomparison.csv", index=False)

if __name__ == "__main__":

  main()