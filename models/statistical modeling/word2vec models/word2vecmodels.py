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
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd


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

# make sure you have nltk.download('punkt')
stop_words = set(stopwords.words("english"))

def normalize_for_w2v(text):
    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    

    # tokenize
    tokens = nltk.word_tokenize(text)

    tokens = [
        t for t in tokens
        if t.isalpha() and len(t) > 1
    ]

    return tokens

def train_word2vec_from_df(df, vector_size=300, window=10, min_count=2):
    toksents = []

    for text in df["text"]:
        tokens = normalize_for_w2v(text)
        if tokens:
            toksents.append(tokens)

    model = Word2Vec(
        sentences=toksents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1  
    )

    print("Word2Vec model built!")
    return model

def get_vector(model, token):
    if hasattr(model, "wv"):  # Word2Vec
        return model.wv[token] if token in model.wv else None
    else:  # KeyedVectors
        return model[token] if token in model else None
    
def document_to_vector(text, model):
    tokens = normalize_for_w2v(text)

    vectors = []
    for t in tokens:
        vec = get_vector(model, t)
        if vec is not None:
            vectors.append(vec)

    if not vectors:
        return np.zeros(model.vector_size)

    vecs = np.array(vectors)
    doc_vec = np.mean(vecs, axis=0)

    # normalize the final document vector
    norm = np.linalg.norm(doc_vec)
    if norm > 1e-9:
        doc_vec = doc_vec / norm

    return doc_vec



def add_embedding_features(df, model, tfidf_vectorizer=None):
    embeddings = []

    for text in df["text"]:
        if tfidf_vectorizer is not None:
            vec = document_to_vector_weighted(text, model, tfidf_vectorizer)
        else:
            vec = document_to_vector(text, model)

        embeddings.append(vec)

    X = np.vstack(embeddings)

    emb_df = pd.DataFrame(
        X,
        columns=[f"emb_{i}" for i in range(X.shape[1])]
    )

    df = df.reset_index(drop=True)
    df = pd.concat([df, emb_df], axis=1)

    return df, X


def document_to_vector_weighted(text, model, tfidf_vectorizer):
    tokens = normalize_for_w2v(text)

    tfidf_vec = tfidf_vectorizer.transform([" ".join(tokens)])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    weights = dict(zip(feature_names, tfidf_vec.toarray()[0]))

    vectors = []
    wts = []

    for token in tokens:
        vec = get_vector(model, token)   # <-- FIX IS HERE
        if vec is not None and token in weights:
            vectors.append(vec)
            wts.append(weights[token])

    if not vectors:
        return np.zeros(model.vector_size)

    vectors = np.array(vectors)
    wts = np.array(wts).reshape(-1, 1)

    return np.sum(vectors * wts, axis=0) / (np.sum(wts) + 1e-9)


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

def load_small_model(df, save_path):
    if save_path.exists():
        return Word2Vec.load(str(save_path))

    model = train_word2vec_from_df(df)
    model.save(str(save_path))
    return model

def load_big_model():
    print("Loading GoogleNews embeddings...")
    return gensim.models.KeyedVectors.load_word2vec_format(
        "nlp-predicting-politics/statistical modeling/word2vec models/models/GoogleNews-vectors-negative300-SLIM.bin.gz",
        binary=True
    )

def run_experiment(folder, embedding_model, model_name="model", tfidf_vectorizer=None):
    # currently using modern
    train_path = Path("nlp-predicting-politics/eo_data/clean_modern_eo_split/train")
    test_path = Path("nlp-predicting-politics/eo_data/clean_modern_eo_split/test")

    train_df = load_text_dataset(
        train_path / "democrat",
        train_path / "republican"
    )

    test_df = load_text_dataset(
        test_path / "democrat",
        test_path / "republican"
    )

    y_train = train_df["label"]


    y_test = test_df["label"]

    train_df, X_train = add_embedding_features(train_df, embedding_model, tfidf_vectorizer)
    test_df, X_test = add_embedding_features(test_df, embedding_model, tfidf_vectorizer)

    results = []

    models = [
        (GaussianNB(), "GaussianNB"),
        (LogisticRegression(max_iter=1000), "LogReg"),
        (LinearSVC(), "LinearSVC"),
        (DecisionTreeClassifier(), "DecisionTree")
    ]

    for model, name in models:
        res = score_model(model, f"{model_name}_{name}", X_train, y_train, X_test, y_test)
        results.append(res)

    return pd.DataFrame(results)


def main():
    folder = Path("nlp-predicting-politics/statistical modeling/word2vec models/word2vec results")

    train_path = Path("nlp-predicting-politics/eo_data/clean_modern_eo_split/train")
    train_df = load_text_dataset(
        train_path / "democrat",
        train_path / "republican"
    )

    # small word2vec
    small_model_path = folder / "modern_word2vec_small.model"
    small_model = load_small_model(train_df, small_model_path)

    big_model = load_big_model()

    # small contextual model
    print("\nSmall model experiment:")
    small_results = run_experiment(folder, small_model, "small")

    # big general model
    print("\nBig model experiment:")
    big_results = run_experiment(folder, big_model, "big")


    # combine
    all_results = pd.concat(
        [small_results, big_results],
        ignore_index=True
    )

    print("\n===== COMBINED RESULTS =====")
    print(all_results)

    all_results.to_csv(folder / "modern_embedding_comparison_results.csv", index=False)

if __name__ == "__main__":
  nltk.download('punkt')
  nltk.download('punkt_tab')

  main()
