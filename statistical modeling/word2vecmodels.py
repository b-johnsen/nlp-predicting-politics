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
import os
import pandas as pd

# currently this version uses on the smallest word2vec model
# no stop words yet
# test with/without stop words

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

def train_word2vec_from_df(df, vector_size=100, window=5, min_count=3):
    # this is using the small model, but we can also perform analysis using the big model
    # bigmodel = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True)
    # need to implement later

    toksents = []

    # do we want stop words?
    for text in df["text"]:
        text.replace("\n", " ")
        # sentence tokenize
        sentences = nltk.sent_tokenize(text)

        # word tokenize each sentence
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            toksents.append(tokens)

    # train model
    model = Word2Vec(
        toksents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )

    print("Word2Vec model built!")
    return model

import numpy as np

def document_to_vector(text, model):
    tokens = nltk.word_tokenize(text)

    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if len(vectors) == 0:
        # edge case: no known words
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def add_embedding_features(df, model):
    embeddings = []

    for text in df["text"]:
        vec = document_to_vector(text, model)
        embeddings.append(vec)

    X = np.vstack(embeddings)

    # create all embedding columns at once
    emb_df = pd.DataFrame(
        X,
        columns=[f"emb_{i}" for i in range(X.shape[1])]
    )

    # reset index to ensure clean concat alignment
    df = df.reset_index(drop=True)

    # concat features
    df = pd.concat([df, emb_df], axis=1)

    return df, X


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
    

def main():
  folder = Path("/Users/jacktimmermans/Documents/BC Spring 2026/NLP/nlp-predicting-politics/scripts/statistical modeling scripts/word2veceomodel")
  os.makedirs(folder, exist_ok=True)

  train_path = Path("nlp-predicting-politics/clean_data/clean_eo_split/train")

  train_df = load_text_dataset(
      train_path / "democrat",
      train_path / "republican"
  )

  # train word2vec model
  # built only on training dataset for now
  word2vec_path = Path("nlp-predicting-politics/scripts/statistical modeling scripts/word2veceomodel/word2vec.model")

  word2vec_model = np.nan
  # if exists load
  if word2vec_path.exists():
      print(f"Loading existing Word2Vec model from {word2vec_path}")
      word2vec_model = Word2Vec.load(str(word2vec_path))
  else:
    word2vec_model = train_word2vec_from_df(train_df)
    word2vec_model.save(str(folder / "word2vec.model"))

  # create features and make dataframe
  train_df, X_train = add_embedding_features(train_df, word2vec_model)

  # labels
  y_train = train_df["label"]

  train_df.to_pickle(folder/"train_dataset.pkl")

  test_path = Path("nlp-predicting-politics/clean_data/clean_eo_split/test")

  test_df = load_text_dataset(
      test_path / "democrat",
      test_path / "republican"
  )

  # create features and make dataframe
  test_df, X_test = add_embedding_features(test_df, word2vec_model)

  # labels
  y_test = test_df["label"]

  test_df.to_pickle(folder/"test_dataset.pkl")

  results = []

  models = [
    (GaussianNB(), "GaussianNB"),
    (LogisticRegression(max_iter=1000), "LogReg"),
    (LinearSVC(), "LinearSVC"),
  ]

  for i in models:
    print(i)
    model, name = i[0], i[1]
    res = score_model(model, name, X_train, y_train, X_test, y_test)
    results.append(res)

  results_df = pd.DataFrame(results)

  results_df.to_csv(str(folder / "model_results.csv"), index=False)


if __name__ == "__main__":
  nltk.download('punkt')
  nltk.download('punkt_tab')

  main()