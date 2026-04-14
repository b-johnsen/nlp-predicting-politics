import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
    # democrat folders
    process_folder(folder_0, label=0)
    # republican folders
    process_folder(folder_1, label=1)

    df = pd.DataFrame(data)

    return df

def main():
  folder = Path("nlp-predicting-politics/scripts/statistical modeling scripts/word2veceomodel")
  os.makedirs(folder, exist_ok=True)

  train_path = Path("nlp-predicting-politics/clean_data/clean_eo_split/train")

  # First path is democrat, second is for republican
  train_df = load_text_dataset(
      train_path / "democrat",
      train_path / "republican"
  )

  # run tokenization, etc on text column of dataframe and adjust as necessary
  # see word2vecmodels for example

  # labels
  y_train = train_df["label"]

  # saving dataframe
  train_df.to_pickle(folder/"train_dataset.pkl")

  test_path = Path("nlp-predicting-politics/clean_data/clean_eo_split/test")

  test_df = load_text_dataset(
      test_path / "democrat",
      test_path / "republican"
  )

  # labels
  y_test = test_df["label"]

  # saving dataframe as compressed file for storage
  test_df.to_pickle(folder/"test_dataset.pkl")


if __name__ == "__main__":
  main()