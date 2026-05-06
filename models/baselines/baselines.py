import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from pathlib import Path
import os

# one of the folds is missing one of the labels
import warnings
warnings.filterwarnings('ignore')

def run_baselines(X, y):
    # These are the scoring metrics we will consider
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Here we are looping throught our three favorite naive baselines
    # * most_frequent is majority class
    # * stratified is selecting randomly at the probability distribution in the dataset
    # * uniform is selecting, for a binary class problem like this, 50/50

    score_dict = {}

    for s in ["most_frequent", "stratified", "uniform"]:

        # initialize the classifier
        dummy_classifier = DummyClassifier(strategy=s,random_state=42)

        # run the classifier for each fold, where the number of folds is 5
        scores = cross_validate(dummy_classifier, X, y, cv=5, scoring=scoring_metrics)

        score_dict[s] = scores

    return score_dict

def count_files(folder_path):
    count = 0
    for entry in os.scandir(folder_path):
        if entry.is_file():
            count += 1
    return count

def main():

    rep_subdirs = []
    for entry in os.scandir("nlp-predicting-politics/data/eo_data/eo_labeled_split/test/republican"):
        if entry.is_dir():
            rep_subdirs.append(entry.path)
    for entry in os.scandir("nlp-predicting-politics/data/eo_data/eo_labeled_split/train/republican"):
        if entry.is_dir():
            rep_subdirs.append(entry.path)


    dem_subdirs = []
    for entry in os.scandir("nlp-predicting-politics/data/eo_data/eo_labeled_split/test/democrat"):
        if entry.is_dir():
            dem_subdirs.append(entry.path)
    for entry in os.scandir("nlp-predicting-politics/data/eo_data/eo_labeled_split/train/democrat"):
        if entry.is_dir():
            dem_subdirs.append(entry.path)

    y = []

    # Count Republican files (1)
    for subdir in rep_subdirs:
        num_files = count_files(subdir)
        y.extend([1] * num_files)

    # Count Democrat files (0)
    for subdir in dem_subdirs:
        num_files = count_files(subdir)
        y.extend([0] * num_files)

    y = np.array(y)

    # Dummy feature matrix 
    X = np.zeros((len(y), 1))

    # Run baselines
    results = run_baselines(X, y)

    for strategy, scores in results.items():
        print(f"\nBasline: {strategy}")
        for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
            print(f"{metric}: {scores[metric].mean():.4f}")

    
if __name__ == "__main__":
    main()



