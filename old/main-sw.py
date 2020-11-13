from methods import GCC

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

from metrics import confusion_matrix_scores
from sklearn.metrics import accuracy_score

import os
import numpy as np
from keel import load_dataset, find_datasets
import pandas as pd


DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')

ESTIMATORS = (
    ("GNB", GaussianNB()),
    ("DT", DecisionTreeClassifier())
)

RESULTS = []

for dataset_name in find_datasets(DATASETS_DIR):
    print(f"Dataset: {dataset_name}")
    X, y = load_dataset(dataset_name, return_X_y=True, storage=DATASETS_DIR)


    X = StandardScaler().fit_transform(X, y)

    folding = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=0)

    # fig, axs = plt.subplots(1, 10, figsize=(20,5))
    # axs = iter(axs)

    for fold_idx, (train_index, test_index) in enumerate(folding.split(X, y), 1):
        print(f"  Fold: {fold_idx}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for e_name, estimator in ESTIMATORS:
            clf = clone(estimator)
            print(f"\tClassifier: {e_name} ...")

            clf.fit(X_train, y_train)
            cm = confusion_matrix(y_test, clf.predict(X_test))
            RESULTS.append({
                "Dataset": dataset_name,
                "Classifier": e_name,
                "Fold": fold_idx,
                **confusion_matrix_scores(cm)
            })

            # ax = next(axs)
            #
            # ax.imshow(clf.image, vmin=0.0, vmax=1.0)
            # ax.set_title("%.3f" % score, fontsize=13)

# Store results to unprocessed csv
df = pd.DataFrame(RESULTS)
df.to_csv("results.csv")
print(df)
