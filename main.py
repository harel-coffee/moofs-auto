import os
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from keel import load_dataset, find_datasets
from chi_square import ChiSquare
from scipy.stats import rankdata

DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')

classifiers = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=10),
}

# n_datasets = len(datasets)
n_datasets = 3
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(classifiers), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    print(f"Dataset: {dataset}")
    X, y = load_dataset(dataset, return_X_y=True, storage=DATASETS_DIR)
    # X = StandardScaler().fit_transform(X, y)
    # normalization - transform data to [0, 1]
    X = MinMaxScaler().fit_transform(X, y)

    # Get feature names
    with open('datasets/%s/%s-header.txt' % (dataset, dataset), 'rt') as file:
        header = file.read()
        a, feature_names = header.split("@inputs ")
        feature_names = feature_names.split("\n")
        del feature_names[-2:]
        feature_names = feature_names[0].split(', ')

    # # Two features with highest chi-squared statistics are selected
    # chi2_features = SelectKBest(chi2, k=2)
    # X_kbest_features = chi2_features.fit_transform(X, y)
    # # Reduced features
    # print('Original feature number:', X.shape[1])
    # print('Reduced feature number:', X_kbest_features.shape[1])

    # Feature Selection with Chi square
    X_df = pd.DataFrame(X)
    X_df.columns = feature_names
    X_df['class'] = y
    y_df = pd.DataFrame(y)
    chi_sq = ChiSquare(X_df)
    for col in feature_names:
        chi_sq.TestIndependence(colX=col, colY='class')
    f_df = chi_sq.select_features(feature_names)
    print("Selected features:")
    print(f_df)
    # Jak wyjąć i stworzyć zestaw cech, na którym potem bdmy trenować clfs?

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for clf_id, clf_name in enumerate(classifiers):

            clf = clone(classifiers[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_id, data_id, fold_id] = accuracy_score(y_test, y_pred)



np.save('results_new', scores)
scores = np.load('results_new.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

# ranks = []
# for ms in mean_scores:
#     ranks.append(rankdata(ms).tolist())
# ranks = np.array(ranks)
# print("\nRanks:\n", ranks)
# mean_ranks = np.mean(ranks, axis=0)
# print("\nMean ranks:\n", mean_ranks)
