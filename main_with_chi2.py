import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from keel import load_dataset, find_datasets
from chi_square import ChiSquare
from scipy.stats import rankdata

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

from optimization import make_validate, optimize


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
scores_chi2 = np.zeros((len(classifiers), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    print(f"Dataset: {dataset}")
    X, y = load_dataset(dataset, return_X_y=True, storage=DATASETS_DIR)
    # normalization - transform data to [0, 1]
    X = MinMaxScaler().fit_transform(X, y)

    # Get feature names
    with open('datasets/%s/%s-header.txt' % (dataset, dataset), 'rt') as file:
        header = file.read()
        a, feature_names = header.split("@inputs ")
        feature_names = feature_names.split("\n")
        del feature_names[-2:]
        feature_names = feature_names[0].split(', ')

    # # K-best feature selection
    # # Two features with highest chi-squared statistics are selected
    # chi2_features = SelectKBest(chi2, k=2)
    # X_kbest_features = chi2_features.fit_transform(X, y)
    # # Reduced features
    # print('Original feature number:', X.shape[1])
    # print('Reduced feature number:', X_kbest_features.shape[1])

    # Original data
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        for clf_id, clf_name in enumerate(classifiers):
            clf = clone(classifiers[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_id, data_id, fold_id] = accuracy_score(y_test, y_pred)

    # Feature Selection with Chi square
    X_df = pd.DataFrame(X)
    X_df.columns = feature_names
    X_df['class'] = y
    y_df = pd.DataFrame(y)
    chi_sq = ChiSquare(X_df)
    for col in feature_names:
        chi_sq.TestIndependence(colX=col, colY='class')
    selected_features = chi_sq.features
    X_df_selected = X_df[selected_features]
    X_selected = X_df_selected.to_numpy()

    for fold_id, (train, test) in enumerate(rskf.split(X_selected, y)):
        X_train, X_test = X_selected[train], X_selected[test]
        y_train, y_test = y[train], y[test]

        for clf_id, clf_name in enumerate(classifiers):
            clf = clone(classifiers[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores_chi2[clf_id, data_id, fold_id] = accuracy_score(y_test, y_pred)

print(classifiers.keys())

np.save('results_new', scores)
scores = np.load('results_new.npy')
# print("\nScores:\n", scores.shape)
mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

np.save('results_chi2', scores_chi2)
scores = np.load('results_chi2.npy')
# print("\nScores Chi2:\n", scores_chi2.shape)
mean_scores_chi2 = np.mean(scores_chi2, axis=2).T
print("\nMean scores Chi2:\n", mean_scores_chi2)


# # Optimization - sw
# # Make split for optimizer
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, stratify=y, test_size=0.5, random_state=0)

# Czym jest L i jakie powinno być w moim przypadku - to chyba wszystkie cechy
# L = feature_names
# print(L)
# n_max = 200

# # Single objective
# for metric in METRICS:
#     clf = clone(clf_)
#     validate = make_validate(clf, X_train, X_test, y_train, y_test)
#
#     fig, axs = plt.subplots(2, 1)
#     fig.suptitle(f'{metric}', fontsize=15)
#
#     res = optimize(L, n_max, validate, [metric])
#     solution = L[res.X]
#
#     m = validate(solution)
#     axs[0].bar(list(m.keys()), m.values())
#     axs[0].set_ylim(0.0, 1.0)
#     for k in m:
#         axs[0].annotate(f"{m[k]:.2f}", xy=(k, m[k]), ha='center', va='bottom')
#
#     s_map = np.ones(len(y_train), dtype=bool)
#     s_map[solution] = False
#     opt_X, opt_y = X_train[s_map], y_train[s_map]
#
#     xx, yy = np.meshgrid(np.arange(-2.5, 3.5, h), np.arange(-2, 2.5, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     axs[1].set_xlim(xx.min(), xx.max())
#     axs[1].set_ylim(yy.min(), yy.max())
#     Z = Z.reshape(xx.shape)
#     plt.pcolormesh(xx, yy, Z, alpha=0.3, shading='auto')
#     axs[1].scatter(*opt_X.T, c=opt_y)
#
#     plt.tight_layout()
#     plt.savefig(f'img/moon_opt_{metric}.png')
#     plt.clf()

# scores_single = np.zeros((len(classifiers), n_datasets, n_splits * n_repeats))
# objectives = 1
# h = 0.05
#
# for data_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
#     print(f"Dataset: {dataset}")
#     X, y = load_dataset(dataset, return_X_y=True, storage=DATASETS_DIR)
#     # normalization - transform data to [0, 1]
#     X = MinMaxScaler().fit_transform(X, y)
#
#     # Get feature names
#     with open('datasets/%s/%s-header.txt' % (dataset, dataset), 'rt') as file:
#         header = file.read()
#         a, feature_names = header.split("@inputs ")
#         feature_names = feature_names.split("\n")
#         del feature_names[-2:]
#         feature_names = feature_names[0].split(', ')
#
#     L = feature_names
#     print(L)
#
#     # Single objective
#     for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#         X_train, X_test = X[train], X[test]
#         y_train, y_test = y[train], y[test]
#
#         for clf_id, clf_name in enumerate(classifiers):
#             clf = clone(classifiers[clf_name])
#             validate = make_validate(clf, X_train, X_test, y_train, y_test)
#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_test)
#             scores_single[clf_id, data_id, fold_id] = accuracy_score(y_test, y_pred)
#
#     res = optimize(L, n_max, validate, objectives)
#     print("Function value: %s" % res.F[0])
#     print("Subset:", np.where(res.X)[0])
    # solution = L[res.X]
    #
    # m = validate(solution)
    # s_map = np.ones(len(y_train), dtype=bool)
    # s_map[solution] = False
    # opt_X, opt_y = X_train[s_map], y_train[s_map]
    #
    # fig, axs = plt.subplots(2, 1)
    # xx, yy = np.meshgrid(np.arange(-2.5, 3.5, h), np.arange(-2, 2.5, h))
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # axs[1].set_xlim(xx.min(), xx.max())
    # axs[1].set_ylim(yy.min(), yy.max())
    # Z = Z.reshape(xx.shape)
    # plt.pcolormesh(xx, yy, Z, alpha=0.3, shading='auto')
    # axs[1].scatter(*opt_X.T, c=opt_y)
    #
    # plt.tight_layout()
    # plt.savefig(f'img/moon_opt_{dataset}.png')
    # plt.clf()
