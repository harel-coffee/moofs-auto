import os
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

from keel import load_dataset, find_datasets
from methods.mooclf import MooClf
from methods.fsclf import FSClf

DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')

base_classifiers = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=10),
}

methods = {}
scale_features = 0.7
for key, base in base_classifiers.items():
    methods['MOO-{}'.format(key)] = MooClf(base, scale_features, objectives=1, test_size=0.2)
    methods['SF-{}'.format(key)] = FSClf(base, chi2, scale_features)

n_datasets = 5
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    scores = np.zeros((len(methods), n_splits * n_repeats))
    print(f"Dataset: {dataset}")
    X, y = load_dataset(dataset, return_X_y=True, storage=DATASETS_DIR)
    # Normalization - transform data to [0, 1]
    X = MinMaxScaler().fit_transform(X, y)

    # Get feature names
    with open('datasets/%s/%s-header.txt' % (dataset, dataset), 'rt') as file:
        header = file.read()
        a, feature_names = header.split("@inputs ")
        feature_names = feature_names.split("\n")
        del feature_names[-2:]
        feature_names = feature_names[0].split(', ')
    print(feature_names)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for clf_id, clf_name in enumerate(methods):
            print(clf_name)
            clf = clone(methods[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_id, fold_id] = accuracy_score(y_test, y_pred)

    for clf_id, clf_name in enumerate(methods):
        filename = "results/experiment1/%s/%s.csv" % (dataset, clf_name)
        if not os.path.exists("results/experiment1/%s/" % (dataset)):
            os.makedirs("results/experiment1/%s/" % (dataset))
        np.savetxt(fname=filename, fmt="%f", X=scores[clf_id, :])
