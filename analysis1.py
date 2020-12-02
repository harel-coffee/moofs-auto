import os
import numpy as np
import texttable as tt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import chi2

from keel import find_datasets
from methods.mooclf import MooClf
from methods.fsclf import FSClf
from methods.gaacccost import GAAccCost

DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

base_classifiers = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=10),
}

objectives = 1
test_size = 0.2
scale_features = 0.7
methods = {}
for key, base in base_classifiers.items():
    methods['MOO-{}'.format(key)] = MooClf(base, scale_features, objectives, test_size)
    methods['SF-{}'.format(key)] = FSClf(base, chi2, scale_features)
    methods['GAAccCost-{}'.format(key)] = GAAccCost(base, scale_features, objectives, test_size)

mean_scores = np.zeros((n_datasets, len(methods)))
stds = np.zeros((n_datasets, len(methods)))

for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    for clf_id, clf_name in enumerate(methods):
        try:
            # Load data from file
            filename = "results/experiment1/%s/%s.csv" % (dataset, clf_name)
            scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
            mean_score = np.mean(scores)
            mean_scores[dataset_id, clf_id] = mean_score
            std = np.std(scores)
            stds[dataset_id, clf_id] = std
        except IOError:
            print("File", filename, "not found")


print(mean_scores)

for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    print("\n",dataset)
    for clf_id, clf_name in enumerate(methods):
        print(clf_name, "%.2f" % mean_scores[dataset_id, clf_id], u"\u00B1", "%.2f" % stds[dataset_id, clf_id])
