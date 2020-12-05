import os
import numpy as np
import csv

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import chi2

from keel import find_datasets
from methods.fsclf import FeatueSelectionClf
from methods.gaaccclf import GeneticAlgorithmAccuracyClf
from methods.gaacccost import GAAccCost
from methods.nsgaacccost import NSGAAccCost

DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

base_classifiers = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=10),
}

test_size = 0.2
scale_features = 0.7
methods = {}
for key, base in base_classifiers.items():
    methods['FS-{}'.format(key)] = FeatueSelectionClf(base, chi2, scale_features)
    methods['GAacc-{}'.format(key)] = GeneticAlgorithmAccuracyClf(base, scale_features, test_size)
    methods['GAaccCost-{}'.format(key)] = GAAccCost(base, scale_features, test_size)
    methods['NSGAaccCost-{}'.format(key)] = NSGAAccCost(base, scale_features, test_size)

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

# for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
#     print("\n",dataset)
#     for clf_id, clf_name in enumerate(methods):
#         print(clf_name, "%.2f" % mean_scores[dataset_id, clf_id], u"\u00B1", "%.2f" % stds[dataset_id, clf_id])

# Save scores in nice csv tables
column_names = ["Dataset", "FS", "GA Acc", "GA Acc Cost", "NSGA"]
for key, base in base_classifiers.items():
    contents = [[None for _ in range(4+1)] for _ in range(n_datasets)]

    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        contents[dataset_id][0] = dataset
        i = 0
        for clf_id, clf_name in enumerate(methods):
            if key in clf_name:
                ms = mean_scores[dataset_id, clf_id].tolist()
                std = stds[dataset_id, clf_id].tolist()
                string = str(float("{:.2f}".format(ms))) + " +- " + str(float("{:.2f}".format(std)))
                # contents[dataset_id][i+1] = float("{:.2f}".format(ms)), float("{:.2f}".format(std))
                contents[dataset_id][i+1] = string
                # contents.insert((dataset_id, i+1), float("{:.2f}".format(ms)))
                i += 1
    with open("results/experiment1/tables/Accuracy_%s.csv" % (key), 'w') as f:
        write = csv.writer(f)
        write.writerow(column_names)
        write.writerows(contents)
