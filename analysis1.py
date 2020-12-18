import os
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import chi2

from utils import find_datasets
from methods.fsclf import FeatueSelectionClf
from methods.gaaccclf import GeneticAlgorithmAccuracyClf
from methods.gaacccost import GAAccCost
from methods.nsgaacccost import NSGAAccCost

DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

base_classifiers = {
    'GNB': GaussianNB(),
    # 'SVM': SVC(),
    # 'kNN': KNeighborsClassifier(),
    # 'CART': DecisionTreeClassifier(random_state=10),
}

test_size = 0.2
scale_features = 0.7
methods = {}
for key, base in base_classifiers.items():
    methods['FS_{}'.format(key)] = FeatueSelectionClf(base, chi2, scale_features)
    methods['GAacc_{}'.format(key)] = GeneticAlgorithmAccuracyClf(base, scale_features, test_size)
    methods['GAaccCost_{}'.format(key)] = GAAccCost(base, scale_features, test_size)

    methods['NSGAaccCost_acc_{}'.format(key)] = NSGAAccCost(base, scale_features, test_size)
    methods['NSGAaccCost_cost_{}'.format(key)] = NSGAAccCost(base, scale_features, test_size)
    # methods['NSGAaccCost_promethee_{}'.format(key)] = NSGAAccCost(base, scale_features, test_size)

methods_names = [name for name in methods.keys()]
opt_methods = [
                # "FS",
                # "GA_a",
                "GA_ac",
                "NSGA_a",
                # "NSGA_c",
                # "NSGA_p"
]

mean_scores = np.zeros((n_datasets, len(methods)))
stds = np.zeros((n_datasets, len(methods)))
mean_costs = np.zeros((n_datasets, len(methods)))
datasets = []
n_base_clfs = len(base_classifiers)

# Load data from file
for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    datasets.append(dataset)
    for clf_id, clf_name in enumerate(methods):
        try:
            # Accuracy
            filename_acc = "results/experiment1/accuracy/%s/%s.csv" % (dataset, clf_name)
            scores = np.genfromtxt(filename_acc, delimiter=',', dtype=np.float32)
            mean_score = np.mean(scores)
            mean_scores[dataset_id, clf_id] = mean_score
            std = np.std(scores)
            stds[dataset_id, clf_id] = std

            # Cost
            filename_cost = "results/experiment1/cost/%s/%s.csv" % (dataset, clf_name)
            total_cost = np.genfromtxt(filename_cost, delimiter=',', dtype=np.float32)
            mean_cost = np.mean(total_cost)
            mean_costs[dataset_id, clf_id] = mean_cost

        except IOError:
            print("File", filename_acc, "not found")
            print("File", filename_cost, "not found")

# Latex tables
column_names = ["GA acc cost", "NSGA acc"]
for key in base_classifiers:
    acc_contents = [[None for _ in range(len(opt_methods))] for _ in range(n_datasets)]
    cost_contents = [[None for _ in range(len(opt_methods))] for _ in range(n_datasets)]
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        i = 0
        for clf_id, clf_name in enumerate(methods):
            if key in clf_name:
                # Accuracy
                ms = mean_scores[dataset_id, clf_id].tolist()
                std = stds[dataset_id, clf_id].tolist()
                string = str(float("{:.2f}".format(ms))) + " +- " + str(float("{:.2f}".format(std)))
                acc_contents[dataset_id][i] = string

                # Cost
                mc = mean_costs[dataset_id, clf_id].tolist()
                value = float("{:.2f}".format(mc))
                cost_contents[dataset_id][i] = value
                i += 1

        if not os.path.exists("results/experiment1/tables/"):
            os.makedirs("results/experiment1/tables/")

        acc_df = pd.DataFrame(data=acc_contents, index=datasets, columns=column_names)
        print("\nACCURACY:\n", acc_df)
        filename_acc = "results/experiment1/tables/Acc_%s.tex" % (key)
        with open(filename_acc, 'w') as f:
            f.write(acc_df.to_latex())

        cost_df = pd.DataFrame(data=cost_contents, index=datasets, columns=column_names)
        print("\nCOST:\n", cost_df)
        filename_cost = "results/experiment1/tables/Cost_%s.tex" % (key)
        with open(filename_cost, 'w') as f:
            f.write(cost_df.to_latex())
