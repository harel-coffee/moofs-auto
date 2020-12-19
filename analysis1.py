import os
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import chi2

from utils import find_datasets, load_feature_costs
from methods.fsclf import FeatueSelectionClf
from methods.gaaccclf import GeneticAlgorithmAccuracyClf
from methods.gaacccost import GAAccCost
from methods.nsgaacccost import NSGAAccCost

DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

base_classifiers = {
    # 'GNB': GaussianNB(),
    # 'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=10),
}

test_size = 0.2

methods_alias = [
                # "FS",
                # "GA_a",
                "GA_ac",
                "NSGA_a",
                "NSGA_c",
                # "NSGA_p"
]
n_methods = len(methods_alias) * len(base_classifiers)
mean_scores = np.zeros((n_datasets, 21, n_methods))
stds = np.zeros((n_datasets, 21, n_methods))
mean_costs = np.zeros((n_datasets, 21, n_methods))
datasets = []
n_base_clfs = len(base_classifiers)
# Pareto decision for NSGA
pareto_decision_a = 'accuracy'
pareto_decision_c = 'cost'
# pareto_decision_p = 'promethee'

# Load data from file
for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    print(f"Dataset: {dataset}")
    datasets.append(dataset)
    feature_number = len(load_feature_costs(dataset))
    scale_features = np.linspace(1/feature_number, 1.0, feature_number)
    scale_features += 0.01

    for scale_id, scale in enumerate(scale_features):
        selected_feature_number = int(scale * feature_number)
        print(f"Number of selected features: {selected_feature_number}")

        methods = {}
        for key, base in base_classifiers.items():
            # methods['FS_{}'.format(key)] = FeatueSelectionClf(base, chi2, scale)
            # methods['GAacc_{}'.format(key)] = GeneticAlgorithmAccuracyClf(base, scale, test_size)
            methods['GAaccCost_{}'.format(key)] = GAAccCost(base, scale, test_size)

            methods['NSGAaccCost_acc_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_a)
            methods['NSGAaccCost_cost_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_c)
            # methods['NSGAaccCost_promethee_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_p)

        for clf_id, clf_name in enumerate(methods):
            try:
                print(clf_name)
                # Accuracy
                filename_acc = "results/experiment1/accuracy/%s/f%d/%s.csv" % (dataset, selected_feature_number, clf_name)
                scores = np.genfromtxt(filename_acc, delimiter=',', dtype=np.float32)
                mean_score = np.mean(scores)
                mean_scores[dataset_id, scale_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, scale_id, clf_id] = std
                print("Accuracy: ", mean_score, std)

                # Cost
                filename_cost = "results/experiment1/cost/%s/f%d/%s.csv" % (dataset, selected_feature_number, clf_name)
                total_cost = np.genfromtxt(filename_cost, delimiter=',', dtype=np.float32)
                mean_cost = np.mean(total_cost)
                mean_costs[dataset_id, scale_id, clf_id] = mean_cost
                print("Cost: ", mean_cost)

            except IOError:
                print("File", filename_acc, "not found")
                print("File", filename_cost, "not found")

n_features = []
for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    n_features.append(len(load_feature_costs(dataset)))
# Latex tables
for key in base_classifiers:
    acc_contents = [[None for _ in range(len(methods_alias))] for _ in range(n_datasets)]
    cost_contents = [[None for _ in range(len(methods_alias))] for _ in range(n_datasets)]
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        feature_number = len(load_feature_costs(dataset))
        scale_features = np.linspace(1/feature_number, 1.0, feature_number)
        scale_features += 0.01

        # for nf_id, nf in enumerate(n_features):
        #     if (dataset_id != nf_id) and (feature_number <= nf):
        #         print("TAK")

        for scale_id, scale in enumerate(scale_features):
            selected_feature_number = int(scale * feature_number)
            i = 0
            for clf_id, clf_name in enumerate(methods):
                if key in clf_name:
                    # Accuracy
                    ms = mean_scores[dataset_id, scale_id, clf_id].tolist()
                    std = stds[dataset_id, scale_id, clf_id].tolist()
                    string = str(float("{:.2f}".format(ms))) + " +- " + str(float("{:.2f}".format(std)))
                    acc_contents[dataset_id][i] = string

                    # Cost
                    mc = mean_costs[dataset_id, scale_id, clf_id].tolist()
                    value = float("{:.2f}".format(mc))
                    cost_contents[dataset_id][i] = value
                    i += 1

            if not os.path.exists("results/experiment1/tables/f%d/" % (selected_feature_number)):
                os.makedirs("results/experiment1/tables/f%d/" % (selected_feature_number))
            acc_df = pd.DataFrame(data=acc_contents, index=datasets, columns=methods_alias)
            print("\nACCURACY:\n", acc_df)
            filename_acc = "results/experiment1/tables/f%d/Acc_%s.tex" % (selected_feature_number, key)
            with open(filename_acc, 'w') as f:
                f.write(acc_df.to_latex())

            cost_df = pd.DataFrame(data=cost_contents, index=datasets, columns=methods_alias)
            print("\nCOST:\n", cost_df)
            filename_cost = "results/experiment1/tables/f%d/Cost_%s.tex" % (selected_feature_number, key)
            with open(filename_cost, 'w') as f:
                f.write(cost_df.to_latex())
