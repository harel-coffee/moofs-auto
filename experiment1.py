import os
import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

from utils import load_dataset, find_datasets, load_feature_costs
from methods.fsclf import FeatueSelectionClf
from methods.gaaccclf import GeneticAlgorithmAccuracyClf
from methods.gaacccost import GAAccCost
from methods.nsgaacccost import NSGAAccCost


import seaborn as sns


# !!! Move specific datasets into dataset folder !!!
DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

base_classifiers = {
    # 'GNB': GaussianNB(),
    # 'SVM': SVC(),
    # 'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=10),
}

test_size = 0.2
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
n_folds = n_splits * n_repeats
# Pareto decision for NSGA
pareto_decision_a = 'accuracy'
pareto_decision_c = 'cost'
pareto_decision_p = 'promethee'
criteria_weights = np.array([0.5, 0.5])
n_rows_p = 50

# Dodaj zabezpieczenie, że jeśli coś jest już policzone, to żeby się nie liczyło od nowa

for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    # Get feature names
    with open('datasets/%s/%s-header.txt' % (dataset, dataset), 'rt') as file:
        header = file.read()
        a, feature_names = header.split("@inputs ")
        feature_names = feature_names.split("\n")
        del feature_names[-2:]
        feature_names = feature_names[0].split(', ')
    print(f"Dataset: {dataset}")
    print(f"Features: {feature_names}")

    feature_number = len(feature_names)
    scale_features = np.linspace(1/feature_number, 1.0, feature_number)
    scale_features += 0.01

    X, y = load_dataset(dataset, return_X_y=True, storage=DATASETS_DIR)

    # Pair plot shows placement of classes according to features
    # sns.set_theme(style="whitegrid")
    # df = pd.DataFrame({'classes': y[:], 'Preg': X[:, 0], 'Plas': X[:, 1], 'Pres': X[:, 2], 'Skin': X[:, 3], 'Insu': X[:, 4], 'Mass': X[:, 5], 'Pedi': X[:, 6], 'Age': X[:, 7]})
    # fig = sns.pairplot(df, hue="classes")
    # fig.savefig("results/output.png")

    # Normalization - transform data to [0, 1]
    X = MinMaxScaler().fit_transform(X, y)

    # Get feature costs
    feature_costs = load_feature_costs(dataset)
    # Normalization
    feature_costs_norm = [(float(i)-min(feature_costs))/(max(feature_costs)-min(feature_costs)) for i in feature_costs]
    feature_costs_norm_after = []
    for f_norm in feature_costs_norm:
        f_norm += 0.01
        feature_costs_norm_after.append(f_norm)
    print(feature_costs_norm_after)

    for scale in scale_features:
        methods = {}
        for key, base in base_classifiers.items():
            # methods['FS_{}'.format(key)] = FeatueSelectionClf(base, chi2, scale)
            # methods['GAacc_{}'.format(key)] = GeneticAlgorithmAccuracyClf(base, scale, test_size)
            # methods['GAaccCost_{}'.format(key)] = GAAccCost(base, scale, test_size)

            # methods['NSGAaccCost_acc_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_a)
            # methods['NSGAaccCost_cost_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_c)
            methods['NSGAaccCost_promethee_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_p, criteria_weights)

        selected_feature_number = int(scale * feature_number)
        print(f"Number of selected features: {selected_feature_number}")

        scores = np.zeros((len(methods), n_folds))
        total_cost = np.zeros((len(methods), n_folds))
        pareto_solutions = np.zeros((n_folds, n_rows_p, 2))

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            for clf_id, clf_name in enumerate(methods):
                print(f"Fold number: {fold_id}, clf: {clf_name}")
                clf = clone(methods[clf_name])
                clf.feature_costs = feature_costs_norm_after
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                scores[clf_id, fold_id] = accuracy_score(y_test, y_pred)
                total_cost[clf_id, fold_id] = clf.selected_features_cost()

                if hasattr(clf, 'solutions'):
                    for sol_id, solution in enumerate(clf.solutions):
                        for s_id, s in enumerate(solution):
                            pareto_solutions[fold_id, sol_id, s_id] = s
        print(scores)
        # Save results accuracy and total cost of selected features to csv
        for clf_id, clf_name in enumerate(methods):
            filename_acc = "results/experiment1/accuracy/%s/f%d/%s.csv" % (dataset, selected_feature_number, clf_name)
            if not os.path.exists("results/experiment1/accuracy/%s/f%d/" % (dataset, selected_feature_number)):
                os.makedirs("results/experiment1/accuracy/%s/f%d/" % (dataset, selected_feature_number))
            np.savetxt(fname=filename_acc, fmt="%f", X=scores[clf_id, :])

            filename_cost = "results/experiment1/cost/%s/f%d/%s.csv" % (dataset, selected_feature_number, clf_name)
            if not os.path.exists("results/experiment1/cost/%s/f%d/" % (dataset, selected_feature_number)):
                os.makedirs("results/experiment1/cost/%s/f%d/" % (dataset, selected_feature_number))
            np.savetxt(fname=filename_cost, fmt="%f", X=total_cost[clf_id, :])

        # Save results pareto_solutions to csv
        for fold_id in range(n_folds):
            for sol_id in range(n_rows_p):
                if (pareto_solutions[fold_id, sol_id, 0] != 0.0) and (pareto_solutions[fold_id, sol_id, 1] != 0.0):
                    filename_pareto = "results/experiment1/pareto/%s/f%d/fold%d/sol%d.csv" % (dataset, selected_feature_number, fold_id, sol_id)
                    if not os.path.exists("results/experiment1/pareto/%s/f%d/fold%d/" % (dataset, selected_feature_number, fold_id)):
                        os.makedirs("results/experiment1/pareto/%s/f%d/fold%d/" % (dataset, selected_feature_number, fold_id))
                    np.savetxt(fname=filename_pareto, fmt="%f", X=pareto_solutions[fold_id, sol_id, :])


# Na serwerze liczy się teraz:
# tylko dla thyroid
# metody: FS, GA ac, GA cost,
