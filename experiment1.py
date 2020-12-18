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

from utils import load_dataset, find_datasets, load_feature_costs, plotting_pareto
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
# scale_features = 0.7
# methods = {}
# for key, base in base_classifiers.items():
#     methods['FS_{}'.format(key)] = FeatueSelectionClf(base, chi2, scale_features)
#     methods['GAacc_{}'.format(key)] = GeneticAlgorithmAccuracyClf(base, scale_features, test_size)
#     methods['GAaccCost_{}'.format(key)] = GAAccCost(base, scale_features, test_size)
#
#     methods['NSGAaccCost_acc_{}'.format(key)] = NSGAAccCost(base, scale_features, test_size)
#     methods['NSGAaccCost_cost_{}'.format(key)] = NSGAAccCost(base, scale_features, test_size)
#     # methods['NSGAaccCost_promethee_{}'.format(key)] = NSGAAccCost(base, scale_features, test_size)

# Pareto decision NSGA
pareto_decision = 'accuracy'
pareto_decision = 'cost'
# pareto_decision = 'promethee'

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
n_folds = n_splits * n_repeats

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


    X, y = load_dataset(dataset, return_X_y=True, storage=DATASETS_DIR)
    # Normalization - transform data to [0, 1]
    X = MinMaxScaler().fit_transform(X, y)

    # Get feature costs
    feature_costs = load_feature_costs(dataset)
    # Normalization
    feature_costs_norm = [(float(i)-min(feature_costs))/(max(feature_costs)-min(feature_costs)) for i in feature_costs]

    for scale in scale_features:
        methods = {}
        for key, base in base_classifiers.items():
            methods['FS_{}'.format(key)] = FeatueSelectionClf(base, chi2, scale)
            # methods['GAacc_{}'.format(key)] = GeneticAlgorithmAccuracyClf(base, scale, test_size)
            # methods['GAaccCost_{}'.format(key)] = GAAccCost(base, scale, test_size)
            #
            # methods['NSGAaccCost_acc_{}'.format(key)] = NSGAAccCost(base, scale, test_size)
            # methods['NSGAaccCost_cost_{}'.format(key)] = NSGAAccCost(base, scale, test_size)
            # methods['NSGAaccCost_promethee_{}'.format(key)] = NSGAAccCost(base, scale, test_size)

        scale_percent = int(scale * 100)

        selected_feature_number = int(scale * len(feature_names))
        print(f"Number of selected features: {selected_feature_number}")

        scores = np.zeros((len(methods), n_folds))
        total_cost = np.zeros((len(methods), n_folds))

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            for clf_id, clf_name in enumerate(methods):
                print(f"Fold number: {fold_id}, clf: {clf_name}")
                clf = clone(methods[clf_name])
                clf.feature_costs = feature_costs_norm

                if hasattr(clf, 'solutions'):
                    filename = ("%s_%s" % (clf_name, fold_id))
                    print(clf.solutions)
                    plotting_pareto(clf.solutions, filename)
                if hasattr(clf, 'pareto_decision'):
                    clf.pareto_decision = pareto_decision

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                scores[clf_id, fold_id] = accuracy_score(y_test, y_pred)

                total_cost[clf_id, fold_id] = clf.selected_features_cost()

        # Save results to csv
        for clf_id, clf_name in enumerate(methods):
            filename_acc = "results/experiment1/accuracy/%s/f%d/%s.csv" % (dataset, selected_feature_number, clf_name)
            if not os.path.exists("results/experiment1/accuracy/%s/f%d/" % (dataset, selected_feature_number)):
                os.makedirs("results/experiment1/accuracy/%s/f%d/" % (dataset, selected_feature_number))
            np.savetxt(fname=filename_acc, fmt="%f", X=scores[clf_id, :])

            filename_cost = "results/experiment1/cost/%s/f%d/%s.csv" % (dataset, selected_feature_number, clf_name)
            if not os.path.exists("results/experiment1/cost/%s/f%d/" % (dataset, selected_feature_number)):
                os.makedirs("results/experiment1/cost/%s/f%d/" % (dataset, selected_feature_number))
            np.savetxt(fname=filename_cost, fmt="%f", X=total_cost[clf_id, :])
