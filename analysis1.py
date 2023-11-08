import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import chi2

from utils import find_datasets, load_feature_costs
# from methods.fsclf import FeatueSelectionClf
# from methods.gaaccclf import GeneticAlgorithmAccuracyClf
# from methods.gaacccost import GAAccCost
# from methods.nsgaacccost import NSGAAccCost


DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

# !!! Change before run !!!
base_classifiers = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=10),
}

base_classifiers_alias = [
                        'GNB',
                        'SVM',
                        'kNN',
                        'CART'
]

# !!! Change before run !!!
methods_alias = [
                "FS",
                "GA-a",
                "GA-ac",
                "NSGA-a",
                "NSGA-c",
                "NSGA-p"
]

test_size = 0.2
n_folds = 10
n_methods = len(methods_alias) * len(base_classifiers)
methods_labels = methods_alias * len(base_classifiers)
# Number 21 is a number of max features among dataset
mean_scores = np.zeros((n_datasets, 21, n_methods))
stds = np.zeros((n_datasets, 21, n_methods))
mean_costs = np.zeros((n_datasets, 21, n_methods))
datasets = []
n_base_clfs = len(base_classifiers)
# Pareto decision for NSGA
pareto_decision_a = 'accuracy'
pareto_decision_c = 'cost'
pareto_decision_p = 'promethee'
n_rows_p = 50

# Load data from file
for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    datasets.append(dataset)
    feature_number = len(load_feature_costs(dataset))
    scale_features = np.linspace(1/feature_number, 1.0, feature_number)
    scale_features += 0.01
    for scale_id, scale in enumerate(scale_features):
        selected_feature_number = int(scale * feature_number)
        methods = {}
        for key, base in base_classifiers.items():

            # !!! Change before run !!!
            # old version of pymoo
            # methods['FS_{}'.format(key)] = FeatueSelectionClf(base, chi2, scale)
            # methods['GAacc_{}'.format(key)] = GeneticAlgorithmAccuracyClf(base, scale, test_size)
            # methods['GAaccCost_{}'.format(key)] = GAAccCost(base, scale, test_size)
            # methods['NSGAaccCost_acc_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_a)
            # methods['NSGAaccCost_cost_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_c)
            # methods['NSGAaccCost_promethee_{}'.format(key)] = NSGAAccCost(base, scale, test_size, pareto_decision_p)

            methods['FS_{}'.format(key)] = 0
            methods['GAacc_{}'.format(key)] = 0
            methods['GAaccCost_{}'.format(key)] = 0
            methods['NSGAaccCost_acc_{}'.format(key)] = 0
            methods['NSGAaccCost_cost_{}'.format(key)] = 0
            methods['NSGAaccCost_promethee_{}'.format(key)] = 0

        for clf_id, clf_name in enumerate(methods):
            try:
                # Load accuracy score
                filename_acc = "results/experiment1/accuracy/%s/f%d/%s.csv" % (dataset, selected_feature_number, clf_name)
                scores = np.genfromtxt(filename_acc, delimiter=',', dtype=np.float32)
                mean_score = np.mean(scores)
                mean_scores[dataset_id, scale_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, scale_id, clf_id] = std

                # Load feature cost
                filename_cost = "results/experiment1/cost/%s/f%d/%s.csv" % (dataset, selected_feature_number, clf_name)
                total_cost = np.genfromtxt(filename_cost, delimiter=',', dtype=np.float32)
                mean_cost = np.mean(total_cost)
                mean_costs[dataset_id, scale_id, clf_id] = mean_cost
            except IOError:
                print("File", filename_acc, "not found")
                print("File", filename_cost, "not found")
# print(mean_scores)


# Plotting
# Bar chart function
def bar_chart():
    width = 1/(len(methods_alias)+1)
    tr = 1/(len(methods_alias)/2)
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        print(dataset)
        feature_number = len(load_feature_costs(dataset))
        scale_features = np.linspace(1/feature_number, 1.0, feature_number)
        scale_features += 0.01
        # Plotting accuracy
        for key, base in base_classifiers.items():
            r = 0
            for clf_id, (clf_name, method_label) in enumerate(zip(methods, methods_labels)):
                if key in clf_name:
                    plot_data = []
                    for scale_id, scale in enumerate(scale_features):
                        plot_data.append(mean_scores[dataset_id, scale_id, clf_id])
                    position = list(range(1, len(plot_data)+1))
                    # Add plot_data to bars in the chart
                    plt.bar([p - tr + width*r for p in position], plot_data, width, edgecolor='white', label=method_label)
                    r += 1
            # Save plot
            filename = "results/experiment1/plot_bar/bar_%s_%s_acc" % (dataset, key)
            if not os.path.exists("results/experiment1/plot_bar/"):
                os.makedirs("results/experiment1/plot_bar/")

            plt.ylabel("Accuracy")
            plt.xlabel("Number of selected features")
            plt.ylim(bottom=0.0, top=1.0)
            plt.title(f"Accuracy for dataset {dataset} and base classifier {key}")
            plt.legend(loc='lower right')
            plt.grid(True, color="silver", linestyle=":", axis='y')
            plt.xticks(range(1, len(plot_data)+1), labels=range(1, len(plot_data)+1))
            plt.gcf().set_size_inches(9, 6)
            plt.savefig(filename+".png", bbox_inches='tight')
            plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
            plt.clf()
            plt.close()
        # Plotting cost
        for key, base in base_classifiers.items():
            r = 0
            for clf_id, (clf_name, method_label) in enumerate(zip(methods, methods_labels)):
                if key in clf_name:
                    plot_data = []
                    for scale_id, scale in enumerate(scale_features):
                        plot_data.append(mean_costs[dataset_id, scale_id, clf_id])
                    position = list(range(1, len(plot_data)+1))
                    # Add plot_data to bars in the chart
                    plt.bar([p - tr + width*r for p in position], plot_data, width, edgecolor='white', label=method_label)
                    r += 1
            # Save plot
            filename = "results/experiment1/plot_bar/bar_%s_%s_cost" % (dataset, key)
            if not os.path.exists("results/experiment1/plot_bar/"):
                os.makedirs("results/experiment1/plot_bar/")

            plt.ylabel("Cost")
            plt.xlabel("Number of selected features")
            plt.ylim(bottom=0.0)
            plt.title(f"Cost for dataset {dataset} and base classifier {key}")
            plt.legend(loc='best')
            plt.xticks(range(1, len(plot_data)+1), labels=range(1, len(plot_data)+1))
            plt.grid(True, color="silver", linestyle=":", axis='y')
            plt.gcf().set_size_inches(9, 6)
            plt.savefig(filename+".png", bbox_inches='tight')
            plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
            plt.clf()
            plt.close()


# Plotting bar chart
# bar_chart()


def total_norm_feat_cost(dataset_name):
    # Get feature costs
    feature_costs = load_feature_costs(dataset_name)
    # Normalization
    feature_costs_norm = [(float(i)-min(feature_costs))/(max(feature_costs)-min(feature_costs)) for i in feature_costs]
    feature_costs_norm_after = []
    for f_norm in feature_costs_norm:
        f_norm += 0.01
        feature_costs_norm_after.append(f_norm)
    return sum(feature_costs_norm_after)


# Micro chart function
def micro_chart():
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        print(dataset)
        total_norm_cost = total_norm_feat_cost(dataset)
        feature_number = len(load_feature_costs(dataset))
        scale_features = np.linspace(1/feature_number, 1.0, feature_number)
        scale_features += 0.01
        figsize = (10, 15)
        cols = int(len(methods_alias)/2)
        rows = len(base_classifiers)*2
        axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols, sharex=True, sharey=True)
        axs = axs.flatten()
        axs2 = [0] * len(axs)

        for clf_id, (clf_name, method_label) in enumerate(zip(methods, methods_labels)):
            for b_clf_id, b_clf_a in enumerate(base_classifiers_alias):
                if b_clf_a in clf_name:
                    plot_data_acc = []
                    plot_data_cost = []
                    for scale_id, scale in enumerate(scale_features):
                        plot_data_acc.append(mean_scores[dataset_id, scale_id, clf_id])
                        plot_data_cost.append(mean_costs[dataset_id, scale_id, clf_id])
                    position = list(range(1, len(plot_data_acc)+1))

                    if b_clf_a == 'CART':
                        b_clf_a = 'DT'
                    tittle_subfig = method_label + " " + b_clf_a
                    axs[clf_id].plot(position, plot_data_acc, color="tab:orange", linewidth=2.5)
                    axs[clf_id].set_title(tittle_subfig, fontsize=12)
                    # make a plot with different y-axis using second axis object
                    axs2[clf_id] = axs[clf_id].twinx()
                    axs2[clf_id].plot(position, plot_data_cost, color="tab:blue", linewidth=2.5)
                    plt.setp(axs2[clf_id], yticks=range(0, int(total_norm_cost + 2)))

        # Save plot
        filename = "results/experiment1/plot_micro/micro_%s" % (dataset)
        if not os.path.exists("results/experiment1/plot_micro/"):
            os.makedirs("results/experiment1/plot_micro/")
        plt.suptitle(f"Dataset {dataset}", fontweight='bold')
        for i in range(0, rows*cols, 3):
            axs[i].set_ylabel("Accuracy", color="tab:orange", fontsize=12, fontweight='bold')
        for i in range(2, rows*cols, 3):
            axs2[i].set_ylabel("Cost", color="tab:blue", fontsize=12, fontweight='bold')
        for i in range(21, rows*cols):
            axs[i].set_xlabel("Features", fontsize=12)
        if feature_number > 10:
            plt.xticks(range(1, int(feature_number + 1), 2), labels=range(1, int(feature_number + 1), 2))
        else:
            plt.xticks(range(1, int(feature_number + 1), 1), labels=range(1, int(feature_number + 1), 1))
        plt.savefig(filename+".png", bbox_inches='tight')
        plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()


# Plotting micro charts
micro_chart()


# Plot pareto front scatter function
def scatter_pareto_chart():
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        print(dataset)
        for scale_id, scale in enumerate(scale_features):
            selected_feature_number = int(scale * feature_number)
            for fold_id in range(n_folds):
                solutions = []
                for sol_id in range(n_rows_p):
                    try:
                        filename_pareto = "results/experiment1/pareto/%s/f%d/fold%d/sol%d.csv" % (dataset, selected_feature_number, fold_id, sol_id)
                        solution = np.genfromtxt(filename_pareto, dtype=np.float32)
                        solution = solution.tolist()
                        solution[0] = solution[0] * (-1)
                        solutions.append(solution)
                    except IOError:
                        pass
                filename_pareto_chart = "results/experiment1/plot_scatter/%s/%s_f%d_fold%d_pareto" % (dataset, dataset, selected_feature_number, fold_id)
                if not os.path.exists("results/experiment1/plot_scatter/%s/" % (dataset)):
                    os.makedirs("results/experiment1/plot_scatter/%s/" % (dataset))
                # print(solutions)
                x = []
                y = []
                for solution in solutions:
                    x.append(solution[0])
                    y.append(solution[1])
                x = np.array(x)
                y = np.array(y)
                plt.scatter(x, y, color='black')
                plt.title("Objective Space", fontsize=12)
                plt.xlabel('Accuracy', fontsize=12)
                plt.ylabel('Cost', fontsize=12)
                plt.grid(True, color="silver", linestyle=":", axis='both')
                plt.gcf().set_size_inches(6, 3)
                plt.savefig(filename_pareto_chart+".png", bbox_inches='tight')
                plt.savefig(filename_pareto_chart+".eps", format='eps', bbox_inches='tight')
                plt.clf()
                plt.close()


# Plot pareto front scatter
# scatter_pareto_chart()
