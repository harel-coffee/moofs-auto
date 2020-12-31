import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling

from sklearn.base import ClassifierMixin, BaseEstimator

from methods.optimization.optimizationAccCostMulti import FeatureSelectionAccuracyCostMultiProblem


class NSGAAccCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, scale_features=0.5, test_size=0.5, pareto_decision='accuracy', criteria_weights=None, objectives=2, p_size=100, c_prob=0.1, m_prob=0.1):
        self.base_estimator = base_estimator
        self.test_size = test_size
        self.p_size = p_size
        self.c_prob = c_prob
        self.m_prob = m_prob

        self.feature_costs = None
        self.estimator = None
        self.res = None
        self.selected_features = None
        self.fig_filename = None
        self.solutions = None
        self.pareto_decision = pareto_decision
        self.objectives = objectives
        self.scale_features = scale_features
        self.criteria_weights = criteria_weights

    def fit(self, X, y):
        features = range(X.shape[1])
        problem = FeatureSelectionAccuracyCostMultiProblem(X, y, self.test_size, self.base_estimator, features, self.feature_costs, self.scale_features, self.objectives)

        algorithm = NSGA2(
                       pop_size=self.p_size,
                       sampling=get_sampling("bin_random"),
                       crossover=get_crossover("bin_two_point"),
                       mutation=get_mutation("bin_bitflip"),
                       eliminate_duplicates=True)

        res = minimize(
                       problem,
                       algorithm,
                       ('n_eval', 1000),
                       seed=1,
                       verbose=False,
                       save_history=True)

        # Select solution from the Pareto front
        # F returns all solutions in form [-accuracy, total_cost]
        self.solutions = res.F

        # X returns True and False which features has been selected
        if self.pareto_decision == 'accuracy':
            index = np.argmin(self.solutions[:, 0], axis=0)
            self.selected_features = res.X[index]
        elif self.pareto_decision == 'cost':
            index = np.argmin(self.solutions[:, 1], axis=0)
            self.selected_features = res.X[index]
        elif self.pareto_decision == 'promethee':
            # if only one solution has been found
            if self.solutions.shape[0] == 1:
                index = 0
                self.selected_features = res.X[index]
            else:
                # criteria min (0) or max (1) optimization array
                self.criteria_min_max = ([0, 0])
                # u - usual
                self.preference_function = (['u', 'u'])
                net_flows = promethee_function(self.solutions, self.criteria_min_max, self.preference_function, self.criteria_weights)
                # Ranking of the net flows
                index = np.argmax(net_flows, axis=0)
                self.selected_features = res.X[index]

        self.estimator = self.base_estimator.fit(X[:, self.selected_features], y)
        return self

    def predict(self, X):
        return self.estimator.predict(X[:, self.selected_features])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[:, self.selected_features])

    def selected_features_cost(self):
        total_cost = 0
        for id, cost in enumerate(self.feature_costs):
            if self.selected_features[id]:
                total_cost += cost
        return total_cost


# Calculation uni weighted to promethee method
def uni_cal(solutions_col, criteria_min_max, preference_function, criteria_weights):
    uni = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
    uni_weighted = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
    for i in range(np.size(uni, 0)):
        for j in range(np.size(uni, 1)):
            if i == j:
                uni[i, j] = 0
            # Usual preference function
            elif preference_function == 'u':
                diff = solutions_col[j] - solutions_col[i]
                if diff > 0:
                    uni[i, j] = 1
                else:
                    uni[i, j] = 0
            uni_weighted[i][j] = criteria_weights * uni[i, j]
    # criteria min (0) or max (1) optimization array
    if criteria_min_max == 0:
        uni_weighted = uni_weighted
    elif criteria_min_max == 1:
        uni_weighted = uni_weighted.T
    return uni_weighted


# promethee method to choose one solution from the pareto front
def promethee_function(solutions, criteria_min_max, preference_function, criteria_weights):
    weighted_unis = []
    for i in range(solutions.shape[1]):
        weighted_uni = uni_cal(solutions[:, i:i + 1], criteria_min_max[i], preference_function[i], criteria_weights[i])
        weighted_unis.append(weighted_uni)
    agregated_preference = []
    uni_acc = weighted_unis[0]
    uni_cost = weighted_unis[1]
    # Combine two criteria into agregated_preference
    for (item1, item2) in zip(uni_acc, uni_cost):
        agregated_preference.append((item1 + item2)/sum(criteria_weights))
    agregated_preference = np.array(agregated_preference)

    n_solutions = agregated_preference.shape[0] - 1
    # Sum by rows - positive flow
    pos_flows = []
    pos_sum = np.sum(agregated_preference, axis=1)
    for element in pos_sum:
        pos_flows.append(element/n_solutions)
    # Sum by columns - negative flow
    neg_flows = []
    neg_sum = np.sum(agregated_preference, axis=0)
    for element in neg_sum:
        neg_flows.append(element/n_solutions)
    # Calculate net_flows
    net_flows = []
    for i in range(len(pos_flows)):
        net_flows.append(pos_flows[i] - neg_flows[i])
    return net_flows
