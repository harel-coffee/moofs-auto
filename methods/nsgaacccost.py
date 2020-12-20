import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling

from sklearn.base import ClassifierMixin, BaseEstimator

from methods.optimization.optimizationAccCostMulti import FeatureSelectionAccuracyCostMultiProblem


class NSGAAccCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, scale_features=0.5, test_size=0.5, pareto_decision='accuracy', objectives=2, p_size=100, c_prob=0.1, m_prob=0.1):
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
        # self.solutions = None
        # self.solutions = []
        # self.solutions = np.zeros((2))
        self.pareto_decision = pareto_decision
        self.objectives = objectives
        self.scale_features = scale_features

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
        # F returns solutions [-accuracy, total_cost]
        self.solutions = res.F
        # X returns True and False which features has been selected
        if self.pareto_decision == 'accuracy':
            index = np.argmin(res.F[:, 0], axis=0)
            self.selected_features = res.X[index]
        elif self.pareto_decision == 'cost':
            index = np.argmin(res.F[:, 1], axis=0)
            self.selected_features = res.X[index]
        # elif self.pareto_decision == 'promethee':
        #     xx

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

# def promethee():
#     # 1. porównaj parowo każdy wiersz z każdym wierszem z uzyskanych rozwiązań: res.F[:,1]
#     # 1. Pairwise differences between each solution are computed for all objectives.
#     # 2. A preference function, based on the significance and insignifi- cance levels, is applied.
#     # 3. The overall preference index is computed by performing a weighted sum of the objectives for each pairwise comparison.
#     # 4. Positive and negative outranking flows are calculated for each solution, based on the overall pairwise comparisons.
#     # 5. The positive and negative outranking flows are subtracted to compute a final outranking flow, used for ranking the solutions.
#
#     return 0
