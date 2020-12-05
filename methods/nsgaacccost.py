import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.visualization.scatter import Scatter

from sklearn.base import ClassifierMixin, BaseEstimator

from methods.optimization.optimizationAccCostMulti import FeatureSelectionAccuracyCostMultiProblem


class NSGAAccCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, scale_features=0.5, objectives=2, test_size=0.5, p_size=100, c_prob=0.1, m_prob=0.1):
        self.base_estimator = base_estimator
        self.test_size = test_size
        self.p_size = p_size
        self.c_prob = c_prob
        self.m_prob = m_prob

        self.feature_costs = None
        self.estimator = None
        self.res = None
        self.selected_features = None
        self.objectives = objectives
        self.scale_features = scale_features

    def fit(self, X, y):
        features = range(X.shape[1])
        problem = FeatureSelectionAccuracyCostMultiProblem(X, y, self.test_size, self.base_estimator, features, self.objectives, self.feature_costs, self.scale_features)

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

        print("Selected features for each fold: {}".format(np.sum(res.opt.get('SF')[0])))
        print(res.opt.get('SF')[0])
        # Scatter().add(res.F).show()

        self.selected_features = res.opt.get('SF')[0]
        self.estimator = self.base_estimator.fit(X[:, self.selected_features], y)
        return self

    def predict(self, X):
        return self.estimator.predict(X[:, self.selected_features])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[:, self.selected_features])
