import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.visualization.scatter import Scatter

from sklearn.base import ClassifierMixin, BaseEstimator

from methods.optimization.optimizationAccCostMulti import FeatureSelectionAccuracyCostMultiProblem


class NSGAAccCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, scale_features=0.5, test_size=0.5, objectives=2, p_size=100, c_prob=0.1, m_prob=0.1):
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
        self.pareto_decision = 'accuracy'
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

        # Plotting Pareto front - jak będą potrzebne, to zrób ładniejsze
        plot = Scatter(title="Objective Space")
        plot.add(res.F, color="red")
        plot.save('results/experiment1/figures/scatter/%s.png' % (self.fig_filename))

        # Select solution from the Pareto front
        # F zwraca wartości accuracy i total cost wybranych rozwiązań:
        # [[-0.65306122  0.51196363]
         # [-0.67346939  0.53793896]]
        print("F", res.F)
        # X zwraca wektor wartości True i False które cechy zostały wybrane dla danych rozwiązań Pareto
        # [[ True  True  True  True  True  True  True False  True  True False  True  False]
        # [ True  True  True  True  True  True  True  True  True False False  True  False]]
        print("X", res.X)
        if self.pareto_decision == 'accuracy':
            index = np.argmin(res.F[:,0], axis=0)
            self.selected_features = res.X[index]
        elif self.pareto_decision == 'cost':
            index = np.argmin(res.F[:,1], axis=0)
            self.selected_features = res.X[index]
        # elif self.pareto_decision == 'promethee':
        #     xx

        print("Selected features for each fold: {}".format(np.sum(self.selected_features)))
        print(self.selected_features)

        self.estimator = self.base_estimator.fit(X[:, self.selected_features], y)
        return self

    def predict(self, X):
        return self.estimator.predict(X[:, self.selected_features])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[:, self.selected_features])
