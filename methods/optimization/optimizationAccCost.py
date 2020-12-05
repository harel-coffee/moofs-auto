import numpy as np

from pymoo.model.problem import Problem

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone


class FeatureSelectionAccuracyCostProblem(Problem):
    def __init__(self, X, y, test_size, estimator, feature_names, objectives, feature_costs, scale_features=0.5, random_state=0):
        self.y = y
        self.test_size = test_size
        self.estimator = estimator
        self.objectives = objectives
        self.L = feature_names
        self.n_max = len(self.L)
        self.scale_features = scale_features
        self.feature_costs = feature_costs

        # If test size is not specify or it is 0, everything is took to test and train
        if self.test_size != 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = np.copy(X), np.copy(y), np.copy(X), np.copy(y)

        super().__init__(n_var=len(self.L), n_obj=objectives,
                         n_constr=1, elementwise_evaluation=True)

    def validation(self, x):
        clf = clone(self.estimator)
        if all(not element for element in x):
            # All elements in x are False
            metrics = 0
            return metrics
        else:
            # Not all elements in x are False
            clf.fit(self.X_train[:, x], self.y_train)
            y_pred = clf.predict(self.X_test[:, x])
            metrics = accuracy_score(self.y_test, y_pred)
            return metrics

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)

        costs_selected = []
        feature_costs = np.array(self.feature_costs)
        costs_selected = feature_costs[np.argwhere(x==True)]
        cost_sum = sum(costs_selected)

        # By default, function F is always minimize, if we want a mximize,
        # the minus sign (-) must be added
        if cost_sum == 0:
            out["F"] = [0]
        else:
            criterion = scores / cost_sum
            out["F"] = [-1 * criterion]

        # scale_features is a number from 0 to 1
        # if number = 1-scale_features, scale_features = 1 means all features
        # and scale_features = 0 means none feature will be selected
        # Function constraint to select specific numbers of features:
        number = int((1 - self.scale_features) * self.n_max)
        out["G"] = (self.n_max - np.sum(x) - number) ** 2

        # Selected Features in form of list of True and False
        out["SF"] = x
