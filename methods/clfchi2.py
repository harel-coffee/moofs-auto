import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin, BaseEstimator

from chi_square import ChiSquare


class ClfChi2(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.estimator = None
        self.selected_features = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        features = range(X.shape[1])
        print(features)
        X_df.columns = features
        X_df['class'] = y
        # y_df = pd.DataFrame(y)
        chi_sq = ChiSquare(X_df)
        for col in features:
            chi_sq.TestIndependence(colX=col, colY='class')
        self.selected_features = chi_sq.features
        # BŁĄD! że wybiera 0 features, daj jakis warunek

        print("Selected features for each fold: {}".format(np.sum(self.selected_features)))
        print(self.selected_features)

        self.estimator = self.base_estimator.fit(X[:, self.selected_features], y)
        return self

    def predict(self, X):
        return self.estimator.predict(X[:, self.selected_features])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[:, self.selected_features])
