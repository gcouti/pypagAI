from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from pypagai.models.base import SciKitModel


class SVMModel(SciKitModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        model = SVC()

        param_grid = {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
            'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000]
        }

        # run grid search
        self._model = GridSearchCV(model, param_grid=param_grid)


class RFModel(SciKitModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        # use a full grid over all parameters
        param_grid = {
            "max_depth": [3, 10, 100, None],
            "max_features": [1, 3, 10],
            "min_samples_split": [2, 3, 10],
            "min_samples_leaf": [1, 3, 10],
            "bootstrap": [True, False],
            "n_estimators": [50, 100, 200, 300],
            "criterion": ["gini", "entropy"]
        }

        model = RandomForestClassifier()

        # run grid search
        self._model = GridSearchCV(model, param_grid=param_grid)
