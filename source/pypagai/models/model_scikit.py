from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from pypagai.models.base import SciKitModel


class SVMModel(SciKitModel):

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self._model_ = SVC(
            C=1.0,
            kernel=model_cfg['kernel'],
            degree=10,
            gamma='auto',
            coef0=0.0,
            shrinking=True,
            probability=True,
            tol=1e-3,
            cache_size=200,
            verbose=self._verbose,
            max_iter=-1,
            decision_function_shape='ovr',
            random_state=None
        )


class RFModel(SciKitModel):

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self._model_ = RandomForestClassifier(
            n_estimators=model_cfg['estimators'],
            max_features="auto",
            min_impurity_decrease=0.,
            bootstrap=True,
            n_jobs=-1,
            verbose=self._verbose,
            warm_start=True,
        )
