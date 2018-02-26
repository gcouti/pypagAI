from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from pypagai.models.base import SciKitModel


class SVMModel(SciKitModel):

    ALIAS = "svm"

    def __init__(self, arg_parser, _):
        super().__init__(arg_parser)

        args = arg_parser.add_argument_group(__name__)
        args.add_argument('--kernel', type=str, default='rbf')
        args = arg_parser.parse()

        self._model_ = SVC(
            C=1.0,
            kernel=args.kernel,
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

    ALIAS = "rf"

    def __init__(self, arg_parser, _):
        super().__init__(arg_parser)

        args = arg_parser.add_argument_group(__name__)
        args.add_argument('--estimators', type=int, default=500)
        args = arg_parser.parse()

        self._model_ = RandomForestClassifier(
            n_estimators=args.estimators,
            max_features="auto",
            min_impurity_decrease=0.,
            bootstrap=True,
            n_jobs=-1,
            verbose=self._verbose,
            warm_start=True,
        )