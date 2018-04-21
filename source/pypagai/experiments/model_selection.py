# aux pkg imports
import pandas as pd

# scikit imports
from experiments.evaluation import make_result_frame
from sklearn.cross_validation import train_test_split
from sklearn.utils import safe_indexing
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, TimeSeriesSplit

# sacred imports
from sacred import Ingredient

cv_ingredient = Ingredient('cv')

cvs = {
    'kf': KFold,
    'skf': StratifiedKFold,
    'rskf': RepeatedStratifiedKFold,
    'ts': TimeSeriesSplit
}


def get_params(cv):
    cv = cvs.get(cv, None)

    if cv is None:
        return {}

#    return get_default_vars(cv.__init__)
    return {}


@cv_ingredient.config
def tcfg():
    # data splitter name (options: kf, skf, rskf, ts, None).
    name = None

    # data splitter parameters
    params = get_params(name)


@cv_ingredient.capture
def get_cv(name, params):
    cv = cvs.get(name, None)

    if cv is None:
        return None

    return cv(**params)


@cv_ingredient.capture
def cv_predict(estimator, X):
    """ Generate cross-validated estimates for each input data point.

    :param estimator: estimator object implementing ‘fit’ and ‘predict’
        The object to use to fit the data.
    :param X: array-like.
        The data to fit. Can be, for example a list, or an array at least 2d.
    :param y: array-like.
        The target variable to try to predict in the case of supervised learning.
    :param name: str
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
        - kf, KFold cross-validation,
        - skf, Stratified KFold cross-validation,
        - rskf, Repeated Stratified KFold cross-validation,
    :param params: dict,
        Cross-validation strategy parameters.
    """
    #X_train, X_test, y_train, y_test = train_test_split(train., y, )

    dfs = []

    cv = None
    #cv = get_cv()

    if cv is None:
        cv = KFold()

    fold = size = repeat = 0
    for train_index, test_index in cv.split(X, y):

        X_train, X_test = safe_indexing(X, train_index), safe_indexing(X, test_index)
        y_train, y_test = safe_indexing(y, train_index), safe_indexing(y, test_index)

        y_pred = estimator.fit(X_train, y_train).predict(X_test)
        result = make_result_frame(y_test, y_pred, index=test_index, repeat=repeat, fold=fold)

        dfs.append(result)

        fold += 1
        size += X_test.shape[0]

        if size == X.shape[0]:
            fold = size = 0
            repeat += 1

    return pd.concat(dfs)
