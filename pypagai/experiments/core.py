# aux pkg imports
import pandas as pd

# scikit imports
from pypagai.experiments.evaluation import make_result_frame
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
def validate_estimator(estimator, data):
    """ Generate cross-validated estimates for each input data point.

    :param estimator: estimator object implementing ‘fit’ and ‘predict’
        The object to use to fit the data.

    """
    #X_train, X_test, y_train, y_test = train_test_split(train., y, )

    dfs = []

    cv = None

    if cv is None:
        cv = KFold()

    fold = size = repeat = 0
    for train_index, test_index in cv.split(data.context):
        train = data.filter(train_index)
        test = data.filter(test_index)

        #TODO: clear model
        estimator.train(train)
        y_pred = estimator.predict(test)
        result = make_result_frame(test.answer, y_pred, index=test_index, repeat=repeat, fold=fold)

        dfs.append(result)

        fold += 1
        size += test.answer.shape[0]

        # if size == X.shape[0]:
        #     fold = size = 0
        #     repeat += 1

    return pd.concat(dfs)
