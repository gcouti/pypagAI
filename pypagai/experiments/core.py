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