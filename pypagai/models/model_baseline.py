import random

import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from pypagai.models.base import BaseModel, SciKitModel


class RandomModel(BaseModel):
    """
    Model to choose randomly words on the vocabulary. This model has a purpose of
    """

    def _train_(self, data, report, valid=None):
        return report

    def predict(self, data):
        results = []
        for i in range(len(data.context)):
            v = data.context[i]
            r = random.choice(np.where(v.flat > 0)[0])
            results.append(v.flat[r])
        return results


class TFIDFModel(SciKitModel):

    @staticmethod
    def default_config():
        return SciKitModel.default_config()

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self._model = TfidfTransformer()

    def __count__(self, X):
        count_vector = [1]*X.shape[0]

        for i in range(1, np.max(X)+1):
            count_vector = np.column_stack((count_vector, (X[:] == i).sum(axis=1)))
            #count_vector = np.column_stack((count_vector, (X[:] == i).any(axis=1)))

        return count_vector

    def _train_(self, data, report, valid=None):
        X = np.hstack([data.context, data.query])
        count_vector = self.__count__(X)

        self._model.fit(count_vector)

    def predict(self, data):
        X = np.hstack([data.context, data.query])
        count_vector = self.__count__(X)

        y = self._model.transform(count_vector)

        return np.argmax(y, axis=1).T.tolist()[0]



