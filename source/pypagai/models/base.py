import keras
import logging
import numpy as np

from pypagai import settings
from sklearn.metrics import accuracy_score

callback = keras.callbacks.TensorBoard(log_dir='.log/', histogram_freq=0, write_graph=True, write_images=True)

logging.basicConfig(level=settings.LOG_LEVEL)
LOG = logging.getLogger(__name__)


class BaseModel:

    def __init__(self, model_cfg):
        # args = arg_parser.add_argument_group('BaseModel')
        # args.add_argument('-v', '--verbose', type=bool, default=True)
        #
        # args = arg_parser.parse()

        self._model = None
        self._verbose = model_cfg['verbose'] if 'verbose' in model_cfg else False

        self._vocab_size = model_cfg['vocab_size']
        self._story_maxlen = model_cfg['story_maxlen']
        self._query_maxlen = model_cfg['query_maxlen']
        self._sentences_maxlen = model_cfg['sentences_maxlen']

    def train(self, data, valid=None):
        raise Exception("Not implemented")

    def predict(self, data):
        raise Exception("Not implemented")


class SciKitModel(BaseModel):

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self._model_ = model_cfg['model']

    def train(self, data, valid=None):
        self._model_.fit(self._network_input_(data), data.answer)

        predictions = self._model_.predict(self._network_input_(data))
        acc = accuracy_score(predictions, data.answer)
        print("Acc %f" % acc)

    def predict(self, data):
        self._model_.predict(self._network_input_(data))

    def _network_input_(self, data):
        return np.hstack([data.context, data.query])


class BaseNeuralNetworkModel(BaseModel):

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self._epochs = model_cfg['epochs'] if 'epochs' in model_cfg else 100

    def _network_input_(self, data):
        """
        Format how the network will receive the inputs

        :return: Expected format from keras models
        """
        if self._sentences_maxlen:
            return [data.context, data.query, data.labels]
        else:
            return [data.context, data.query]


class KerasModel(BaseNeuralNetworkModel):

    def train(self, data, valid=None):
        """
        Train models with neural network inputs "story" and "question" with the expected result "answer"
        """
        nn_input = self._network_input_(data)
        nn_valid = self._network_input_(valid)
        for epoch in range(self._epochs):
            LOG.debug("epoch %i/%i" % (epoch+1, self._epochs))
            self._model.fit(nn_input, data.answer, verbose=self._verbose, callbacks=[callback],
                            validation_data=(nn_valid, valid.answer))

    def predict(self, data):
        nn_input = self._network_input_(data)
        return self._model.predict(nn_input, verbose=False)


class TensorFlowModel(BaseNeuralNetworkModel):

    def train(self, data, valid=None):
        pass

    def predict(self, data):
        pass
