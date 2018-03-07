import keras
import logging
import numpy as np
import sys

from pypagai import settings
from sklearn.metrics import accuracy_score, f1_score

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

    def valid(self, data):
        raise Exception("Not implemented")

    def metrics(self, pred, true):
        """
        Print metrics based on predicted answers and true answers

        :param pred: Predicted
        :param true: True answers
        """
        acc = accuracy_score(np.argsort(pred)[:, ::-1][:, 0], true)
        f1 = f1_score(np.argsort(pred)[:, ::-1][:, 0], true, average="macro")
        LOG.info("Accuracy: %f F1: %f", acc, f1)
        return acc, f1


class SciKitModel(BaseModel):

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

    def train(self, data, valid=None):
        self._model.fit(self._network_input_(data), data.answer)

    def valid(self, data):
        predictions = self._model.predict(self._network_input_(data))
        self.metrics(predictions, data.answer)

    def predict(self, data):
        self._model.predict(self._network_input_(data))

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
        for epoch in range(self._epochs):
            # , validation_data=(nn_valid, valid.answer)
            LOG.debug("Epoch %i/%i" % (epoch+1, self._epochs))
            self._model.fit(nn_input, data.answer, callbacks=[callback], verbose=self._verbose)

            if epoch % 10 == 0:
                acc, f1 = self.valid(valid)
                LOG.info("Epoch %i/%i, acc: %f f1: %f" % (epoch+1, self._epochs, acc, f1))
                if acc > 0.95:
                    LOG.info("Complete before epochs finished %f", acc)
                    break

    def valid(self, data):
        nn_input = self._network_input_(data)
        predictions = self._model.predict(nn_input)
        return self.metrics(predictions, data.answer)

    def predict(self, data):
        nn_input = self._network_input_(data)
        return self._model.predict(nn_input, verbose=False)


class TensorFlowModel(BaseNeuralNetworkModel):

    def train(self, data, valid=None):
        pass

    def predict(self, data):
        pass
