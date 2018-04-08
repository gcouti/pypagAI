import keras
import logging
import numpy as np

from sacred import Ingredient
from sklearn.metrics import accuracy_score, f1_score

callback = keras.callbacks.TensorBoard(log_dir='.log/', histogram_freq=0, write_graph=True, write_images=True)

LOG = logging.getLogger('pypagai-logger')
model_ingredient = Ingredient('model_default_cfg')


@model_ingredient.config
def default_model_configuration():
    """
    Model configuration
    """
    model = 'pypagai.models.model_lstm.SimpleLSTM'    # Path to the ML model
    verbose = False                                    # True to print info about train


class BaseModel:

    def __init__(self, model_cfg):
        self._model = None
        self._verbose = model_cfg['verbose'] if 'verbose' in model_cfg else False

        self._vocab_size = model_cfg['vocab_size']
        self._story_maxlen = model_cfg['story_maxlen']
        self._query_maxlen = model_cfg['query_maxlen']
        self._sentences_maxlen = model_cfg['sentences_maxlen']
        self._maximum_acc = model_cfg['maximum_acc'] if 'maximum_acc' in model_cfg else 1

    @staticmethod
    def default_config():
        return {
            'maximum_acc': .95,
        }

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
        self._log_every_ = model_cfg['log_every']
        self._epochs = model_cfg['epochs']
        self._keras_log = model_cfg['keras_log']
        self._batch_size = model_cfg['batch_size']

    @staticmethod
    def default_config():
        config = BaseModel.default_config()
        config['log_every'] = 10
        config['epochs'] = 1000
        config['keras_log'] = False
        config['batch_size'] = 512

        return config

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
    """
    https://github.com/xkortex/Siraj_Chatbot_Challenge
    https://github.com/erilyth/DeepLearning-Challenges/blob/master/Text_Based_Chatbot/memorynetwork.py
    https://github.com/EibrielInv/ice-cream-truck/blob/master/chatbot.py
    https://github.com/llSourcell/How_to_make_a_chatbot/blob/master/memorynetwork.py

    https://ethancaballero.pythonanywhere.com/
    """

    def train(self, data, valid=None):
        """
        Train models with neural network inputs "story" and "question" with the expected result "answer"
        """
        nn_input = self._network_input_(data)
        for epoch in range(self._epochs):
            # , validation_data=(nn_valid, valid.answer)
            if self._verbose:
                LOG.debug("Epoch %i/%i" % (epoch+1, self._epochs))

            call = [callback] if self._keras_log else []
            self._model.fit(nn_input, data.answer, callbacks=call, verbose=self._verbose, batch_size=self._batch_size)

            # TODO: check if it is possible to done that with capture from sacred
            if epoch % self._log_every_ == 0:
                acc, f1 = self.valid(valid)
                LOG.info("Epoch %i/%i, acc: %f f1: %f" % (epoch+1, self._epochs, acc, f1))
                if acc > self._maximum_acc:
                    LOG.info("Complete before epochs finished %f", acc)
                    break

    def valid(self, data):
        nn_input = self._network_input_(data)
        predictions = self._model.predict(nn_input)
        return self.metrics(predictions, data.answer)

    def predict(self, data):
        nn_input = self._network_input_(data)
        return self._model.predict(nn_input, verbose=self._verbose)


class TensorFlowModel(BaseNeuralNetworkModel):

    def train(self, data, valid=None):
        pass

    def predict(self, data):
        pass
