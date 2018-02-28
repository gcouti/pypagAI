import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

callback = keras.callbacks.TensorBoard(log_dir='.log/', histogram_freq=0, write_graph=True, write_images=True)


class BaseModel:

    def __init__(self, arg_parser):
        args = arg_parser.add_argument_group('BaseModel')
        args.add_argument('-v', '--verbose', type=bool, default=True)

        args = arg_parser.parse()

        self._model = None
        self._verbose = args.verbose

        self._vocab_size = args.vocab_size
        self._story_maxlen = args.story_maxlen
        self._query_maxlen = args.query_maxlen
        self._sentences_maxlen = args.sentences_maxlen

    def train(self, data, valid=None):
        raise Exception("Not implemented")

    def predict(self, data):
        raise Exception("Not implemented")


class SciKitModel(BaseModel):

    def __init__(self, arg_parser):
        super().__init__(arg_parser)

        self._model_ = None

    def train(self, data, valid=None):
        self._model_.fit(self._network_input_(data), data.answer)

        predictions = self._model_.predict(self._network_input_(data))
        acc = accuracy_score(predictions, data.answer)
        print("Acc %f" % acc)

    def predict(self, data):
        self._model_.predict(self._network_input_(data))

    @staticmethod
    def _network_input_(data):
        return np.hstack([data.context, data.query])


class BaseNeuralNetworkModel(BaseModel):

    def __init__(self, arg_parser):
        super().__init__(arg_parser)

        args = arg_parser.add_argument_group('BaseModel')
        args.add_argument('--epochs', type=int, default=1000)

        args = arg_parser.parse()
        self._epochs = args.epochs

    @staticmethod
    def _network_input_(data):
        """
        Format how the network will receive the inputs
        :param story: Values from history
        :param question: Values from questions

        :return: Expected format from keras models
        """
        return [np.array(data.context), data.query]


class KerasModel(BaseNeuralNetworkModel):

    def train(self, data, valid=None):
        """
        Train models with neural network inputs "story" and "question" with the expected result "answer"
        """
        nn_input = self._network_input_(data)
        nn_valid = self._network_input_(valid)
        for epoch in range(self._epochs):
            self._model.fit(nn_input, data.answer, verbose=self._verbose, callbacks=[callback], validation_data=(nn_valid, valid.answer))

    def predict(self, data):
        nn_input = self._network_input_(data)
        return self._model.predict(nn_input, verbose=False)


class TensorFlowModel(BaseNeuralNetworkModel):

    def train(self, data, valid=None):
        pass

    def predict(self, data):
        pass
