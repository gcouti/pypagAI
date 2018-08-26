import keras
import tensorflow as tf

import math
import logging
import numpy as np
import pandas as pd
from keras.callbacks import Callback, EarlyStopping

from sacred import Ingredient
from sklearn import preprocessing

tb_callback = keras.callbacks.TensorBoard(log_dir='.log/', histogram_freq=0, write_graph=True, write_images=True)

LOG = logging.getLogger('pypagai-logger')
model_ingredient = Ingredient('model_default_cfg')


@model_ingredient.config
def default_model_configuration():
    """
    Model configuration
    """
    model = 'pypagai.models.model_lstm.SimpleLSTM'  # Path to the ML model
    verbose = False  # True to print info about train


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
            'maximum_acc': 1,
        }

    def print(self):
        LOG.info(self._model)

    def train(self, data, valid=None):
        raise Exception("Not implemented")

    def predict(self, data):
        raise Exception("Not implemented")


class SciKitModel(BaseModel):
    @staticmethod
    def default_config():
        config = BaseModel.default_config()
        return config

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self._model = model_cfg['model']
        self._le = preprocessing.LabelBinarizer()

    def train(self, data, valid=None):
        self._le.fit(np.array([data.answer]).T)
        trans = self._network_input_(self._le.fit_transform, data)
        # answer = self._le.transform(np.array([data.answer]).T)

        self._model.fit(trans, data.answer)

    def predict(self, data):
        trans = self._network_input_(self._le.transform, data)
        pred = self._model.predict(trans)
        return pred

    @staticmethod
    def _network_input_(func, data):
        trans = np.hstack([data.context, data.query])
        shape = trans.shape
        trans = np.reshape(trans, (1, shape[0] * shape[1]))[0]
        trans = func(trans)
        trans = np.reshape(trans, (shape[0], shape[1] * trans.shape[1]))
        return trans


class BaseNeuralNetworkModel(BaseModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self._epochs = model_cfg['epochs']
        self._patience = model_cfg['patience']
        self._keras_log = model_cfg['keras_log']
        self._log_every = model_cfg['log_every']
        self._batch_size = model_cfg['batch_size']

    @staticmethod
    def default_config():
        config = BaseModel.default_config()

        config['epochs'] = 1000
        config['patience'] = 25
        config['log_every'] = 50
        config['keras_log'] = False
        config['batch_size'] = 32

        return config

    def _network_input_(self, data):
        """
        Format how the network will receive the inputs

        :return: Expected format from keras models
        """
        if self._sentences_maxlen:
            return [data.context, data.query, data.labels.T]
        else:
            return [data.context, data.query]


class TestCallback(Callback):
    def __init__(self, test_data, maximum=1.0, log_every=50, verbose=False):
        super().__init__()
        self._test_data = test_data
        self._maximum = maximum
        self._verbose = verbose
        self._log_every = log_every
        self._stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self._log_every != 0 or epoch == 0:
            return

        x, y = self._test_data
        loss, acc = self.model.evaluate(x, y, verbose=self._verbose)

        if self._verbose:
            LOG.info('[TEST SET] epoch: {} - loss: {}, acc: {}\n'.format(epoch, loss, acc))

        if acc > self._maximum:
            self._stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self._stopped_epoch > 0 and self._verbose:
            LOG.debug('Epoch %05d: early stopping' % (self._stopped_epoch + 1))


class KerasModel(BaseNeuralNetworkModel):
    """
    https://github.com/xkortex/Siraj_Chatbot_Challenge

    https://ethancaballero.pythonanywhere.com/
    """

    def train(self, data, valid=None):
        """
        Train models with neural network inputs "story" and "question" with the expected result "answer"
        """
        nn_input = self._network_input_(data)
        nn_input_test = self._network_input_(valid)

        cb = [EarlyStopping(patience=self._patience)]
        cb += [TestCallback((nn_input_test, valid.answer),
                            self._maximum_acc,
                            log_every=self._log_every,
                            verbose=self._verbose)]
        cb += [tb_callback] if self._keras_log else []

        self._model.fit(nn_input, data.answer,
                        # validation_split=0.2,
                        callbacks=cb,
                        verbose=self._verbose,
                        batch_size=self._batch_size,
                        epochs=self._epochs)

        return pd.DataFrame(self._model.history.history)

    def predict(self, data):
        nn_input = self._network_input_(data)
        pred = self._model.predict(nn_input, verbose=self._verbose)
        return np.argsort(pred)[:, ::-1][:, 0]


class TensorFlowModel(BaseNeuralNetworkModel):

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self._story = None
        self._question = None
        self._answer = None

        self._train_op = None
        self._loss_op = None
        self._accuracy = None

    def train(self, data, valid=None):

        # Run the initializer
        self._sess = tf.Session()
        self._sess.run(tf.local_variables_initializer())
        self._sess.run(tf.global_variables_initializer())

        nn_input = self._network_input_(data)
        nn_input_test = self._network_input_(valid)

        for step in range(1, self._epochs + 1):

            # Run optimization op (backprop)
            size = len(nn_input[0])
            for batch in range(math.ceil(size / self._batch_size)):
                self._sess.run(self._train_op, feed_dict={
                    self._story: nn_input[0][batch*self._batch_size:(batch+1)*self._batch_size],
                    self._question: nn_input[1][batch*self._batch_size:(batch+1)*self._batch_size],
                    self._answer: np.array([data.answer[batch*self._batch_size:(batch+1)*self._batch_size]]).T})

            if step % self._log_every == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = self._sess.run([self._loss_op, self._accuracy],
                                           feed_dict={
                                               self._story: nn_input[0],
                                               self._question: nn_input[1],
                                               self._answer: np.array([data.answer]).T
                                           })
                print("Step " + str(step) +
                      ", Training Accuracy={:.3f}".format(acc) +
                      ", Minibatch Loss={:.4f}".format(loss))

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:",
              self._sess.run(self._accuracy, feed_dict={
                  self._story: nn_input_test[0],
                  self._question: nn_input_test[1],
                  self._answer: np.array([valid.answer]).T
              }))

    def position_encoding(self, sentence_size, embedding_size):
        """
        Position Encoding described in section 4.1 [1]
        """
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size + 1
        le = embedding_size + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        # Make position encoding of time words identity to avoid modifying them
        encoding[:, -1] = 1.0
        return np.transpose(encoding)

    def predict(self, data):
        nn_input = self._network_input_(data)

        feed_dict = {
            self._story: nn_input[0],
            self._question: nn_input[1],
        }

        return np.argmax(self._model.eval(feed_dict=feed_dict, session=self._sess), 1)
