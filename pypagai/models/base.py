import keras
import tensorflow as tf

import math
import logging
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

from sacred import Ingredient
from sklearn import preprocessing
from sklearn.cross_validation import KFold

from pypagai.util.class_loader import ClassLoader

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

        # Number of kfolds
        self._splits = model_cfg['kfold_splits'] if 'kfold_splits' in model_cfg else 2

        # Verbose mode
        self._verbose = model_cfg['verbose'] if 'verbose' in model_cfg else False

        # Skip experiment with kfolds. When it is False skip kfold validation and go thought train all dataset directly
        self._experiment = model_cfg['experiment'] if 'experiment' in model_cfg else False

        self._maximum_acc = model_cfg['maximum_acc'] if 'maximum_acc' in model_cfg else 1
        self._vocab_size = model_cfg['vocab_size']
        self._story_maxlen = model_cfg['story_maxlen']
        self._query_maxlen = model_cfg['query_maxlen']
        self._sentences_maxlen = model_cfg['sentences_maxlen']

    @staticmethod
    def default_config():
        return {
            'maximum_acc': 1,
        }

    def name(self):
        return self.__class__.__name__

    def print(self):
        LOG.info(self.name())

    def train(self, data):

        fold = 0
        final_report = pd.DataFrame()
        fold_list = KFold(len(data.answer), self._splits) if self._experiment else [(range(len(data.answer)), [])]
        for train_index, valid_index in fold_list:
            report = pd.DataFrame()
            self._train_(data[train_index], report, data[valid_index])

            fold += 1
            report['fold'] = fold
            final_report = pd.concat([final_report, report])

        if self._experiment:
            self._train_(data, None)

        return final_report

    def _train_(self, data, report, valid=None):
        raise Exception("Not implemented")

    def predict(self, data):
        raise Exception("Not implemented")


class SciKitModel(BaseModel):
    @staticmethod
    def default_config():
        config = BaseModel.default_config()

        config['classifier'] = 'sklearn.ensemble.RandomForestClassifier'

        return config

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        cl = ClassLoader()
        klass = cl.load(model_cfg['classifier'])
        self._model = klass()
        self._le = preprocessing.LabelBinarizer()

    def _train_(self, data, report, valid=None):
        self._le.fit(np.array([data.answer]).T)
        trans = self.__input__(self._le.fit_transform, data)
        # answer = self._le.transform(np.array([data.answer]).T)

        self._model.fit(trans, data.answer)

    def predict(self, data):
        trans = self.__input__(self._le.transform, data)
        pred = self._model.predict(trans)
        return pred

    @staticmethod
    def __input__(func, data):
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
            return [data.context, data.query, data.labels]
        else:
            return [data.context, data.query]


class KerasModel(BaseNeuralNetworkModel):
    """
    https://github.com/xkortex/Siraj_Chatbot_Challenge

    https://ethancaballero.pythonanywhere.com/
    """

    def __build_model__(self):
        self._create_network_()
        return self._model

    def _create_network_(self):
        raise Exception("Not implemented")

    def _train_(self, data, report, valid=None):
        """
        Train models with neural network inputs "story" and "question" with the expected result "answer"
        """
        nn_input = self._network_input_(data)
        nn_input_valid = self._network_input_(valid) if valid and len(valid.answer) > 0 else None

        cb = [EarlyStopping(patience=self._patience)]
        cb += [tb_callback] if self._keras_log else []

        self._model = self.__build_model__()
        self._model.fit(nn_input, data.answer,
                        callbacks=cb,
                        epochs=self._epochs,
                        verbose=2 if self._verbose else 0,
                        validation_data=(nn_input_valid, valid.answer) if valid and len(valid.answer) > 0 else None)

        report = pd.DataFrame(self._model.history.history)
        report = report.reset_index()
        report = report.rename(columns={'index': 'epoch'})

        return report

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

    def _train_(self, data, report, valid=None):

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
                    self._story: nn_input[0][batch * self._batch_size:(batch + 1) * self._batch_size],
                    self._question: nn_input[1][batch * self._batch_size:(batch + 1) * self._batch_size],
                    self._answer: np.array([data.answer[batch * self._batch_size:(batch + 1) * self._batch_size]]).T})

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

        # def positional_encoding(self):
        #         D, M, N = self.params.embed_size, self.params.max_sent_size, self.params.batch_size
        #     encoding = np.zeros([M, D])
        #     for j in range(M):
        #         for d in range(D):
        #             encoding[j, d] = (1 - float(j)/M) - (float(d)/D)*(1 - 2.0*j/M)

        # return encoding

    def predict(self, data):
        nn_input = self._network_input_(data)

        feed_dict = {
            self._story: nn_input[0],
            self._question: nn_input[1],
        }

        return np.argmax(self._model.eval(feed_dict=feed_dict, session=self._sess), 1)
