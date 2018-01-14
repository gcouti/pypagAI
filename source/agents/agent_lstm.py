import numpy as np

from keras import Model, Input, Sequential
from keras.layers import Dense, add, concatenate, LSTM, Reshape
from keras.optimizers import Adam

from agents.base import Networks, BaseKerasAgent


class SimpleLSTM(Networks):
    """
    Use a simple seq2se neural network
    """

    def __init__(self, vocab_size, story_maxlen):

        super().__init__()
        self._vocab_size = vocab_size
        self._story_maxlen = story_maxlen
        self._query_maxlen = story_maxlen

        layers = 16
        hidden = 128
        story = Input((self._story_maxlen,self._vocab_size), name='story')
        question = Input((self._query_maxlen,self._vocab_size), name='question')

        conc = concatenate([story, question])

        response = LSTM(hidden, return_sequences=True)(conc)
        response = LSTM(hidden)(response)

        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)

        optimizer = Adam(lr=2e-4)
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


class LSTMAgent(BaseKerasAgent):

    @staticmethod
    def add_cmdline_args(parser):
        BaseKerasAgent.add_cmdline_args(parser)

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'LSTMAgent'
        self.opt = opt

        self.__create_network__(opt)

    def __create_network__(self, opt):
        """
        Create neural network

        :param opt: ParlAI opt params
        """

        self._statement_size = 8
        self._vocab_size = len(self._dictionary)
        self._story_maxlen = 15 * 2

        self._model = SimpleLSTM(self._vocab_size, self._story_maxlen)