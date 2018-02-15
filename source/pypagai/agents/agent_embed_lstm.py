import numpy as np

from keras import Model, Input, Sequential
from keras.layers import Dense, add, concatenate, LSTM, Reshape, Embedding
from keras.optimizers import Adam

from pypagai.agents.agent_lstm import LSTMAgent
from pypagai.agents.base import Networks, BaseKerasAgent


class EmbedLSTM(Networks):
    """
    Use a simple lstm neural network
    """

    def __init__(self, vocab_size, story_maxlen, query_maxlen, hidden=32):

        super().__init__()

        self._vocab_size = vocab_size
        self._story_maxlen = story_maxlen
        self._query_maxlen = query_maxlen

        story = Input((self._story_maxlen, ), name='story')
        question = Input((self._query_maxlen, ), name='question')

        conc = concatenate([story, question])
        conc = Embedding(self._vocab_size, 200)(conc)

        response = LSTM(hidden, dropout=0.2, recurrent_dropout=0.2)(conc)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(lr=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class EmbedLSTMAgent(LSTMAgent):

    def _create_network_(self, opt):
        """
        Create neural network

        :param opt: ParlAI opt params
        """
        self._model = EmbedLSTM(opt, self._vocab_size, self._story_maxlen, self._query_maxlen)
