import numpy as np

from keras import Model, Input, Sequential
from keras.layers import Dense, add, concatenate, LSTM, Reshape, Embedding, K
from keras.optimizers import Adam
from keras.utils import to_categorical

from agents.base import Networks, BaseKerasAgent


class SimpleLSTM(Networks):
    """
    Use a simple lstm neural network
    """

    def __init__(self, opt, vocab_size, story_maxlen, query_maxlen):

        super().__init__()

        hidden = opt['hidden']

        self._vocab_size = vocab_size
        self._story_maxlen = story_maxlen
        self._query_maxlen = query_maxlen

        story = Input((self._story_maxlen, ), name='story')
        question = Input((self._query_maxlen, ), name='question')

        conc = concatenate([story, question])
        conc = Reshape((1, int(conc.shape[1])))(conc)

        response = LSTM(hidden, dropout=0.2, recurrent_dropout=0.2)(conc)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(lr=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class LSTMAgent(BaseKerasAgent):

    @staticmethod
    def add_cmdline_args(parser):
        BaseKerasAgent.add_cmdline_args(parser)

        agent = parser.add_argument_group('LSTM Arguments')

        message = 'Number of hidden layers'
        agent.add_argument('-hd', '--hidden', type=int, default=128, help=message)

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'LSTMAgent'
        self.opt = opt

        self._vocab_size = len(self._dictionary)
        self._story_maxlen = opt['story_length']
        self._query_maxlen = opt['query_length']

        self._create_network_(opt)

    def _create_network_(self, opt):
        """
        Create neural network

        :param opt: ParlAI opt params
        """
        self._model = SimpleLSTM(opt, self._vocab_size, self._story_maxlen, self._query_maxlen)
