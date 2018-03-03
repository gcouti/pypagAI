from keras import Model, Input
from keras.layers import Dense, concatenate, LSTM, Embedding
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class EmbedLSTM(KerasModel):

    """
    Use a simple lstm neural network
    """

    def __init__(self, model_cfg):

        super().__init__(model_cfg)

        # args = arg_parser.add_argument_group(__name__)
        # args.add_argument('--hidden', type=int, default=32)
        # args = arg_parser.parse()

        hidden = model_cfg['hidden'] if 'hidden' in model_cfg else 32

        story = Input((self._story_maxlen, ), name='story')
        question = Input((self._query_maxlen, ), name='question')

        conc = concatenate([story, question])
        conc = Embedding(self._vocab_size, 200)(conc)

        response = LSTM(hidden, dropout=0.2, recurrent_dropout=0.2)(conc)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(lr=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
