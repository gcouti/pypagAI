from keras import Model, Input
from keras.layers import Dense, concatenate, LSTM, Reshape
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class SimpleLSTM(KerasModel):
    """
    Use a simple lstm neural network
    """
    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['hidden'] = 32

        return config

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        hidden = model_cfg['hidden']

        story = Input((self._story_maxlen, ), name='story')
        question = Input((self._query_maxlen, ), name='question')

        conc = concatenate([story, question])
        conc = Reshape((1, int(conc.shape[1])))(conc)

        response = LSTM(hidden, dropout=0.2, recurrent_dropout=0.2)(conc)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(lr=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])