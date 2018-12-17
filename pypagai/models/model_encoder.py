from keras import Input, Model
from keras.layers import Dense, add, Embedding, Dropout, LSTM
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class EncoderModel(KerasModel):

    @staticmethod
    def default_config():
        config = KerasModel.default_config()

        return config

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

    def _create_network_(self):
        story = Input((self._story_maxlen, ), name='story')
        eb_story = Embedding(self._vocab_size, 64)(story)
        eb_story = Dropout(0.3)(eb_story)
        eb_story = LSTM(32)(eb_story)

        question = Input((self._query_maxlen, ), name='question')
        eb_quest = Embedding(self._vocab_size, 64)(story)
        eb_quest = Dropout(0.3)(eb_quest)
        eb_quest = LSTM(32)(eb_quest)

        merged = add([eb_story, eb_quest])

        # add the match matrix with the second input vector sequence
        enc_story = Dense(128, activation='relu', )(merged)
        enc_story = Dense(64, activation='relu')(enc_story)
        enc_story = Dense(32, activation='relu')(enc_story)

        response = Dense(32, activation='relu')(enc_story)
        response = Dense(64, activation='relu')(response)
        response = Dense(128, activation='relu')(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(lr=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
