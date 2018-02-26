from keras import Input, Model
from keras.backend import concatenate
from keras.layers import Dense, Reshape, add
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class EncoderModel(KerasModel):

    ALIAS = "encoder"

    def __init__(self, arg_parser, shared=None):
        super().__init__(arg_parser)
        # args = arg_parser.add_argument_group(__name__)
        # args = arg_parser.parse()

        # placeholders
        story = Input((self._story_maxlen, ), name='story')
        question = Input((self._query_maxlen, ), name='question')

        # add the match matrix with the second input vector sequence
        enc_story = Dense(128, activation='relu')(story)
        enc_story = Dense(64, activation='relu')(enc_story)
        enc_story = Dense(32, activation='relu')(enc_story)

        enc_quest = Dense(128, activation='relu')(question)
        enc_quest = Dense(64, activation='relu')(enc_quest)
        enc_quest = Dense(32, activation='relu')(enc_quest)

        merged = add([enc_quest, enc_story])

        response = Dense(64, activation='relu')(merged)
        response = Dense(128, activation='relu')(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(lr=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])