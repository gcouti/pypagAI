from keras import Input, Model
from keras.backend import concatenate
from keras.layers import Dense

from agents.base import BaseKerasAgent


class EncoderAgent(BaseKerasAgent):

    @staticmethod
    def add_cmdline_args(parser):
        BaseKerasAgent.add_cmdline_args(parser)

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'EncoderAgent'
        self.opt = opt

        # placeholders
        input_sequence = Input(shape=(self._text_max_size, ))
        input_question = Input(shape=(self._text_max_size, ))

        input_concatenated = concatenate([input_sequence, input_question])

        # # add the match matrix with the second input vector sequence
        response = Dense(16, activation='relu')(input_concatenated)
        response = Dense(8, activation='relu')(response)
        response = Dense(4, activation='relu')(response)
        response = Dense(8, activation='relu')(response)
        response = Dense(16, activation='relu')(response)
        pred = Dense(len(self._dictionary), activation='softmax')(response)

        self._model = Model(inputs=[input_sequence, input_question], outputs=pred)
        self._model.compile(optimizer="adadelta", loss='categorical_crossentropy', metrics=['accuracy'])

