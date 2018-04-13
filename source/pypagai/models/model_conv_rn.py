from __future__ import print_function

from keras.layers import Input, Embedding, LSTM, Reshape, concatenate, add, regularizers, Bidirectional, Permute, \
    Conv2D, Conv1D, PReLU, MaxPool2D
from keras.layers.core import Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class ConvRN(KerasModel):
    """
    """

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['embed-size'] = 128
        config['lstm-units'] = 32

        return config

    def __init__(self, cfg):
        super().__init__(cfg)

        EMBED_SIZE = cfg['embed-size']
        LSTM_UNITS = cfg['lstm-units']

        story = Input((self._story_maxlen,), name='story')
        story_embedded = Embedding(self._vocab_size, EMBED_SIZE)(story)

        question = Input((self._query_maxlen,), name='question')
        question_embedded = Embedding(self._vocab_size, EMBED_SIZE)(question)

        reg = regularizers.l2(1e-4)
        lstm = LSTM(LSTM_UNITS,
                    recurrent_regularizer=reg,
                    recurrent_dropout=0.25,
                    implementation=2,
                    return_sequences=True)
        story_encoder = Bidirectional(lstm)(story_embedded)

        conv_layers = ['l1']
        story_conv = story_encoder
        for layer in conv_layers:
            story_conv = Conv1D(32, (3,), strides=1, padding='valid', name='convolution_'+layer)(story_conv)
            story_conv = PReLU(shared_axes=[1, 2], name='prelu_'+layer)(story_conv)
            # story_conv = MaxPool2D(pool_size=3, strides=1, padding='same', name='max_'+layer)(story_conv)

        question_encoder = Bidirectional(lstm)(question_embedded)

        objects = Permute((2, 2))(story_conv)
        # for k in range(self._sentences_maxlen):
        #     fact_object = Lambda(lambda x: x[:, k, :])(story_encoder)
        #     objects.append(fact_object)

        # relations = []
        # for fact_object_1 in objects:
        #     for fact_object_2 in objects:
        #         r = concatenate([fact_object_1, fact_object_2, question_encoder])
        #         relations.append(r)

        response = Dense(256, activation='relu')(objects)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)

        combined_relation = add(response)

        response = Dense(256, activation='relu')(combined_relation)
        response = Dense(512, activation='relu')(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question, labels], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
