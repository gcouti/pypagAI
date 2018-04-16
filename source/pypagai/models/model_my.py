from __future__ import print_function

from keras.layers import Input, Embedding, LSTM, Reshape, concatenate, add, regularizers, Bidirectional, Permute, \
    Conv2D, Conv1D, PReLU, MaxPool2D, dot, Activation, SimpleRNN
from keras.layers.core import Dense, Dropout, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class ConvRN(KerasModel):
    """
    """

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['embed-size'] = 64
        config['lstm-units'] = 32

        return config

    def __init__(self, cfg):
        super().__init__(cfg)

        EMBED_SIZE = cfg['embed-size']
        LSTM_UNITS = cfg['lstm-units']

        story = Input((self._story_maxlen,), name='story')
        story_embedded = Embedding(self._vocab_size, EMBED_SIZE)(story)
        story_embedded = Dropout(0.3)(story_embedded)

        question = Input((self._query_maxlen,), name='question')
        question_embedded = Embedding(self._vocab_size, EMBED_SIZE)(question)
        question_embedded = Dropout(0.3)(question_embedded)

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

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class ReluN2NMemory(KerasModel):
    """
    Add two layers inspired n the
    """

    def __init__(self, model_cfg):

        super().__init__(model_cfg)

        # placeholders
        input_sequence = Input((self._story_maxlen, ))
        question = Input((self._query_maxlen,))

        answer = self.create_network(input_sequence, question, model_cfg)

        # build the final model
        self._model = Model([input_sequence, question], answer)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['dropout'] = 0.3
        config['activation'] = 'softmax'
        config['samples'] = 32
        config['embedding'] = 64
        config['rnn-layers'] = 0

        return config

    def create_network(self, input_sequence, question, cfg):

        drop_out = cfg['dropout']
        samples = cfg['samples']
        embedding = cfg['embedding']

        # encoders
        # embed the input sequence into a sequence of vectors

        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=self._vocab_size, output_dim=embedding))
        input_encoder_m.add(Dropout(drop_out))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=self._vocab_size, output_dim=self._query_maxlen))
        input_encoder_c.add(Dropout(drop_out))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=self._vocab_size, output_dim=embedding, input_length=self._query_maxlen))
        question_encoder.add(Dropout(drop_out))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('relu')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        for rnn_layers in range(cfg['rnn-layers']):
            answer = SimpleRNN(samples, return_sequences=True)(answer)
        answer = SimpleRNN(samples)(answer)

        response = Dense(256, activation='relu')(answer)
        response = Dropout(0.3)(response)
        response = Dense(512, activation='relu')(response)
        response = Dropout(0.3)(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        return response
