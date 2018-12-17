from __future__ import print_function

import itertools
from keras.layers import Input, Embedding, LSTM, Reshape, concatenate, regularizers, Bidirectional, Conv1D, \
    MaxPooling1D, Permute, Conv2D, MaxPooling2D, SimpleRNN
from keras.layers.core import Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.backend import stack

from pypagai.models.base import KerasModel


class RN(KerasModel):
    """
    Implementation of Relational Neural Network:

    https://arxiv.org/pdf/1706.01427.pdf

    For the bAbI suite of tasks the natural language inputs must be transformed into a set of objects. This is a
    distinctly different requirement from visual QA, where objects were defined as spatially distinct regions in
    convolved feature maps. So, we first identified up to 20 sentences in the support set that were immediately
    prior to the probe question. Then, we tagged these sentences with labels indicating their relative position in
    the support set, and processed each sentence word-by-word with an LSTM (with the same LSTM acting on each
    sentence independently). We note that this setup invokes minimal prior knowledge, in that we delineate objects
    as sentences, whereas previous bAbI models processed all word tokens from all support sentences sequentially.
    It's unclear how much of an advantage this prior knowledge provides, since period punctuation also unambiguously
    delineates sentences for the token-by-token processing models.  The final state of the sentence-processing-LSTM
    is considered to be an object.  Similar to visual QA, a separate LSTM produced a question embedding, which was
    appended to each object pair as input to the RN. Our model was trained on the joint version of bAbI (all 20
    tasks simultaneously), using the full dataset of 10K examples per task

    For the bAbI task, each of the 20 sentences in the support set was processed through a 32 unit LSTM to produce
    an object.  For the RN,g was a four-layer MLP consisting of 256 units per layer.  For f , we used a three-layer
    MLP consisting of 256, 512, and 159 units, where the final layer was a linear layer that produced logits for a
    softmax over the answer vocabulary.  A separate LSTM with 32 units was used to process the question.
    The softmax output was optimized with a cross-entropy loss function using the Adam optimizer with a learning
    rate of 2e-4 .

    References
    https://github.com/jgpavez/qa-babi/blob/master/babi_rn.py
    https://github.com/juung/Relation-Network/blob/master/model.py
    https://github.com/gitlimlab/Relation-Network-Tensorflow/blob/master/model_rn.py
    https://github.com/kimhc6028/relational-networks/blob/master/model.py
    https://github.com/sujitpal/dl-models-for-qa

    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self._cfg_ = cfg

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['embed-size'] = 128
        config['lstm-units'] = 32

        return config

    def _create_network_(self):
        EMBED_SIZE = self._cfg_['embed-size']
        LSTM_UNITS = self._cfg_['lstm-units']

        story = Input((self._story_maxlen, self._sentences_maxlen,), name='story')
        labels = Input((self._sentences_maxlen,), name='labels')
        question = Input((self._query_maxlen,), name='question')

        embedded = Embedding(self._vocab_size, EMBED_SIZE)(story)
        embedded = Reshape((self._story_maxlen, EMBED_SIZE * self._sentences_maxlen,))(embedded)

        story_encoder = Bidirectional(LSTM(LSTM_UNITS,
                                           recurrent_regularizer=regularizers.l2(1e-4),
                                           recurrent_dropout=0.3,
                                           return_sequences=True))

        story_encoder = story_encoder(embedded)

        embedded = Embedding(self._vocab_size, EMBED_SIZE)(question)
        question_encoder = Bidirectional(LSTM(LSTM_UNITS,
                                              recurrent_regularizer=regularizers.l2(1e-4),
                                              recurrent_dropout=0.3))
        question_encoder = question_encoder(embedded)

        objects = []
        for k in range(self._story_maxlen):
            fact_object = Lambda(lambda x: x[:, k, :])(story_encoder)
            objects.append(fact_object)

        relations = []
        for fact_object_1, fact_object_2 in itertools.combinations(objects, 2):
            relations.append(concatenate([fact_object_1, fact_object_2, question_encoder]))

        relations = concatenate(relations)
        response = Dense(256, activation='relu')(relations)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(256, activation='relu')(response)
        response = Dropout(0.5)(response)

        response = Dense(256, activation='relu')(response)
        response = Dense(512, activation='relu')(response)

        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question, labels], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
