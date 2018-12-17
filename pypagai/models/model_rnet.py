from __future__ import print_function

import itertools

from keras.layers import Input, InputLayer
from keras.layers import LSTM, Reshape, concatenate, regularizers
from keras.layers.core import Dense, RepeatVector, Masking, Dropout
from keras.layers.core import Lambda
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model, Sequential
from keras.optimizers import Adam

from pypagai.models.base import KerasModel

# from keras.layers import QuestionAttnGRU, SelfAttnGRU, PointerGRU, QuestionPooling, VariationalDropout
# , Slice, SharedWeight

"""
    Inspired on
    https://github.com/YerevaNN/R-NET-in-Keras
"""


class RNet(Model):
    def __init__(self, inputs=None, outputs=None,
                 N=None, M=None, C=25, unroll=False,
                 hdim=75, word2vec_dim=300,
                 dropout_rate=0,
                 char_level_embeddings=False,
                 **kwargs):
        # Load model from config
        if inputs is not None and outputs is not None:
            super(RNet, self).__init__(inputs=inputs,
                                       outputs=outputs,
                                       **kwargs)
            return

        '''Dimensions'''
        B = None
        H = hdim
        W = word2vec_dim

        v = SharedWeight(size=(H, 1), name='v')
        WQ_u = SharedWeight(size=(2 * H, H), name='WQ_u')
        WP_u = SharedWeight(size=(2 * H, H), name='WP_u')
        WP_v = SharedWeight(size=(H, H), name='WP_v')
        W_g1 = SharedWeight(size=(4 * H, 4 * H), name='W_g1')
        W_g2 = SharedWeight(size=(2 * H, 2 * H), name='W_g2')
        WP_h = SharedWeight(size=(2 * H, H), name='WP_h')
        Wa_h = SharedWeight(size=(2 * H, H), name='Wa_h')
        WQ_v = SharedWeight(size=(2 * H, H), name='WQ_v')
        WPP_v = SharedWeight(size=(H, H), name='WPP_v')
        VQ_r = SharedWeight(size=(H, H), name='VQ_r')

        shared_weights = [v, WQ_u, WP_u, WP_v, W_g1, W_g2, WP_h, Wa_h, WQ_v, WPP_v, VQ_r]

        P_vecs = Input(shape=(N, W), name='P_vecs')
        Q_vecs = Input(shape=(M, W), name='Q_vecs')

        if char_level_embeddings:
            P_str = Input(shape=(N, C), dtype='int32', name='P_str')
            Q_str = Input(shape=(M, C), dtype='int32', name='Q_str')
            input_placeholders = [P_vecs, P_str, Q_vecs, Q_str]

            char_embedding_layer = TimeDistributed(Sequential([
                InputLayer(input_shape=(C,), dtype='int32'),
                Embedding(input_dim=127, output_dim=H, mask_zero=True),
                Bidirectional(GRU(units=H))
            ]))

            # char_embedding_layer.build(input_shape=(None, None, C))

            P_char_embeddings = char_embedding_layer(P_str)
            Q_char_embeddings = char_embedding_layer(Q_str)

            P = Concatenate()([P_vecs, P_char_embeddings])
            Q = Concatenate()([Q_vecs, Q_char_embeddings])

        else:
            P = P_vecs
            Q = Q_vecs
            input_placeholders = [P_vecs, Q_vecs]

        uP = Masking()(P)
        for i in range(3):
            uP = Bidirectional(GRU(units=H,
                                   return_sequences=True,
                                   dropout=dropout_rate,
                                   unroll=unroll))(uP)
        uP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uP')(uP)

        uQ = Masking()(Q)
        for i in range(3):
            uQ = Bidirectional(GRU(units=H,
                                   return_sequences=True,
                                   dropout=dropout_rate,
                                   unroll=unroll))(uQ)
        uQ = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uQ')(uQ)

        vP = QuestionAttnGRU(units=H,
                             return_sequences=True,
                             unroll=unroll)([
            uP, uQ,
            WQ_u, WP_v, WP_u, v, W_g1
        ])
        vP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, H), name='vP')(vP)

        hP = Bidirectional(SelfAttnGRU(units=H,
                                       return_sequences=True,
                                       unroll=unroll))([
            vP, vP,
            WP_v, WPP_v, v, W_g2
        ])

        hP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='hP')(hP)

        gP = Bidirectional(GRU(units=H,
                               return_sequences=True,
                               unroll=unroll))(hP)

        rQ = QuestionPooling()([uQ, WQ_u, WQ_v, v, VQ_r])
        rQ = Dropout(rate=dropout_rate, name='rQ')(rQ)

        fake_input = GlobalMaxPooling1D()(P)
        fake_input = RepeatVector(n=2, name='fake_input')(fake_input)

        ps = PointerGRU(units=2 * H,
                        return_sequences=True,
                        initial_state_provided=True,
                        name='ps',
                        unroll=unroll)([
            fake_input, gP,
            WP_h, Wa_h, v,
            rQ
        ])

        answer_start = Slice(0, name='answer_start')(ps)
        answer_end = Slice(1, name='answer_end')(ps)

        inputs = input_placeholders + shared_weights
        outputs = [answer_start, answer_end]

        super(RNet, self).__init__(inputs=inputs,
                                   outputs=outputs,
                                   **kwargs)


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

    def __init__(self, cfg):
        super().__init__(cfg)
        self._cfg_ = cfg
