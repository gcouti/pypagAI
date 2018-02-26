from __future__ import print_function

import keras.backend as K
import theano.tensor as T
from keras.callbacks import LearningRateScheduler
from keras.layers import Input, merge, Embedding, LSTM
from keras.layers.core import Dense, Dropout, Lambda, Activation
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import SGD, Adam

from pypagai.models.base import TensorFlowModel


# class SequenceEmbedding(Embedding):
#     def __init__(self, input_dim, output_dim, position_encoding=False, **kwargs):
#         self.position_encoding = position_encoding
#         self.zeros_vector = T.zeros(output_dim, dtype='float32').reshape((1, output_dim))
#         self.dropout = 0
#         super(SequenceEmbedding, self).__init__(input_dim, output_dim, **kwargs)
#
#     def call(self, x, mask=None):
#         if 0. < self.dropout < 1.:
#             retain_p = 1. - self.dropout
#             B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
#             B = K.expand_dims(B)
#             W = K.in_train_phase(self.get_weights() * B, self.get_weights())
#         else:
#             W = self.W
#         W_ = T.concatenate([self.zeros_vector, W], axis=0)
#         out = K.gather(W_, x)
#         return out


class RN(TensorFlowModel):
    ALIAS = "rn"

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

    https://github.com/jgpavez/qa-babi/blob/master/babi_rn.py
    https://github.com/sujitpal/dl-models-for-qa

    """

    def __init__(self, arg_parser, _):
        super().__init__(arg_parser)

        args = arg_parser.add_argument_group(__name__)
        args.add_argument('--hidden', type=int, default=32)
        args.add_argument('--embed', type=int, default=32)
        args.add_argument('--batch-size', type=int, default=32)
        args.add_argument('--context-maxlen', type=int, default=20)

        EMBED_HIDDEN_SIZE = 20
        MLP_unit = 64

        args = arg_parser.parse()

        self.seed = 12
        self.mask_index = 0

        self.batch_size = args.batch_size
        self.c_max_len = args.context_maxlen
        self.s_max_len = self._story_maxlen
        self.q_max_len = self._query_maxlen
        self.s_input_step = self._story_maxlen
        self.q_input_step = self._query_maxlen

        self.s_hidden = args.hidden
        self.q_hidden = args.hidden

        self.c_word_embed = args.embed
        self.q_word_embed = args.embed

        self.context_vocab_size = self._vocab_size + 1  # consider masking
        self.question_vocab_size = self._vocab_size + 1  # consider masking
        self.answer_vocab_size = self._vocab_size

        story = Input((self._story_maxlen, self._facts_maxlen,), name='story')
        question = Input((self._query_maxlen,), name='question')

        embedded = Embedding(EMBED_HIDDEN_SIZE, 200)(question)
        question_encoder = LSTM(EMBED_HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2)(embedded)
        question_encoder = Lambda(lambda x: K.sum(x, axis=1),
                                 output_shape=lambda shape: (shape[0],) + shape[1:])(question_encoder)

        embedded = Embedding(EMBED_HIDDEN_SIZE, 200)(question)
        story_encoder = LSTM(EMBED_HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedded)
        story_encoder = Lambda(lambda x: K.sum(x, axis=2),
                               output_shape=(self._story_maxlen, EMBED_HIDDEN_SIZE,))(story_encoder)

        objects = []
        for k in range(self._story_maxlen):
            fact_object = Lambda(lambda x: x[:, k, :], output_shape=(20,))(story_encoder)
            objects.append(fact_object)

        relations = []
        for fact_object_1 in objects:
            for fact_object_2 in objects:
                relations.append(merge([fact_object_1, fact_object_2, question_encoder], mode='concat',
                                       output_shape=(None, EMBED_HIDDEN_SIZE * 3,)))

        from keras.layers.normalization import BatchNormalization

        MLP_unit = 64

        def stack_layer(layers):
            def f(x):
                for k in range(len(layers)):
                    x = layers[k](x)
                return x
            return f


        def get_MLP(n):
            r = []
            for k in range(n):
                s = stack_layer([
                    Dense(MLP_unit, input_shape=(EMBED_HIDDEN_SIZE * 3,)),
                    BatchNormalization(),
                    Activation('relu')
                ])
                r.append(s)
            return stack_layer(r)

        g_MLP = get_MLP(3)
        mid_relations = []
        for r in relations:
            mid_relations.append(Dense(MLP_unit, input_shape=(EMBED_HIDDEN_SIZE,))(r))
        combined_relation = merge(mid_relations, mode='sum')

        def bn_dense(x):
            y = Dense(MLP_unit)(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Dropout(0.5)(y)
            return y

        #rn = bn_dense(combined_relation)
        response = Dense(self._vocab_size, init='uniform', activation='sigmoid')(combined_relation)

        model = Model(input=[story, question], output=[response])

        #theano.printing.pydotprint(response, outfile="model.png", var_with_name_simple=True)
        #plot(model, to_file='model.png')

        def scheduler(epoch):
            if (epoch + 1) % 25 == 0:
                lr_val = model.optimizer.lr.get_value()
                model.optimizer.lr.set_value(lr_val*0.5)
            return float(model.optimizer.lr.get_value())

        sgd = SGD(lr=0.01, clipnorm=40.)
        adam = Adam(clipnorm = 40.)

        print('Compiling model...')
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[categorical_accuracy])
        print('Compilation done...')

        lr_schedule = LearningRateScheduler(scheduler)

        # label =

        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=40.), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
