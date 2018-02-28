from __future__ import print_function

import keras.backend as K
import theano.tensor as T
from keras.callbacks import LearningRateScheduler
from keras.layers import Input, merge, Embedding, LSTM, Reshape, concatenate, add
from keras.layers.core import Dense, Dropout, Lambda, Activation
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import SGD, Adam

from pypagai.models.base import TensorFlowModel, KerasModel


class RN(KerasModel):
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
        LSTM_UNITS = 32

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

        story = Input((self._story_maxlen, self._sentences_maxlen,), name='story')
        question = Input((self._query_maxlen,), name='question')

        embedded = Embedding(self._vocab_size, 200)(story)
        embedded = Reshape((self._story_maxlen, 200 * self._sentences_maxlen,))(embedded)
        story_encoder = LSTM(32, return_sequences=True)(embedded)

        embedded = Embedding(self._vocab_size, 200)(question)
        question_encoder = LSTM(32)(embedded)

        objects = []
        for k in range(self._story_maxlen):
            fact_object = Lambda(lambda x: x[:, k, :])(story_encoder)
            objects.append(fact_object)

        relations = []
        for fact_object_1 in objects:
            for fact_object_2 in objects:
                r = concatenate([fact_object_1, fact_object_2, question_encoder])
                response = Dense(256, activation='relu')(r)
                response = Dropout(0.5)(response)
                response = Dense(256, activation='relu')(response)
                response = Dropout(0.5)(response)
                response = Dense(256, activation='relu')(response)
                response = Dropout(0.5)(response)
                response = Dense(256, activation='relu')(response)
                response = Dropout(0.5)(response)

                relations.append(response)

        combined_relation = add(relations)

        response = Dense(256, activation='relu')(combined_relation)
        response = Dropout(0.5)(response)
        response = Dense(512, activation='relu')(response)
        response = Dropout(0.5)(response)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(clipnorm=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
