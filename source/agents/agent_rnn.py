import itertools

from keras import backend as K
from keras.layers import Dense, Input, LSTM, Embedding, K, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam

from agents.base import BaseKerasAgent, Networks


def combination(x, embedded_question):
    items = []
    for i in range(x.shape[1]):
        items.append(Reshape((1,))(x[0][i:i+1]))

    rn_inputs = []
    for object_pair in list(itertools.combinations(items, 2)):
        rn_input = K.concatenate([object_pair[0], object_pair[1], embedded_question], axis=1)
        rn_inputs.append(rn_input)

    return K.concatenate(rn_inputs, axis=0)


def wise_sum(x):
    return K.sum(x, axis=0)


class RN(Networks):

    def __init__(self, opt, vocab_size, story_maxlen, query_maxlen):
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

        https://github.com/juung/Relation-Network/blob/master/train.py
        """

        super().__init__()

        self._batch_size = 1 #opt['keras-batch-size']
        self._story_maxlen = story_maxlen
        self._query_maxlen = query_maxlen
        self._vocab_size = vocab_size

        self.answer_vocab_size = vocab_size
        self.context_vocab_size = vocab_size
        self.question_vocab_size = vocab_size

        self.__story_hidden = opt['hidden']
        self.__question_hidden = opt['hidden']
        self.__embed_dimension = opt['embed']


        # Network parameters
        self.mask_index = opt['mask_index'] if 'mask_index' in opt else 0

        self.s_input_step = opt['s_input_step'] if 's_input_step' in opt else 2
        self.q_input_step = opt['q_input_step'] if 'q_input_step' in opt else 8

        self.c_word_embed = opt['c_word_embed'] if 'c_word_embed' in opt else 32
        self.q_word_embed = opt['q_word_embed'] if 'q_word_embed' in opt else 32

        story = Input((self._story_maxlen, ), name='story')
        question = Input((self._query_maxlen, ), name='question')

        embedded_story = Embedding(self._vocab_size, output_dim=self.__embed_dimension)(story)
        embedded_question = Embedding(self._vocab_size, output_dim=self.__embed_dimension)(question)

        embedded_story = LSTM(self.__story_hidden, dropout=0.2, recurrent_dropout=0.2)(embedded_story)
        embedded_question = LSTM(self.__question_hidden, dropout=0.2, recurrent_dropout=0.2)(embedded_question)

        # TODO: How to deal with label?
        # 20 combination 2 --> total 190 object pairs
        rn_input = Lambda(lambda x: combination(x, embedded_question))(embedded_story)

        # add the match matrix with the second input vector sequence
        g_response = Dense(256, activation='relu')(rn_input)
        g_response = Dense(256, activation='relu')(g_response)
        g_response = Dense(256, activation='relu')(g_response)
        g_response = Dense(256, activation='relu')(g_response)
        g_response = Reshape((1, 256))(g_response)

        f_response = Lambda(wise_sum)(g_response)
        f_response = Dense(256, activation='relu')(f_response)
        f_response = Dense(512, activation='relu')(f_response)

        response = Dense(self._vocab_size, activation='softmax')(f_response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(lr=2e-4),  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class RNAgent(BaseKerasAgent):

    @staticmethod
    def add_cmdline_args(parser):
        BaseKerasAgent.add_cmdline_args(parser)

        agent = parser.add_argument_group('RN Network Arguments')

        message = 'Number of hidden layers'
        agent.add_argument('-hd', '--hidden', type=int, default=32, help=message)
        agent.add_argument('-eb', '--embed', type=int, default=32, help=message)

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RNAgent'
        self.opt = opt

        self.__create_network__(opt)

    def __create_network__(self, opt):
        self._vocab_size = len(self._dictionary)
        self._story_maxlen = opt['story_length']
        self._query_maxlen = opt['query_length']

        self._model = RN(opt,  self._vocab_size, self._story_maxlen, self._query_maxlen)
