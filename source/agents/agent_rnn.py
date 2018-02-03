import numpy as np
from keras.layers import Dense, Dropout, Input, LSTM, Lambda, Embedding, Flatten
from keras.layers import add, concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam
import itertools
import tensorflow as tf
from keras.preprocessing import sequence
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib import rnn

from agents.base import BaseKerasAgent, Networks


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

        response = self.__rn__(story, question)
        response = Dense(self._vocab_size, activation='softmax')(response)

        self._model = Model(inputs=[story, question], outputs=response)
        self._model.compile(optimizer=Adam(lr=2e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    def __rn__(self, story, question):

        embedded_story = Embedding(self._vocab_size, output_dim=self.__embed_dimension)(story)
        embedded_question = Embedding(self._vocab_size, output_dim=self.__embed_dimension)(question)

        embedded_story = LSTM(self.__story_hidden, dropout=0.2, recurrent_dropout=0.2)(embedded_story)
        embedded_question = LSTM(self.__question_hidden, dropout=0.2, recurrent_dropout=0.2)(embedded_question)

        rn_input = self.convert_to_RN_input(embedded_story, embedded_question)
        # f_input = self.g_theta(rn_input, reuse=False, phase=self.is_training)
        # pred = self.f_phi(f_input, reuse=False, phase=self.is_training)

        return embedded_story

    def convert_to_RN_input(self, embedded_c, embedded_q):
        """
        Args
            embedded_c: output of contextLSTM, 20 length list of embedded sentences
            embedded_q: output of questionLSTM, embedded question
        Returns
            RN_input: input for RN g_theta, shape = [batch_size*190, (52+52+32)]
            considered batch_size and all combinations
        """
        # 20 combination 2 --> total 190 object pairs
        object_pairs = list(itertools.combinations(embedded_c, 2))
        # concatenate with question
        RN_inputs = []
        for object_pair in object_pairs:
            RN_inputs.append(tf.concat([object_pair[0], object_pair[1], embedded_q], axis=1))

        return tf.concat(RN_inputs, axis=0)

    def batch_norm_relu(self, inputs, output_shape, phase=True, scope=None, activation=True):
        with tf.variable_scope(scope):
            h1 = fully_connected(inputs, output_shape, activation_fn=None, scope="dense")
            h2 = batch_norm(h1, decay=0.95, center=True, scale=True, is_training=phase, scope='bn', updates_collections=None)

            if activation:
                out = tf.nn.relu(h2, 'relu')
            else:
                out = h2
            return out

    def g_theta(self, RN_input, scope='g_theta', reuse=True, phase=True):
        """
        Args
            RN_input: [o_i, o_j, q], shape = [batch_size*190, 136]
        Returns
            g_output: shape = [190, batch_size, 256]
        """
        g_units = [256, 256, 256, 256]
        with tf.variable_scope(scope, reuse=reuse) as scope:
            g_1 = self.batch_norm_relu(RN_input, g_units[0], scope='g_1', phase=phase)
            g_2 = self.batch_norm_relu(g_1, g_units[1], scope='g_2', phase=phase)
            g_3 = self.batch_norm_relu(g_2, g_units[2], scope='g_3', phase=phase)
            g_4 = self.batch_norm_relu(g_3, g_units[3], scope='g_4', phase=phase)
        g_output = tf.reshape(g_4, shape=[-1, self.batch_size, g_units[3]])
        return g_output

    def f_phi(self, g, scope="f_phi", reuse=True, phase=True):
        """
        Args
            g: g_theta result, shape = [190, batch_size, 256]
        Returns
            f_output: shape = [batch_size, 159]
        """
        f_input = tf.reduce_sum(g, axis=0)
        f_units = [256, 512, self.answer_vocab_size]
        with tf.variable_scope(scope, reuse=reuse) as scope:
            f_1 = self.batch_norm_relu(f_input, f_units[0], scope="f_1", phase=phase)
            f_2 = self.batch_norm_relu(f_1, f_units[1], scope="f_2", phase=phase)
            f_3 = self.batch_norm_relu(f_2, f_units[2], scope="f_3", phase=phase)
        return f_3


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
