import copy

from keras.backend import reshape
from keras.legacy.layers import merge
from keras.optimizers import SGD
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import Agent

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, BatchNormalization, \
    Bidirectional, Merge, Reshape
from keras.layers import LSTM, GRU

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding, \
    LSTM, Bidirectional, Lambda, Concatenate, Add, SimpleRNN
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization, regularizers
from keras.optimizers import Adam, RMSprop
from keras import backend as K

import tensorflow as tf

import collections
import numpy as np
import os

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.control_flow_ops import with_dependencies

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

from torch.autograd import Variable


class DummyAgent(Agent):
    """
        This Agent retrieve only the first candidates
    """
    @staticmethod
    def add_cmdline_args(parser):
        DictionaryAgent.add_cmdline_args(parser)

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'DummyAgent'
        self.dictionary = DictionaryAgent(opt)
        self.opt = opt

    def observe(self, obs):
        self.observation = obs
        self.dictionary.observe(obs)
        return obs

    def act(self):
        if self.opt.get('datatype', '').startswith('train'):
            self.dictionary.act()

        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        # Rank candidates
        if 'label_candidates' in obs and len(obs['label_candidates']) > 0:
            reply['text_candidates'] = self._rank_candidates(obs)
            reply['text'] = reply['text_candidates'][0]
        else:
            reply['text'] = "I don't know."
        return reply

    @staticmethod
    def _rank_candidates(obs):
        return list(obs['label_candidates'])




class BaseDictAgent(Agent):

    def __init__(self, opt):
        super().__init__(opt)
        self._dictionary = DictionaryAgent(opt)
        self._episode_done = True

    def save(self, fname):
        self._dictionary.save(fname + '.dict')

    def load(self, fname):
        self._dictionary.load(fname + '.dict')

    def reset(self):
        super().reset()
        self._episode_done = True

    def observe(self, obs):
        observation = copy.deepcopy(obs)
        if not self._episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self._episode_done = observation['episode_done']
        return observation

    def act(self):
        if self.opt.get('datatype', '').startswith('train'):
            self._dictionary.act()

        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        # Rank candidates
        if 'label_candidates' in obs and len(obs['label_candidates']) > 0:
            reply['text_candidates'] = self._rank_candidates(obs)
            reply['text'] = reply['text_candidates'][0]
        else:
            reply['text'] = "I don't know."
        return reply

    def _rank_candidates(self, obs):
        pass


class N2NMemAgent(BaseDictAgent):
    """
    Python + Keras implementation of End-to-end Memory Network

    References:
        - Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
        "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
        http://arxiv.org/abs/1502.05698

        - Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
        "End-To-End Memory Networks",
        http://arxiv.org/abs/1503.08895

        Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
        Time per epoch: 3s on CPU (core i7).
    """
    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument('-lp', '--length_penalty', default=0.5, help='length penalty for responses')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'N2NMemAgent'
        self.opt = opt

        statement_size = 8
        self._vocab_size = len(self._dictionary)
        self._story_maxlen = 15 * statement_size
        self._query_maxlen = 1 * statement_size

        self._model = N2NMemory(self._vocab_size, self._story_maxlen, self._query_maxlen)

    def _rank_candidates(self, obs):

        history = obs['text'].split('\n')
        inputs_train = np.array([0] * self._story_maxlen)
        history_vec = self._dictionary.txt2vec("\n".join(history[:len(history) - 1]))[:self._story_maxlen]

        inputs_train[len(inputs_train) - len(history_vec):] = history_vec

        queries_train = np.array([0] * self._query_maxlen)
        query_vec = self._dictionary.txt2vec(history[len(history) - 1])[:self._query_maxlen]
        queries_train[len(queries_train) - len(query_vec):] = query_vec

        inputs_train = np.array([inputs_train])
        queries_train = np.array([queries_train])

        if 'labels' in obs:
            answers_train = [0.] * self._vocab_size
            answers_train[self._dictionary.txt2vec(obs['labels'][0])[0]] = 1

            self._model.train(inputs_train, queries_train, np.array([answers_train]))

            return [self._dictionary[np.argmax(answers_train)]]
        else:
            predicted = self._model.predict(inputs_train, queries_train)
            return [self._dictionary[int(index)] for index in np.argsort(predicted[0])[::-1][:5]]


class RNAgent(BaseDictAgent):

    @staticmethod
    def add_cmdline_args(parser):
        pass
        # DictionaryAgent.add_cmdline_args(parser)
        # parser.add_argument('-lp', '--length_penalty', default=0.5, help='length penalty for responses')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RNAgent'
        self._dictionary = DictionaryAgent(opt)
        self.opt = opt
        self.episode_done = True

        self._create_network(opt)

    def _create_network(self, opt):
        statement_size = 8
        self._vocab_size = len(self._dictionary)
        self._story_max_len = 1 * statement_size
        self._query_max_len = 1 * statement_size
        self._max_sentences = 20

        self._model = RN(self._vocab_size, self._query_max_len, self._story_max_len, self._max_sentences)

    def _rank_candidates(self, obs):

        iterator = 0
        history = obs['text'].split('\n')
        inputs_train = np.zeros((self._max_sentences, self._story_max_len))
        for story in history[:len(history) - 1][len(history) - self._max_sentences if len(history) > self._max_sentences else 0:]:
            history_vec = self._dictionary.txt2vec(story)[:self._story_max_len]
            inputs_train[iterator][:len(history_vec)] = history_vec
            iterator += 1

        queries_train = np.array([0] * self._query_max_len)
        query_vec = self._dictionary.txt2vec(history[len(history) - 1])
        queries_train[len(queries_train) - len(query_vec):] = query_vec
        queries_train = np.array([queries_train])

        inputs_train = np.reshape(inputs_train, (1, self._story_max_len, len(inputs_train)))
        queries_train = np.reshape(queries_train, (len(queries_train), self._query_max_len, 1))

        if 'labels' in obs:
            answers_train = np.array([0.] * self._vocab_size)
            answers_train[self._dictionary.txt2vec(obs['labels'][0])[0]] = 1
            answers_train = np.array([answers_train])

            self._model.train(inputs_train, queries_train, answers_train)
            return [self._dictionary[np.argmax(answers_train)]]

        else:
            predicted = self._model.predict(inputs_train, queries_train)
            return [self._dictionary[int(index)] for index in np.argsort(predicted[0])[::-1][:5]]


class EnsembleAgent(BaseDictAgent):

    @staticmethod
    def add_cmdline_args(parser):
        pass
        # DictionaryAgent.add_cmdline_args(parser)
        # parser.add_argument('-lp', '--length_penalty', default=0.5, help='length penalty for responses')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'EnsembleAgent'
        self.opt = opt

        statement_size = 8
        self._vocab_size = len(self._dictionary)
        self._story_max_len = 1 * statement_size
        self._query_max_len = 1 * statement_size
        self._max_sentences = 20

        self._rn_model = RN(self._vocab_size, self._query_max_len, self._story_max_len, self._max_sentences)
        self._nn_model = N2NMemory(self._vocab_size, self._query_max_len, self._story_max_len)
        self._model = EnsembleNetwork(self._vocab_size, [self._rn_model, self._nn_model])

    def _rank_candidates(self, obs):

        iterator = 0
        history = obs['text'].split('\n')
        inputs_train = np.zeros((self._max_sentences, self._story_max_len))
        for story in history[:len(history) - 1][len(history) - self._max_sentences if len(history) > self._max_sentences else 0:]:
            history_vec = self._dictionary.txt2vec(story)[:self._story_max_len]
            inputs_train[iterator][:len(history_vec)] = history_vec
            iterator += 1

        queries_train = np.array([0] * self._query_max_len)
        query_vec = self._dictionary.txt2vec(history[len(history) - 1])
        queries_train[len(queries_train) - len(query_vec):] = query_vec
        queries_train = np.array([queries_train])

        inputs_train = np.reshape(inputs_train, (1, self._story_max_len, len(inputs_train)))
        queries_train = np.reshape(queries_train, (len(queries_train), self._query_max_len, 1))

        if 'labels' in obs:
            answers_train = np.array([0.] * self._vocab_size)
            answers_train[self._dictionary.txt2vec(obs['labels'][0])[0]] = 1
            answers_train = np.array([answers_train])

            self._rn_model.train(inputs_train, queries_train, answers_train)
            self._nn_model.train(inputs_train, queries_train, answers_train)

            self._model.train(inputs_train, queries_train, answers_train)
            return [self._dictionary[np.argmax(answers_train)]]

        else:
            predicted = self._model.predict(inputs_train, queries_train)
            return [self._dictionary[int(index)] for index in np.argsort(predicted[0])[::-1][:5]]


class Networks:

    def __init__(self):
        self._model = None

    def train(self, story, question, answer):
        self._model.fit([story, question], answer, verbose=False, epochs=1)

    def predict(self, story, question):
        return self._model.predict([story, question], verbose=False)


class N2NMemory(Networks):

    def __init__(self, vocab_size, story_maxlen, query_maxlen):

        super().__init__()

        # 10% of the bAbI training set was held-out to form a validation set, which was used to select the optimal
        # model architecture and hyperparameters.  Our models were trained using a learning rate of = 0:01, with
        # anneals every 25 epochs by=2 until 100 epochs were reached.  No momentum or weight decay was used.
        # The weights were initialized randomly from a Gaussian distribution with zero mean and= 0:1 .  When trained
        # on all tasks simultaneously with 1k training samples (10k training  samples),  60  epochs  (20  epochs)
        # were  used  with  learning  rate  anneals  of=2 every  15 epochs (5 epochs).  All training uses a batch size
        # of 32 (but cost is not averaged over a batch), and gradients with an`2 norm larger than 40 are divided by a
        # scalar to have norm 40.  In some of our experiments, we explored commencing training with the softmax
        # in each memory layer removed, making  the  model  entirely  linear  except  for  the  final  softmax  for
        # answer  prediction.   When  the validation loss stopped decreasing, the softmax layers were re-inserted and
        # training recommenced. We  refer  to  this  as  linear  start  (LS)  training.   In  LS  training,  the
        # initial  learning  rate  is  set  to = 0:005 . The capacity of memory is restricted to the most recent
        # 50 sentences. Since the number of sentences and the number of words per sentence varied between problems,
        # a null symbol was used to pad them all to a fixed size. The embedding of the null symbol was constrained to
        # be zero.

        self._story_maxlen = story_maxlen
        self._query_maxlen = query_maxlen
        self._vocab_size = vocab_size

        drop_out = 0.3
        activation = 'softmax'
        samples = 32
        embedding = 64

        # placeholders
        input_sequence = Input((self._story_maxlen,))
        question = Input((self._query_maxlen,))

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
        match = Activation(activation)(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = SimpleRNN(samples)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(drop_out)(answer)
        answer = Dense(self._vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation(activation)(answer)

        # build the final model
        self._model = Model([input_sequence, question], answer)
        self._model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


class RN(Networks):

    def __init__(self, vocab_size, story_maxlen, query_maxlen, max_sentences):

        super().__init__()
        self._vocab_size = vocab_size
        self._story_maxlen = story_maxlen
        self._query_maxlen = query_maxlen
        self._max_sentences = max_sentences

        drop_out = 0.5
        hidden = 256
        lstm_units = 32

        # placeholders
        story = Input((self._story_maxlen, max_sentences))
        question = Input((self._query_maxlen, 1))

        question_encoded = LSTM(lstm_units)(question)

        responses = []
        for i in range(story.shape[2]):

            # input_i = story
            input_i = Lambda(lambda x: x[:, :, i])(story)
            input_i = Reshape((self._story_maxlen, 1))(input_i)
            story_encoded = LSTM(lstm_units)(input_i)
            response = concatenate([story_encoded, question_encoded])

            # # add the match matrix with the second input vector sequence
            g_response = Dense(hidden, activation='relu', input_dim=lstm_units)(response)
            g_response = Dense(hidden, activation='relu')(g_response)
            g_response = Dense(hidden, activation='relu')(g_response)
            g_response = Dropout(drop_out)(g_response)
            g_response = Dense(hidden, activation='relu')(g_response)
            responses.append(g_response)

        responses = add(responses)

        f_response = Dense(hidden, activation='relu', input_dim=hidden, name='f')(responses)
        f_response = Dense(512, activation='relu')(f_response)
        f_response = Dropout(drop_out)(f_response)
        pred = Dense(self._vocab_size, activation='softmax')(f_response)

        optimizer = Adam(lr=2e-4)
        self._model = Model(inputs=[story, question], outputs=pred)
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


class EnsembleNetwork(Networks):

    def __init__(self, vocab_size, models):

        super().__init__()
        self._vocab_size = vocab_size
        self._model_list = models

        hidden = 32
        drop_out = 0.3

        input_models = Input((self._vocab_size, len(models)))
        response = Add()(input_models)

        # # add the match matrix with the second input vector sequence
        response = Dense(hidden, activation='relu', input_dim=self._vocab_size)(response)
        response = Dropout(drop_out)(response)
        response = Dense(hidden, activation='relu')(response)
        response = Dropout(drop_out)(response)
        response = Dense(hidden, activation='relu')(response)
        response = Dropout(drop_out)(response)
        pred = Dense(hidden, activation='softmax')(response)

        self._model = Model(inputs=models, outputs=pred)
        self._model.compile(optimizer=SGD(lr=0.01, clipvalue=0.5), loss='categorical_crossentropy', metrics=['accuracy'])
