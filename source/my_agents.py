import copy

from keras.optimizers import SGD
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import Agent

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, BatchNormalization, \
    Bidirectional
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

    def save(self, fname):
        self.dictionary.save(fname + '.dict')

    def load(self, fname):
        self.dictionary.load(fname + '.dict')


class N2NMemAgent(Agent):
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

        self._dictionary = DictionaryAgent(opt)
        self.opt = opt
        self.episode_done = True

        self._create_network(opt)

    def _create_network(self, opt):

        statement_size = 8
        self._vocab_size = len(self._dictionary)
        self._story_maxlen = 15 * statement_size
        self._query_maxlen = 1 * statement_size
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

    def observe(self, obs):
        observation = copy.deepcopy(obs)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
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

        history = obs['text'].split('\n')
        inputs_train = np.array([0] * self._story_maxlen)
        history_vec = self._dictionary.txt2vec("\n".join(history[:len(history) - 1]))
        inputs_train[len(inputs_train) - len(history_vec):] = history_vec

        queries_train = np.array([0] * self._query_maxlen)
        query_vec = self._dictionary.txt2vec(history[len(history) - 1])
        queries_train[len(queries_train) - len(query_vec):] = query_vec

        inputs_train = np.array([inputs_train])
        queries_train = np.array([queries_train])

        if 'labels' in obs:
            answers_train = [0.] * self._vocab_size
            answers_train[self._dictionary.txt2vec(obs['labels'][0])[0]] = 1

            # train
            self._model.fit([inputs_train, queries_train], np.array([answers_train]),  verbose=False, epochs=1)

            return [self._dictionary[np.argmax(answers_train)]]
        else:
            predicted = self._model.predict([inputs_train, queries_train], verbose=False)
            return [self._dictionary[int(index)] for index in np.argsort(predicted[0])[::-1][:5]]

    def save(self, fname):
        self._dictionary.save(fname + '.dict')

    def load(self, fname):
        self._dictionary.load(fname + '.dict')

    def reset(self):
        super().reset()
        self.episode_done = True


class RNAgent(Agent):

    @staticmethod
    def add_cmdline_args(parser):
        # DictionaryAgent.add_cmdline_args(parser)
        parser.add_argument('-lp', '--length_penalty', default=0.5, help='length penalty for responses')
        # parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
        # parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
        # parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
        # parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
        # parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        # parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
        # parser.add_argument('--resume', type=str, help='resume from model stored')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'DNCAgent'
        self._dictionary = DictionaryAgent(opt)
        self.opt = opt
        self.episode_done = True

        self._create_network(opt)

    def _create_network(self, opt):
        statement_size = 8
        self._vocab_size = len(self._dictionary)
        self._story_max_len = 15 * statement_size
        self._query_max_len = 15 * statement_size

        self._model = RN(opt)

    def save(self, fname):
        self._dictionary.save(fname + '.dict')

    def load(self, fname):
        self._dictionary.load(fname + '.dict')

    def reset(self):
        super().reset()
        self._episode_done = True

    def observe(self, obs):
        observation = copy.deepcopy(obs)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
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

        history = obs['text'].split('\n')
        inputs_train = np.array([0] * self._story_max_len)
        history_vec = self._dictionary.txt2vec("\n".join(history[:len(history) - 1]))
        inputs_train[len(inputs_train) - len(history_vec):] = history_vec

        queries_train = np.array([0] * self._query_max_len)
        query_vec = self._dictionary.txt2vec(history[len(history) - 1])
        queries_train[len(queries_train) - len(query_vec):] = query_vec

        inputs_train = np.array([inputs_train])
        queries_train = np.array([queries_train])

        inputs_train = np.reshape(inputs_train, (len(inputs_train), self._story_max_len, 1))
        queries_train = np.reshape(queries_train, (len(queries_train), self._query_max_len, 1))

        if 'labels' in obs:
            answers_train = np.array([0.] * self._story_max_len)
            answers_train[self._dictionary.txt2vec(obs['labels'][0])[0]] = 1
            answers_train = np.array([answers_train])

            predicted = self._model.train(inputs_train, queries_train, answers_train)
            return [self._dictionary[int(index)] for index in np.argsort(predicted[0])[::-1][:5]]

        else:
            predicted = self._model.predict(inputs_train, queries_train)
            return [self._dictionary[int(index)] for index in np.argsort(predicted[0])[::-1][:5]]


class RN:

    def __init__(self, args):

        statement_size = 8
        self._vocab_size = 15 * statement_size
        self._story_maxlen = 15 * statement_size
        self._query_maxlen = 15 * statement_size
        drop_out = 0.5
        embedding = 32
        hidden = 256
        lstm_units = 128

        # placeholders
        story = Input((self._story_maxlen,1))
        question = Input((self._query_maxlen,1))

        # encoders
        # embed the input sequence into a sequence of vectors
        # story_encoder = Sequential()
        # story_encoder.add(Embedding(input_dim=self._vocab_size, output_dim=embedding))
        # story_encoder.add(Dropout(drop_out))
        # output: (samples, story_maxlen, embedding_dim)

        # # embed the question into a sequence of vectors
        # question_encoder = Sequential()
        # question_encoder.add(Embedding(input_dim=self._vocab_size, output_dim=embedding, input_length=self._query_maxlen))
        # question_encoder.add(Dropout(drop_out))
        # output: (samples, query_maxlen, embedding_dim)

        # Encode
        # story_encoded = story_encoder(story)
        # question_encoded = question_encoder(question)

        story_encoded = LSTM(lstm_units)(story)
        question_encoded = LSTM(lstm_units)(question)

        response = concatenate([story_encoded, question_encoded])

        # # add the match matrix with the second input vector sequence
        g_response = Dense(hidden, activation='relu', input_dim=lstm_units, name='g')(response)
        g_response = Dense(hidden, activation='relu')(g_response)
        g_response = Dense(hidden, activation='relu')(g_response)
        g_response = Dense(hidden, activation='relu')(g_response)

        g_response = g_response.reshape()

        f_response = Dense(hidden, activation='relu', input_dim=hidden, name='f')(g_response)
        f_response = Dense(hidden, activation='relu')(f_response)
        f_response = Dense(hidden, activation='relu')(f_response)
        f_response = Dropout(drop_out)(f_response)

        pred = Dense(self._vocab_size, activation='softmax')(f_response)

        self.model = Model(inputs=[story, question], outputs=pred)

        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, story, question, answer):
        prediction = self.model.predict([story, question], verbose=False)
        self.model.fit([story, question], answer, verbose=False)
        return prediction

    def predict(self, story, question):
        return self.model.predict([story, question], verbose=False)

