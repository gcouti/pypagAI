from __future__ import print_function

import logging

from pypagai import settings

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re

from pypagai.agents.agent_embed_lstm import EmbedLSTM
from pypagai.agents.agent_n2nmemory import N2NMemory
from pypagai.preprocessing.dataset_babi import BaBIDataset
from pypagai.preprocessing.read_data import DataReader, RemoteDataReader

logging.basicConfig(level=settings.LOG_LEVEL)
LOG = logging.getLogger(__name__)


class ProcessedData:

    context = None
    query = None
    answer = None


def vectorize_stories(word_idx, data,story_maxlen, query_maxlen):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))


class ExperimentFlow:
    """
    """

    def __init__(self, parser):
        opt = parser.parse_args()

        self.__model__ = None

    def run(self):

        reader = BaBIDataset()
        train_stories, test_stories = reader.read()

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

        print('-')
        print('Vocab size:', vocab_size, 'unique words')
        print('Story max length:', story_maxlen, 'words')
        print('Query max length:', query_maxlen, 'words')
        print('Number of training stories:', len(train_stories))
        print('Number of test stories:', len(test_stories))
        print('-')
        print('Here\'s what a "story" tuple looks like (input, query, answer):')
        print(train_stories[0])
        print('-')
        print('Vectorizing the word sequences...')

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        inputs_train, queries_train, answers_train = vectorize_stories(word_idx,train_stories, story_maxlen, query_maxlen)
        inputs_test, queries_test, answers_test = vectorize_stories(word_idx, test_stories, story_maxlen, query_maxlen)

        print('-')
        print('inputs: integer tensor of shape (samples, max_length)')
        print('inputs_train shape:', inputs_train.shape)
        print('inputs_test shape:', inputs_test.shape)
        print('-')
        print('queries: integer tensor of shape (samples, max_length)')
        print('queries_train shape:', queries_train.shape)
        print('queries_test shape:', queries_test.shape)
        print('-')
        print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
        print('answers_train shape:', answers_train.shape)
        print('answers_test shape:', answers_test.shape)
        print('-')
        print('Compiling...')

        train_data = ProcessedData()
        train_data.context = inputs_train
        train_data.query = queries_train

        train_data.answer = answers_train

        validation_data = ProcessedData()
        validation_data.context = inputs_test
        validation_data.query = queries_test

        validation_data.answer = answers_test

        self.__model__ = EmbedLSTM(vocab_size, story_maxlen, query_maxlen)
        self.__model__.train(train_data, validation_data)

        # self.__validate__()
        # self.__show_results__()

    # def __train__(self, data, valid=None):
    #     LOG.debug("Training models")



    # def __validate__(self):
    #     LOG.debug("Validate models")
    #
    # def __show_results__(self):
    #     LOG.debug("Show models")

        # Imprimir alguns exemplos certos
        # Imprimir alguns exemplis incorretos
        # Imprimir matriz de confusão
