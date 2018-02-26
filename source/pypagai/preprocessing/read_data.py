from __future__ import print_function

import logging

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from pypagai import settings
from pypagai.preprocessing.parser import SimpleParser

logging.basicConfig(level=settings.LOG_LEVEL)
LOG = logging.getLogger(__name__)


class ProcessedData:
    """

    """
    context = None
    query = None
    answer = None


class DataReader:

    def __init__(self, args_parser):

        self._args_parser_ = args_parser
        args = self._args_parser_.add_argument_group('DataReader')
        args.add_argument('--vocab_size', type=int, default=None)
        args.add_argument('--story_maxlen', type=int, default=None)
        args.add_argument('--query_maxlen', type=int, default=None)

        self._args_ = args_parser.parse()
        self._parser_ = SimpleParser()


class RemoteDataReader(DataReader):

    def _download_(self):
        raise Exception("It should be implemented by children classes")

    def __vectorize_stories__(self, word_idx, data, story_maxlen, query_maxlen):
        inputs, queries, answers = [], [], []
        for story, query, answer in data:
            inputs.append([word_idx[w] for w in story])
            queries.append([word_idx[w] for w in query])
            answers.append(word_idx[answer])
        return (pad_sequences(inputs, maxlen=story_maxlen),
                pad_sequences(queries, maxlen=query_maxlen),
                np.array(answers))


    # for story, query, answer in data:
        #     x = np.zeros((len(story), fact_maxlen),dtype='int32')
        #     for k,facts in enumerate(story):
        #         if not enable_time:
        #             x[k][-len(facts):] = np.array([word_idx[w] for w in facts])[:fact_maxlen]
        #         else:
        #             x[k][-len(facts)-1:-1] = np.array([word_idx[w] for w in facts])[:facts_maxlen-1]
        #             x[k][-1] = len(word_idx) + len(story) - k
        #     xq = [word_idx[w] for w in query]
        #     y = np.zeros(len(word_idx) + 1) if not enable_time else np.zeros(len(word_idx) + 1 + story_maxlen)
        #     y[word_idx[answer]] = 1
        #     X.append(x)
        #     Xq.append(xq)
        #     Y.append(y)
        # return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

    def read(self):
        try:
            # TODO: check if download
            train_stories, test_stories = self._download_()
        except Exception as e:
            raise Exception('Error downloading dataset, please download it manually', e)

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        if not self._args_.vocab_size:
            vocab_size = len(vocab) + 1
        else:
            vocab_size = self._args_.vocab_size

        if not self._args_.story_maxlen:
            story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        else:
            story_maxlen = self._args_.story_maxlen

        if not self._args_.query_maxlen:
            query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
        else:
            query_maxlen = self._args_.query_maxlen

        self._args_.vocab_size = vocab_size
        self._args_.story_maxlen = story_maxlen
        self._args_.query_maxlen = query_maxlen

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        inputs_train, queries_train, answers_train = self.__vectorize_stories__(word_idx, train_stories, story_maxlen, query_maxlen)
        inputs_test, queries_test, answers_test = self.__vectorize_stories__(word_idx, test_stories, story_maxlen, query_maxlen)

        train_data = ProcessedData()
        train_data.context = inputs_train
        train_data.query = queries_train
        train_data.answer = answers_train

        validation_data = ProcessedData()
        validation_data.context = inputs_test
        validation_data.query = queries_test
        validation_data.answer = answers_test

        return train_data, validation_data
