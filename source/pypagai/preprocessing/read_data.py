from __future__ import print_function

import logging

import numpy as np
from nltk import flatten
from sacred import Ingredient
from pypagai import settings
from keras.preprocessing.sequence import pad_sequences

from pypagai.preprocessing.parser import SimpleParser

logging.basicConfig(level=settings.LOG_LEVEL)
LOG = logging.getLogger(__name__)

data_ingredient = Ingredient('dataset_default_cfg')


@data_ingredient.config
def default_dataset_configuration():
    """
    Dataset configuration
    """
    reader = 'pypagai.preprocessing.dataset_babi.BaBIDataset'  # Path to dataset reader
    parser = SimpleParser    # Path to dataset parser ex.: pypagai.preprocessing.parser.SimpleParser
    strip_sentences = False  # Property to split sentences


class ProcessedData:
    """
    Class to store information from dataset
    """
    context = None
    query = None
    answer = None


class DataReader:
    def __init__(self, reader_cfg, model_cfg):
        self._cfg_ = reader_cfg
        self._model_cfg_ = model_cfg
        self._parser_ = self._cfg_['parser']()

    @staticmethod
    def default_config():
        return {}


class RemoteDataReader(DataReader):
    def _download_(self):
        raise Exception("It should be implemented by children classes")

    @staticmethod
    def __vectorize_stories__(word_idx, data, story_maxlen, query_maxlen, sentences_maxlen=None):
        inputs, queries, answers, labels = [], [], [], []
        for story, query, answer in data:
            if sentences_maxlen:
                facts = []
                for sentence in story:
                    s = []
                    for w in sentence:
                        s.append(word_idx[w])
                    facts.append(s)
                labels.append(np.arange(len(facts)))
                inputs.append(pad_sequences(facts, maxlen=sentences_maxlen))
            else:
                inputs.append([word_idx[w] for w in story])

            queries.append([word_idx[w] for w in query])
            answers.append(word_idx[answer])

        dt = ProcessedData()
        dt.context = pad_sequences(inputs, maxlen=story_maxlen)
        dt.query = pad_sequences(queries, maxlen=query_maxlen)
        dt.answer = np.array(answers)
        dt.labels = pad_sequences(labels, maxlen=sentences_maxlen) if sentences_maxlen else []

        return dt

    def read(self):
        train_stories, test_stories = self._download_()

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(flatten(story) + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        if 'vocab_size' not in self._cfg_:
            vocab_size = len(vocab) + 1
        else:
            vocab_size = self._cfg_['vocab_size']

        if 'story_maxlen' not in self._cfg_:
            story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        else:
            story_maxlen = self._cfg_['story_maxlen']

        if 'query_maxlen' not in self._cfg_:
            query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
        else:
            query_maxlen = self._cfg_['query_maxlen']

        if 'strip_sentences' not in self._cfg_ or not self._cfg_['strip_sentences']:
            sentences_maxlen = None
        else:
            sentences_maxlen = max(map(len, (x for h, _, _ in train_stories + test_stories for x in h)))

        self._model_cfg_['vocab_size'] = vocab_size
        self._model_cfg_['story_maxlen'] = story_maxlen
        self._model_cfg_['query_maxlen'] = query_maxlen
        self._model_cfg_['sentences_maxlen'] = sentences_maxlen

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        train_data = self.__vectorize_stories__(word_idx, train_stories, story_maxlen, query_maxlen, sentences_maxlen)
        test_data = self.__vectorize_stories__(word_idx, test_stories, story_maxlen, query_maxlen, sentences_maxlen)

        return train_data, test_data
