import logging
import os
import tarfile
import zipfile

import numpy as np
from keras.utils import get_file

LOG = logging.getLogger('pypagai-logger')


class GloveLoader:
    __GLOVE_FILE__ = 'glove.6B.zip'
    __EMBEDDING_FILE__ = 'glove.6B.%id.txt'
    __EMBEDDING_CACHE_FOLDER__ = 'embeddings'
    __URL__ = 'http://nlp.stanford.edu/data/' + __GLOVE_FILE__

    def __init__(self, word_idx, embedding_dimension=100):
        """
        Read glove vectors from file. It creates a cache on home folder using Keras function

        :param embedding_dimension: (dimensions of glove 50,100,)
        """
        self._embeddings_index = {}
        self.embedding_dimension = embedding_dimension

        self.embedding_matrix = np.zeros((len(word_idx) + 1, self.embedding_dimension))

        self.__read_vectors__()
        self.__populate_embedding__(word_idx)

    def __read_vectors__(self):
        """
        Read vectors from glove

        :param glove_dir:
        """
        path = get_file(self.__GLOVE_FILE__, origin=self.__URL__, cache_subdir=self.__EMBEDDING_CACHE_FOLDER__)

        with zipfile.ZipFile(path) as z:
            with z.open(self.__EMBEDDING_FILE__ % self.embedding_dimension) as f:
                for line in f:
                    values = line.split()
                    word = values[0].decode("utf-8")
                    coefs = np.asarray(values[1:], dtype='float32')
                    self._embeddings_index[word] = coefs

        LOG.debug('Found %s word vectors.' % len(self._embeddings_index))

    def __populate_embedding__(self, word_idx):
        """
        Populate embedding matrix with vectors

        :param word_idx:
        """
        for word, i in word_idx.items():
            embedding_vector = self._embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

