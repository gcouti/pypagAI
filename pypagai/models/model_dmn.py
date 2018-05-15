from keras import Input
from keras.layers import Embedding, Dropout

from pypagai.models.base import KerasModel


class DMN(KerasModel):
    """
        Dynamic Memory Networks (March 2016 Version - https://arxiv.org/abs/1603.01417)

        Improved End-To-End version.


        Inspired on: https://github.com/patrickachase/dynamic-memory-networks/blob/master/python/dynamic_memory_network.py
        https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_basic.py
    """
    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['hidden'] = 50

        return config

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        story_maxlen = self._story_maxlen
        query_maxlen = self._query_maxlen
        vocab_size = self._vocab_size

        embed_size = model_cfg['hidden']

        sentence = Input(shape=(story_maxlen,), dtype='int32')
        encoded_sentence = Embedding(vocab_size, embed_size)(sentence)
        encoded_sentence = Dropout(0.3)(encoded_sentence)

        question = Input(shape=(query_maxlen,), dtype='int32')
        encoded_question = Embedding(vocab_size, embed_size)(question)
        encoded_question = Dropout(0.3)(encoded_question)



