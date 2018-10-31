import unittest

from pypagai.preprocessing.parser import SimpleParser
from pypagai.preprocessing.read_data import RemoteDataReader


class RemoteDataReaderTest(unittest.TestCase):

    def setUp(self):
        self.model_cfg = {}
        self.reader_cfg = {'parser': SimpleParser}
        self.parser = RemoteDataReader(self.reader_cfg, self.model_cfg)

    def test_vectorize_stories_case_1(self):

        word_idx = {'i': 0, 'like': 1, 'banana': 2, 'what': 3}
        data = [(['i', 'like', 'banana'], ['what'], 'banana')]
        story_maxlen = None
        query_maxlen = None
        sentences_maxlen = None

        result = self.parser.__vectorize_stories__(word_idx, data, story_maxlen, query_maxlen, sentences_maxlen)

        assert len(result.context) == 1
        assert len(result.query) == 1
        assert len(result.answer) == 1

    def test_vectorize_stories_case_2(self):

        word_idx = {'i': 99, 'like': 1, 'banana': 2, 'what': 3, 'hate': 4, 'soda': 5}
        data = [([['i', 'like', 'banana'], ['i', 'hate', 'soda']], ['what'], 'banana')]
        story_maxlen = None
        query_maxlen = None
        sentences_maxlen = 5

        result = self.parser.__vectorize_stories__(word_idx, data, story_maxlen, query_maxlen, sentences_maxlen)

        assert result.context.shape[1] == 2
        assert len(result.query) == 1
        assert len(result.answer) == 1
