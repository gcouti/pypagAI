import unittest

from pypagai.preprocessing.parser import SimpleParser


class SimpleParserTest(unittest.TestCase):

    def setUp(self):
        self.parser = SimpleParser()

    def test_tokenize_simple(self):
        string = "test"
        result = self.parser.tokenize(string)

        assert len(result) == 1, "Size is incorrect, should be 1, but was " + str(len(result))
        assert result[0] == "test", "Token should be 'test', but was " + result[0]

    def test_tokenize_complete(self):
        string = "my name is Gabriel"
        result = self.parser.tokenize(string)

        assert len(result) == 4, "Size is incorrect, should be 1, but was " + str(len(result))
        assert result[0] == "my", "Token should be 'my', but was " + result[0]
        assert result[1] == "name", "Token should be 'name', but was " + result[1]
        assert result[2] == "is", "Token should be 'is', but was " + result[2]
        assert result[3] == "gabriel", "Token should be 'gabriel', but was " + result[3]

    def test_tokenize_punct(self):
        string = "finished!"
        result = self.parser.tokenize(string)

        assert len(result) == 2, "Size is incorrect, should be 1, but was " + str(len(result))
        assert result[0] == "finished", "Token should be 'finished', but was " + result[0]
        assert result[1] == "!", "Token should be '!', but was " + result[1]
