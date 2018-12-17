import re
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string


class SimpleParser:

    def tokenize(self, sent):
        """
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        """
        return [x.strip() for x in re.split('(\W+)?', sent.lower()) if x.strip()]


class NonPunctuationParser:

    def tokenize(self, sent):
        """
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', 'Where', 'is', 'the', 'apple']
        """
        tokenizer = RegexpTokenizer(r'\w+')
        return [x.strip() for x in tokenizer.tokenize(sent.lower()) if x.strip()]


class NoStopWordsParser:

    def __init__(self):
        self.__stop_words__ = stopwords.words('english') + list(string.punctuation)

    def tokenize(self, sent):
        """
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'apple', 'Where', 'is', 'apple']
        """
        tokenizer = RegexpTokenizer(r'\w+')
        return [x.strip().lower() for x in tokenizer.tokenize(sent.lower()) if x.strip() not in self.__stop_words__]
