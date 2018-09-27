import re


class SimpleParser:

    def tokenize(self, sent):
        """
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        """
        return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]
