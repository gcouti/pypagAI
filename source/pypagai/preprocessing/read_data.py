from __future__ import print_function

import re
import logging

from functools import reduce
from pypagai import settings

logging.basicConfig(level=settings.LOG_LEVEL)
LOG = logging.getLogger(__name__)


class DataReader:

    def __init__(self):
        # Read flow parameters
        pass


class RemoteDataReader(DataReader):
    def _download_(self):
        raise Exception("It should be implemented by children classes")

    def tokenize(self, sent):
        """
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        """
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

    def parse_stories(self, lines, only_supporting=False):
        """Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        """
        data = []
        story = []
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                substory = None
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def get_stories(self, f, only_supporting=False, max_length=None):
        '''Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        '''
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]

        return data

    def read(self):

        try:
            # TODO: check if download
            train_stories, test_stories = self._download_()
        except Exception as e:
            raise Exception('Error downloading dataset, please download it manually', e)

        return train_stories, test_stories
