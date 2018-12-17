import tarfile
from functools import reduce
from keras.utils import get_file

from pypagai.preprocessing.read_data import RemoteDataReader


class CBTDataset(RemoteDataReader):
    ALIAS = 'cbt'
    __URL__ = 'http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz'

    def __init__(self, reader_cfg, model_cfg):
        super().__init__(reader_cfg, model_cfg)
        self.__task__ = reader_cfg['task']
        self.__max_size__ = 20
        self.__only_supporting__ = reader_cfg['only_supporting'] if 'only_supporting' in reader_cfg else False
        self.__strip_sentences__ = reader_cfg['strip_sentences'] if 'strip_sentences' in reader_cfg else False

    @staticmethod
    def default_config():
        return {
            'task': 'V',
        }

    def parse_stories(self, lines, only_supporting=False):
        """Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        """
        data = []
        story = []
        i = 0
        for line in lines:
            if line == b'\n':
                continue

            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, _, supporting = line.split('\t')
                q = self._parser_.tokenize(q)

                # TODO: Filtrar as frases que tem as palavras
                # if only_supporting:
                #     # Only select the related substory
                #     supporting = list(map(int, supporting.split()))
                #     story_range = list(range(len(story)))
                #     supporting = self.select_sentences(story_range, supporting, self.__max_size__)
                #     substory = [story[i - 1] for i in supporting]
                # else:

                # Provide all the substories
                substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self._parser_.tokenize(line)
                story.append(sent)
        return data

    def __get_stories__(self, f, only_supporting=False, max_length=None):
        """
        Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
        If max_length is supplied, any stories longer than max_length tokens will be discarded.
        """
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)

        # TODO: Suspect that max_length is not working when strip sentence is True
        if not self.__strip_sentences__:
            data = [(flatten(story), q, answer) for story, q, answer in data if
                    not max_length or len(flatten(story)) < max_length]

        return data

    def _download_(self):

        challenges = {
            'CN': 'CBTest/data/cbtest_CN_cbt_{}.txt',
            'NE': 'CBTest/data/cbtest_NE_{}.txt',
            'P': 'CBTest/data/cbtest_P_{}.txt',
            'V': 'CBTest/data/cbtest_V_{}.txt',
            # 'generic': 'CBTest/data/cbt_{}.txt',
        }

        path = get_file('CBTest.tar', origin=self.__URL__)

        with tarfile.open(path) as tar:

            challenge = challenges[self.__task__]
            train = 'train'
            valid = 'valid_2000ex'
            test = 'test_2500ex'

            ex_file = tar.extractfile(challenge.format(train))
            train_stories = self.__get_stories__(ex_file, only_supporting=self.__only_supporting__)

            ex_file = tar.extractfile(challenge.format(valid))
            train_stories += self.__get_stories__(ex_file, only_supporting=self.__only_supporting__)

            ex_file = tar.extractfile(challenge.format(test))
            test_stories = self.__get_stories__(ex_file, only_supporting=self.__only_supporting__)

        return train_stories, test_stories

    @staticmethod
    def select_sentences(story, supporting, max_size):
        story = set(story)
        supporting = set(supporting)
        story -= supporting
        story = list(story)
        story = story[::-1]
        story = story[:max_size-len(supporting)]
        story += supporting
        story = sorted(story)
        return list(story)
