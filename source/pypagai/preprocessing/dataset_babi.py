import tarfile
from functools import reduce
from keras.utils import get_file

from pypagai.preprocessing.read_data import RemoteDataReader


class BaBIDataset(RemoteDataReader):
    ALIAS = 'babi'
    __URL__ = 'https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz'

    def __init__(self, reader_cfg, model_cfg):
        super().__init__(reader_cfg, model_cfg)
        self.__size__ = '-' + reader_cfg['size'] if 'size' in reader_cfg else ''
        self.__task__ = reader_cfg['task']
        self.__strip_sentences__ = reader_cfg['strip_sentences'] if 'strip_sentences' in reader_cfg else False

    def __get_stories__(self, f, only_supporting=False, max_length=None):
        """
        Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
        If max_length is supplied, any stories longer than max_length tokens will be discarded.
        """
        data = self._parser_.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)

        if not self.__strip_sentences__:
            data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]

        return data

    def _download_(self):

        challenges = {
            '1': 'tasks_1-20_v1-2/en{}/qa1_single-supporting-fact_{}.txt',
            '2': 'tasks_1-20_v1-2/en{}/qa2_two-supporting-facts_{}.txt',
            '3': 'tasks_1-20_v1-2/en{}/qa3_three-supporting-facts_{}.txt',
            '4': 'tasks_1-20_v1-2/en{}/qa4_two-arg-relations_{}.txt',
            '5': 'tasks_1-20_v1-2/en{}/qa5_three-arg-relations_{}.txt',
            '6': 'tasks_1-20_v1-2/en{}/qa6_yes-no-questions_{}.txt',
            '7': 'tasks_1-20_v1-2/en{}/qa7_counting_{}.txt',
            '8': 'tasks_1-20_v1-2/en{}/qa8_lists-sets_{}.txt',
            '9': 'tasks_1-20_v1-2/en{}/qa9_simple-negation_{}.txt',
            '10': 'tasks_1-20_v1-2/en{}/qa10_indefinite-knowledge_{}.txt',
            '11': 'tasks_1-20_v1-2/en{}/qa11_basic-coreference_{}.txt',
            '12': 'tasks_1-20_v1-2/en{}/qa12_conjunction_{}.txt',
            '13': 'tasks_1-20_v1-2/en{}/qa13_compound-coreference_{}.txt',
            '14': 'tasks_1-20_v1-2/en{}/qa14_time-reasoning_{}.txt',
            '15': 'tasks_1-20_v1-2/en{}/qa15_basic-deduction_{}.txt',
            '16': 'tasks_1-20_v1-2/en{}/qa16_basic-induction_{}.txt',
            '17': 'tasks_1-20_v1-2/en{}/qa17_positional-reasoning_{}.txt',
            '18': 'tasks_1-20_v1-2/en{}/qa18_size-reasoning_{}.txt',
            '19': 'tasks_1-20_v1-2/en{}/qa19_path-finding_{}.txt',
            '20': 'tasks_1-20_v1-2/en{}/qa20_agents-motivations_{}.txt',
        }

        path = get_file('babi-tasks-v1-2.tar.gz', origin=self.__URL__)

        challenge = challenges[str(self.__task__)]

        with tarfile.open(path) as tar:
            train_stories = self.__get_stories__(tar.extractfile(challenge.format(self.__size__, 'train')))
            test_stories = self.__get_stories__(tar.extractfile(challenge.format(self.__size__, 'test')))

        return train_stories, test_stories
