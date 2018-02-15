import tarfile
from keras.utils import get_file
from pypagai.preprocessing.read_data import RemoteDataReader


class BaBIDataset(RemoteDataReader):

    __URL__ = 'https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz'

    def _download_(self):
        path = get_file('babi-tasks-v1-2.tar.gz', origin=self.__URL__)

        challenges = {
            # QA1 with 10,000 samples
            'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
            # QA2 with 10,000 samples
            'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
        }
        challenge_type = 'single_supporting_fact_10k'
        challenge = challenges[challenge_type]

        print('Extracting stories for the challenge:', challenge_type)
        with tarfile.open(path) as tar:
            train_stories = self.get_stories(tar.extractfile(challenge.format('train')))
            test_stories = self.get_stories(tar.extractfile(challenge.format('test')))

        return train_stories, test_stories
