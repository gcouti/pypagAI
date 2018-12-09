import json
import tarfile
from functools import reduce
from keras.utils import get_file

from pypagai.preprocessing.read_data import RemoteDataReader


class SQuADataset(RemoteDataReader):
    ALIAS = 'squad'
    __URL__ = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'

    def __init__(self, reader_cfg, model_cfg):
        super().__init__(reader_cfg, model_cfg)
        self.__dataset_version__ = reader_cfg['version'] if 'version' in reader_cfg else '1.1'
        self.__strip_sentences__ = reader_cfg['strip_sentences'] if 'strip_sentences' in reader_cfg else False

    @staticmethod
    def default_config():
        return {
            'version': '1.1',
        }

    def __get_stories__(self, f, only_supporting=False, max_length=None):
        """
        Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
        If max_length is supplied, any stories longer than max_length tokens will be discarded.
        """
        data = []
        readed_data = json.load(open(f))

        for data_dict in readed_data['data']:
            for paragraph in data_dict['paragraphs']:
                if self.__strip_sentences__:
                    story = []
                    for txt in paragraph['context'].split('.'):
                        story.append(self._parser_.tokenize(txt))
                else:
                    story = self._parser_.tokenize(paragraph['context'])
                for qas in paragraph['qas']:
                    question = self._parser_.tokenize(qas['question'])
                    for ans in qas['answers']:
                        answer = self._parser_.tokenize(ans['text'])
                        if len(answer) == 0 or len(answer) > 1:
                            continue
                        # TODO: Fix answer with more than one words
                        data.append((story, question, answer[0]))

        # TODO: Fix strip sentences
        # TODO: Fix sentence max length

        return data

    def _download_(self):
        file_name = 'train-v1.1.json'
        train_path = get_file(file_name, origin=self.__URL__ + file_name)
        file_name = 'dev-v1.1.json'
        dev_path = get_file(file_name, origin=self.__URL__ + file_name)

        train_stories = self.__get_stories__(train_path)
        test_stories = self.__get_stories__(dev_path)

        return train_stories, test_stories
