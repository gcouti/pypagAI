import unittest
import numpy as np

from models.model_baseline import RandomModel
from preprocessing.read_data import ProcessedData


class TestRandomModel(unittest.TestCase):

    def test_predict(self):
        """
        While the Random Model has random responses, just check if return one element
        """
        model_cfg = {}
        model_cfg['vocab_size'] = 0
        model_cfg['story_maxlen'] = 0
        model_cfg['query_maxlen'] = 0
        model_cfg['sentences_maxlen'] = 0

        model = RandomModel(model_cfg)

        data = ProcessedData()
        data.context = np.array([[1], [2]])

        result = model.predict(data)
        self.assertEqual(len(result), len(data.context))


if __name__ == '__main__':
    unittest.main()
