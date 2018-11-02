import unittest

from pypagai.preprocessing.dataset_babi import BaBIDataset


class BaBIDataseetTest(unittest.TestCase):

    def test_only_support(self):
        story = [0, 1]
        supporting = [0]
        max_size = 5

        substory = BaBIDataset.select_sentences(story, supporting, max_size)

        assert len(substory) == 2, "Length should be 2, but was %i" % len(substory)
        assert substory[0] == 0, "Story position should be 0 but was %i" % substory[0]
        assert substory[1] == 1, "Story position should be 1 but was %i" % substory[1]

    def test_only_support_2(self):
        story = [0, 1, 2, 3, 4, 5]
        supporting = [0]
        max_size = 5

        substory = BaBIDataset.select_sentences(story, supporting, max_size)

        assert len(substory) == 5, "Length should be 5, but was %i" % len(substory)
        assert substory[0] == 0, "Story position should be 0 but was %i" % substory[0]
        assert substory[1] == 2, "Story position should be 2 but was %i" % substory[1]
        assert substory[2] == 3, "Story position should be 3 but was %i" % substory[2]
        assert substory[3] == 4, "Story position should be 4 but was %i" % substory[3]
        assert substory[4] == 5, "Story position should be 5 but was %i" % substory[4]

    def test_only_support_3(self):
        story = [0, 1, 2, 3, 4, 5]
        supporting = [5]
        max_size = 5

        substory = BaBIDataset.select_sentences(story, supporting, max_size)

        assert len(substory) == 5, "Length should be 5, but was %i" % len(substory)
        assert substory[0] == 1, "Story position should be 1 but was %i" % substory[0]
        assert substory[1] == 2, "Story position should be 2 but was %i" % substory[1]
        assert substory[2] == 3, "Story position should be 3 but was %i" % substory[2]
        assert substory[3] == 4, "Story position should be 4 but was %i" % substory[3]
        assert substory[4] == 5, "Story position should be 5 but was %i" % substory[4]
