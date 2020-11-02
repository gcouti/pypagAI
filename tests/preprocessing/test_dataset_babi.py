import unittest

from pypagai.preprocessing.dataset_babi import BaBIDataset


class BaBIDatasetTest(unittest.TestCase):

    def test_only_support(self):
        story = [0, 1]
        supporting = [0]
        max_size = 5

        sub_story = BaBIDataset.select_sentences(story, supporting, max_size)

        self.assertEqual(len(sub_story) == 2, "Length should be 2, but was %i" % len(sub_story))
        self.assertEqual(sub_story[0] == 0, "Story position should be 0 but was %i" % sub_story[0])
        self.assertEqual(sub_story[1] == 1, "Story position should be 1 but was %i" % sub_story[1])

    def test_only_support_2(self):
        story = [0, 1, 2, 3, 4, 5]
        supporting = [0]
        max_size = 5

        sub_story = BaBIDataset.select_sentences(story, supporting, max_size)

        self.assertEqual(len(sub_story) == 5, "Length should be 5, but was %i" % len(sub_story))
        self.assertEqual(sub_story[0] == 0, "Story position should be 0 but was %i" % sub_story[0])
        self.assertEqual(sub_story[1] == 2, "Story position should be 2 but was %i" % sub_story[1])
        self.assertEqual(sub_story[2] == 3, "Story position should be 3 but was %i" % sub_story[2])
        self.assertEqual(sub_story[3] == 4, "Story position should be 4 but was %i" % sub_story[3])
        self.assertEqual(sub_story[4] == 5, "Story position should be 5 but was %i" % sub_story[4])

    def test_only_support_3(self):
        story = [0, 1, 2, 3, 4, 5]
        supporting = [5]
        max_size = 5

        sub_story = BaBIDataset.select_sentences(story, supporting, max_size)

        self.assertEqual(len(sub_story) == 5, "Length should be 5, but was %i" % len(sub_story))
        self.assertEqual(sub_story[0] == 1, "Story position should be 1 but was %i" % sub_story[0])
        self.assertEqual(sub_story[1] == 2, "Story position should be 2 but was %i" % sub_story[1])
        self.assertEqual(sub_story[2] == 3, "Story position should be 3 but was %i" % sub_story[2])
        self.assertEqual(sub_story[3] == 4, "Story position should be 4 but was %i" % sub_story[3])
        self.assertEqual(sub_story[4] == 5, "Story position should be 5 but was %i" % sub_story[4])
