from __future__ import print_function

import logging

from pypagai import settings
from pypagai.util.class_loader import ModelLoader, DataLoader

logging.basicConfig(level=settings.LOG_LEVEL)
LOG = logging.getLogger(__name__)


class ExperimentFlow:
    """
    """

    def __init__(self, arg_parser):

        self.__arg_parser__ = arg_parser
        self.__reader__ = DataLoader(self.__arg_parser__).load()

        self.__model__ = None

    def run(self):
        train_data, test_data = self.__reader__.read()

        self.__model__ = ModelLoader(self.__arg_parser__).load()
        self.__model__.train(train_data, test_data)
