import argparse


class PypagaiParser(argparse.ArgumentParser):
    """
    Add arguments
    """

    def __init__(self):
        super().__init__(description='Pypagai parser.')

        self.__args__ = None

    def parse(self):
        self.__args__, _ = self.parse_known_args(namespace=self.__args__)
        return self.__args__