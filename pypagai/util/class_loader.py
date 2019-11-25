from pydoc import locate


class ClassLoader:
    """
    Load classes with string
    """

    def _aliases_(self):
        raise Exception("It mus be implemented by children class")

    def _default_location_(self):
        raise Exception("It mus be implemented by children class")

    def _reader_(self):
        raise Exception("It mus be implemented by children class")

    def load(self, class_path):
        if '.' not in class_path:
            if class_path in self._aliases_():
                class_path = self._aliases_()[class_path]

        loaded_class = locate(class_path)

        if loaded_class is None:
            raise Exception("Could not found class " + class_path)

        return loaded_class


class ModelLoader(ClassLoader):

    def _default_location_(self):
        return "pypagai.models"

    def _aliases_(self):
        return {
            'lstm': 'pypagai.models.model_lstm.SimpleLSTM',
            'embed_lstm': 'pypagai.models.model_embed_lstm.EmbedLSTM',
        }


class DatasetLoader(ClassLoader):

    def _default_location_(self):
        return "pypagai.preprocessing."

    def _aliases_(self):
        return {
            'babi': 'pypagai.preprocessing.dataset_babi.BaBIDataset',
            'squad': 'pypagai.preprocessing.dataset_squad.SQuADataset',
            'cbt': 'pypagai.preprocessing.dataset_cbt.CBTDataset',
        }
