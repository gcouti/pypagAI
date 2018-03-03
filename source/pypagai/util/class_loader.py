from pydoc import locate


class ClassLoader:

    def _aliases_(self):
        raise Exception("It mus be implemented by children class")

    def _reader_(self):
        raise Exception("It mus be implemented by children class")

    def load(self, class_path):
        if '.' not in class_path:
            class_path = self._aliases_()[class_path]

        return locate(class_path)


class ModelLoader(ClassLoader):

    def _aliases_(self):
        return {
            'lstm': 'pypagai.models.model_lstm.SimpleLSTM',
            'embed_lstm': 'pypagai.models.model_embed_lstm.EmbedLSTM',
        }

