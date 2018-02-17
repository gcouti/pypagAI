from pydoc import locate


class ClassLoader:
    def __init__(self, arg_parser):
        self._args_, _ = arg_parser.parse_known_args()
        self._arg_parser_ = arg_parser

    def _aliases_(self):
        raise Exception("It mus be implemented by children class")

    def _reader_(self):
        raise Exception("It mus be implemented by children class")

    def load(self):
        value = self._reader_()
        class_name = value[0]
        params = value[1:]

        if '.' not in class_name:
            class_name = self._aliases_()[class_name]

        instance = locate(class_name)
        return instance(self._arg_parser_, params)


class DataLoader(ClassLoader):

    def __init__(self, arg_parser):
        args = arg_parser.add_argument_group('DataReader')
        args.add_argument('-d', '--data', type=str)

        super().__init__(arg_parser)

    def _aliases_(self):
        return {
            'babi': 'pypagai.preprocessing.dataset_babi.BaBIDataset'
        }

    def _reader_(self):
        if self._args_.data:
            return self._args_.data.split(":")
        else:
            raise Exception("You must specify which reader you will use.")


class ModelLoader(ClassLoader):

    def __init__(self, arg_parser):
        args = arg_parser.add_argument_group('ModelReader')
        args.add_argument('-m', '--model', type=str)

        super().__init__(arg_parser)

    def _aliases_(self):
        return {
            'lstm': 'pypagai.models.model_lstm.SimpleLSTM',
            'embed_lstm': 'pypagai.models.model_embed_lstm.EmbedLSTM',
        }

    def _reader_(self):
        if self._args_.model:
            return self._args_.model.split(":")
        else:
            message = "You must specify which reader you will use.\n"
            message += "There are those models available, but you can create your own:\n"
            message += "\n".join([" -> " + k + " or " + y for k, y in self._aliases_().items()])

            raise Exception(message)
