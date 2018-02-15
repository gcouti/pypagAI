import copy
import numpy as np

from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

import keras
callback = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)


class BaseNeuralNetworkAgent(Agent):
    """
    Super class which can be used with all types of neural network pypagAI.agents.
    """

    @staticmethod
    def add_cmdline_args(parser):
        DictionaryAgent.add_cmdline_args(parser)

        agent = parser.add_argument_group('Neural Networks Arguments')

        message = 'Text maximum size'
        agent.add_argument('-sl', '--story-length', type=int, default=15, help=message)

        message = 'Text maximum size'
        agent.add_argument('-ql', '--query-length', type=int, default=15, help=message)

        message = ''
        agent.add_argument('-uc', '--use-candidates', type=bool, default=False, help=message)

        message = ''
        agent.add_argument('-iwq', '--input-without-question', type=bool, default=True, help=message)

        message = ''
        agent.add_argument('-iah', '--input-aggregate-history', type=bool, default=True, help=message)

    def __init__(self, opt):
        super().__init__(opt)
        self._dictionary = DictionaryAgent(opt)

        self._categorical = False
        self._episode_done = True
        self._prev_dialogue = ""
        self._story_length = opt['story_length']
        self._query_length = opt['query_length']

        self._model_file = self.opt.get('model_file', None)
        self._use_candidates = self.opt.get('use_candidates')

        self._input_without_question = opt['input_without_question']
        self._input_aggregate_history = opt['input_aggregate_history']

    def _parse(self, texts):
        if type(texts) == str:
            return self._dictionary.txt2vec(texts)
        elif type(texts) in (list, tuple, set):
            result = []
            for t in texts:
                result.append(self._dictionary.txt2vec(t)[0])
            return result
        else:
            raise BaseException("Type not implemented %s " % str(type(texts)))

    def reset(self):
        super().reset()
        self._episode_done = True
        self._prev_dialogue = ""

    def observe(self, obs):
        observation = copy.deepcopy(obs)
        if not self._episode_done:

            self._episode_done = observation['episode_done']

            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            if self._input_without_question:
                observation['text'] = "\n".join(self.observation['text'].split('\n')[:-1])
            else:
                observation['text'] = self.observation['text']

            if self._input_aggregate_history:
                self._prev_dialogue = self._prev_dialogue + "\n" + observation['text']
                observation['text'] = self._prev_dialogue

                if self._episode_done:
                    self._prev_dialogue = ""

        self.observation = observation

        return observation

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):

        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, qs, ys, cands = self.batchify(observations)

        if xs is None:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available
        predictions, cands = self.predict(xs, qs, ys, cands)

        # print(predictions)

        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            curr = batch_reply[i]
            curr['text_candidates'] = [self._dictionary.vec2txt([a]) for a in np.argsort(predictions)[0][::-1]]
            if self._use_candidates:
                i = np.argmax(predictions[i][cands[i]])
                curr['text'] = curr['text_candidates'][i]
            else:
                curr['text'] = self._dictionary.vec2txt([np.argmax(predictions[i])])

        return batch_reply

    def _transform_input_(self, tokenized, max_len=None):
        if not max_len:
            max_len = max([len(x) for x in tokenized])
        return pad_sequences(tokenized, maxlen=max_len)

        # result = []
        # for entry in tokenized:
        #     new_x = [0] * max_len
        #     new_x[:len(entry)] = entry[:max_len]
        #     result.append(new_x)
        # np.array(result)

    def batchify(self, observations):
        """
        This class is used can be extend with any kind of Keras agent. Convert a list of observations into input &
        target tensors.
        """

        # valid examples
        exs = [ex for ex in observations if 'text' in ex]

        # TODO: Fazer alteração aqui para encapsular em uma função que possa ser sobescrita
        # tokenize the text
        parsed = [self._parse('\n'.join(ex['text'].split('\n')[:-1])) for ex in exs]
        xs = self._transform_input_(parsed, max_len=self._story_length)

        if self._categorical:
            xs = keras.utils.np_utils.to_categorical(xs, num_classes=len(self._dictionary))
        else:
            # xs = xs / float(len(self._dictionary))
            pass

        parsed = [self._parse('\n'.join(ex['text'].split('\n')[-1:])) for ex in exs]
        qs = self._transform_input_(parsed, max_len=self._query_length)

        if self._categorical:
            qs = keras.utils.np_utils.to_categorical(qs, num_classes=len(self._dictionary))
        else:
            # qs = qs / float(len(self._dictionary))
            pass

        ys = None
        cands = None

        if 'labels' in exs[0]:
            ys = [self._parse(ex['labels'])[0] for ex in exs]
            # ys = keras.utils.np_utils.to_categorical(self._transform_input_(parsed), num_classes=len(self._dictionary))

        if 'label_candidates' in exs[0]:
            cands = [self._parse(ex['label_candidates']) for ex in exs]
            # cands = self._transform_input_(parsed)

        return xs, qs, ys, cands,

    def predict(self, xs, qs, ys, cands):
        return ys, cands


class Networks:
    """
    Base class for Neural Networks
    """

    def __init__(self):
        self._model = None
        self._epochs = 200
        self._verbose = True

    def train(self, data, valid=None):
        """
        Train models with neural network inputs "story" and "question" with the expected result "answer"

        :param story: Story and context
        :param question: Query made to find the answer
        """
        nn_input = self._network_input_(data.context, data.query)
        for epoch in range(self._epochs):
            self._model.fit(nn_input, data.answer, verbose=self._verbose, callbacks=[callback], validation_data=([valid.context, valid.query], valid.answer))

    def predict(self, story, question):

        nn_input = self._network_input_(story, question)
        return self._model.predict(nn_input, verbose=False)

    def _network_input_(self, story, question):
        """
        Format how the network will receive the inputs
        :param story: Values from history
        :param question: Values from questions

        :return: Expected format from keras models
        """
        return [story, question]


class BaseKerasAgent(BaseNeuralNetworkAgent):
    """
    Super class of Keras Agents. This class have all keras pypagAI.agents common methods
    """

    @staticmethod
    def add_cmdline_args(parser):
        BaseNeuralNetworkAgent.add_cmdline_args(parser)

        agent = parser.add_argument_group('Keras Arguments')
        agent.add_argument('-kv', '--keras-verbose', type=bool, default=False, help='Keras verbose outputs')
        agent.add_argument('-kbs', '--keras-batch-size', type=int, default=32, help='Batch size of keras')
        agent.add_argument('-ke', '--keras-epochs', type=int, default=200, help='Keras number of epochs')
        agent.add_argument('-kefd', '--exclude-from-dict', type=int, default=5,
                           help='Number of responses excluded from dictionary. This option get no effect when option '
                                'use_candidates is True')

    def __init__(self, opt):
        super().__init__(opt)
        self._model = None
        self._verbose = opt['keras_verbose']
        self._epoch = opt['keras_epochs']
        # self._batch_size = opt['keras_batch_size']
        self._batch_size = 2
        self._exclude_from_dict = opt['exclude_from_dict']

    def save(self, path=None):
        """
        Persist model on disk

        :param path: model file location, if it is none it will load the
        """
        path = self._model_file if path is None else path
        self._model._model.save(path)

    def load(self, path):
        """Return opt and model states."""
        self._model._model = keras.models.load_model(path)

        return self.opt, self._model

    def copy(self):
        return copy.deepcopy(self)

    def predict(self, xs, qs, ys=None, candidates=None):
        # TODO: candidates are not implemented yet

        if ys is not None:
            predictions = self._model.predict(xs, qs)
            self._model.train(xs, qs, ys)
        else:
            predictions = self._model.predict(xs, qs)

        return predictions, candidates
