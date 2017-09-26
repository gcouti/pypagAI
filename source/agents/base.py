import copy

import keras
callback = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)


import numpy as np
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent


class BaseNeuralNetworkAgent(Agent):
    """
    Super class that can be used by all types of neural network agents.
    """

    @staticmethod
    def add_cmdline_args(parser):
        DictionaryAgent.add_cmdline_args(parser)

        agent = parser.add_argument_group('Neural Networks Arguments')
        agent.add_argument('-tm', '--text-max-size', type=int, default=30, help='Text maximum size')
        agent.add_argument('-uc', '--use-candidates', type=bool, default=False, help='')

    def __init__(self, opt):
        super().__init__(opt)
        self._dictionary = DictionaryAgent(opt)
        self._episode_done = True
        self._text_max_size = opt['text_max_size']
        self._model_file = self.opt.get('model_file', None)
        self._use_candidates = self.opt.get('use_candidates')

    def _parse(self, texts):
        if type(texts) == str:
            return self._dictionary.txt2vec(texts)
        else:
            result = []
            for t in texts:
                result.append(self._dictionary.txt2vec(t)[0])
            return result

    def reset(self):
        super().reset()
        self._episode_done = True

    def observe(self, obs):
        observation = copy.deepcopy(obs)
        if not self._episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self._episode_done = observation['episode_done']
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

        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            curr = batch_reply[i]
            curr['text_candidates'] = [self._dictionary.vec2txt([c]) for c in cands[i]]
            if self._use_candidates:
                i = np.argmax(predictions[i][cands[i]])
                curr['text'] = curr['text_candidates'][i]
            else:
                curr['text'] = self._dictionary.vec2txt([np.argmax(predictions[i])])

        return batch_reply

    def _transform_input(self, tokenized, max_len=None):

        if not max_len:
            max_len = max([len(x) for x in tokenized])

        result = []
        for entry in tokenized:
            new_x = [0] * max_len
            new_x[:len(entry)] = entry[:max_len]
            result.append(new_x)

        return np.array(result)

    def batchify(self, observations):
        return None, None, None, None

    def predict(self, xs, qs, ys, cands):
        return ys, cands


class BaseKerasAgent(BaseNeuralNetworkAgent):

    @staticmethod
    def add_cmdline_args(parser):
        BaseNeuralNetworkAgent.add_cmdline_args(parser)

        agent = parser.add_argument_group('Keras Arguments')
        agent.add_argument('-kv', '--keras-verbose', type=bool, default=False, help='Keras verbose outputs')
        agent.add_argument('-kbs', '--keras-batch-size', type=int, default=1024, help='Batch size of keras')
        agent.add_argument('-ke', '--keras-epochs', type=int, default=1000, help='Keras number of epochs')
        agent.add_argument('-kefd', '--exclude-from-dict', type=int, default=5,
                           help='Number of responses excluded from dictionary. This option get no effect when option '
                                'use_candidates is True')

    def __init__(self, opt):
        super().__init__(opt)
        self._model = None
        self._verbose = opt['keras_verbose']
        self._epoch = opt['keras_epochs']
        self._batch_size = opt['keras_batch_size']
        self._exclude_from_dict = opt['exclude_from_dict']

    def save(self, path=None):
        """
        Save model into a file

        :param path: model file location, if it is none it will load the
        """
        path = self._model_file if path is None else path

        if path:
            model = {'opt': self.opt}
            # TODO: save keras model into file
            # with open(path, 'wb') as write:
            #     torch.save(model, write)

    @staticmethod
    def load(path):
        """Return opt and model states."""
        model = None
        with open(path, 'rb') as read:
            print(read)
            #TODO: load keras model

        return model['opt'], model

    """
    This class is used can be extend with any kind of Keras agent
    """
    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        # valid examples
        exs = [ex for ex in observations if 'text' in ex]

        # tokenize the text
        parsed = [self._parse('\n'.join(ex['text'].split('\n')[:-1])) for ex in exs]
        xs = self._transform_input(parsed, max_len=self._text_max_size)

        parsed = [self._parse('\n'.join(ex['text'].split('\n')[-1:])) for ex in exs]
        qs = self._transform_input(parsed, max_len=self._text_max_size)

        ys = None
        cands = None

        if 'labels' in exs[0]:
            parsed = [self._parse(ex['labels']) for ex in exs]
            ys = keras.utils.np_utils.to_categorical(self._transform_input(parsed), num_classes=len(self._dictionary))

        if 'label_candidates' in exs[0]:
            parsed = [self._parse(ex['label_candidates']) for ex in exs]
            cands = self._transform_input(parsed)

        return xs, qs, ys, cands,

    def predict(self, xs, qs, ys=None, cands=None):
        if ys is not None:
            self._model.fit(self.format_input(xs, qs), ys, verbose=self._verbose, batch_size=self._batch_size, epochs=self._epoch, callbacks=[callback])
            return ys, cands
        else:
            predictions = self._model.predict(self.format_input(xs, qs))
            return predictions, cands

    def format_input(self, xs, qs):
        """
        Format how the network will recieve the inputs
        :param xs: Values from history
        :param qs: Values from questions

        :return: Expected format from keras models
        """
        return [xs, qs]
