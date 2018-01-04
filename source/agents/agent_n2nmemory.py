from keras import Input, Model, Sequential
from keras.layers import Embedding, Dropout, dot, Activation, Permute, add, concatenate, SimpleRNN, Dense

from source.agents.base import Networks, BaseKerasAgent, BaseNeuralNetworkAgent


class N2NMemory(Networks):

    def __init__(self, vocab_size, story_maxlen, query_maxlen):

        super().__init__()
        """
        10% of the bAbI training set was held-out to form a validation set, which was used to select the optimal
        model architecture and hyperparameters.  Our models were trained using a learning rate of = 0:01, with
        anneals every 25 epochs by=2 until 100 epochs were reached.  No momentum or weight decay was used.
        The weights were initialized randomly from a Gaussian distribution with zero mean and= 0:1 .  When trained
        on all tasks simultaneously with 1k training samples (10k training  samples),  60  epochs  (20  epochs)
        were  used  with  learning  rate  anneals  of=2 every  15 epochs (5 epochs).  All training uses a batch size
        of 32 (but cost is not averaged over a batch), and gradients with an`2 norm larger than 40 are divided by a
        scalar to have norm 40.  In some of our experiments, we explored commencing training with the softmax
        in each memory layer removed, making  the  model  entirely  linear  except  for  the  final  softmax  for
        answer  prediction.   When  the validation loss stopped decreasing, the softmax layers were re-inserted and
        training recommenced. We  refer  to  this  as  linear  start  (LS)  training.   In  LS  training,  the
        initial  learning  rate  is  set  to = 0:005 . The capacity of memory is restricted to the most recent
        50 sentences. Since the number of sentences and the number of words per sentence varied between problems,
        a null symbol was used to pad them all to a fixed size. The embedding of the null symbol was constrained to
        be zero.
        """

        self._story_maxlen = story_maxlen
        self._query_maxlen = query_maxlen
        self._vocab_size = vocab_size

        # placeholders
        input_sequence = Input((None, 128))
        question = Input((None, 128))

        answer = self.create_network(input_sequence, question)

        # build the final model
        self._model = Model([input_sequence, question], answer)
        self._model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def create_network(self, input_sequence, question):

        drop_out = 0.3
        activation = 'softmax'
        samples = 32
        embedding = 64

        # encoders
        # embed the input sequence into a sequence of vectors

        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=self._vocab_size, output_dim=embedding))
        input_encoder_m.add(Dropout(drop_out))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=self._vocab_size, output_dim=self._query_maxlen))
        input_encoder_c.add(Dropout(drop_out))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=self._vocab_size, output_dim=embedding, input_length=self._query_maxlen))
        question_encoder.add(Dropout(drop_out))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation(activation)(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = SimpleRNN(samples)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(drop_out)(answer)
        answer = Dense(self._vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary

        return Activation(activation)(answer)


class N2NMemAgent(BaseKerasAgent):
    """
    Python + Keras implementation of End-to-end Memory Network

    References:
        - Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
        "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
        http://arxiv.org/abs/1502.05698

        - Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
        "End-To-End Memory Networks",
        http://arxiv.org/abs/1503.08895

        Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
        Time per epoch: 3s on CPU (core i7).
    """
    @staticmethod
    def add_cmdline_args(parser):
        BaseNeuralNetworkAgent.add_cmdline_args(parser)
        parser.add_argument('-lp', '--length_penalty', default=0.5, help='length penalty for responses')

        agent = parser.add_argument_group('N2NMem Arguments')

        agent.add_argument('-hs', '--hiddensize', type=int, default=128, help='size of the hidden layers and embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2, help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.5, help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1, help='dropout rate')
        agent.add_argument('-r', '--rank-candidates', type='bool', default=False,
                           help='rank candidates if available. this is done by computing the' +
                                ' mean score per token for each candidate and selecting the ' +
                                'highest scoring one.')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'N2NMemAgent'
        self.opt = opt

        statement_size = 8
        self._vocab_size = len(self._dictionary)
        self._story_maxlen = 15 * statement_size
        self._query_maxlen = 1 * statement_size

        self._model = N2NMemory(self._vocab_size, self._story_maxlen, self._query_maxlen)

    def predict(self, xs, qs, ys=None, cands=None):
        self._model.train(xs, xs, ys)

        # def _rank_candidates(self, obs):
        #
        #     history = obs['text'].split('\n')
        #     inputs_train = np.array([0] * self._story_maxlen)
        #     history_vec = self._dictionary.txt2vec("\n".join(history[:len(history) - 1]))[:self._story_maxlen]
        #
        #     inputs_train[len(inputs_train) - len(history_vec):] = history_vec
        #
        #     queries_train = np.array([0] * self._query_maxlen)
        #     query_vec = self._dictionary.txt2vec(history[len(history) - 1])[:self._query_maxlen]
        #     queries_train[len(queries_train) - len(query_vec):] = query_vec
        #
        #     inputs_train = np.array([inputs_train])
        #     queries_train = np.array([queries_train])
        #
        #     if 'labels' in obs:
        #         answers_train = [0.] * self._vocab_size
        #         answers_train[self._dictionary.txt2vec(obs['labels'][0])[0]] = 1
        #
        #         self._model.train(inputs_train, queries_train, np.array([answers_train]))
        #
        #         return [self._dictionary[np.argmax(answers_train)]]
        #     else:
        #         predicted = self._model.predict(inputs_train, queries_train)
        #         return [self._dictionary[int(index)] for index in np.argsort(predicted[0])[::-1][:5]]
