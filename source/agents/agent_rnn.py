import numpy as np
from keras.layers import Dense, Dropout, Input, LSTM, Lambda
from keras.layers import add, concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam

from source.agents.base import BaseKerasAgent, Networks


class RN(Networks):

    def __init__(self, vocab_size, story_maxlen, query_maxlen, max_sentences):
        """
        Implementation of Relational Neural Network:

        For the bAbI suite of tasks the natural language inputs must be transformed into a set of objects. This is a
        distinctly different requirement from visual QA, where objects were defined as spatially distinct regions in
        convolved feature maps. So, we first identified up to 20 sentences in the support set that were immediately
        prior to the probe question. Then, we tagged these sentences with labels indicating their relative position in
        the support set, and processed each sentence word-by-word with an LSTM (with the same LSTM acting on each
        sentence independently). We note that this setup invokes minimal prior knowledge, in that we delineate objects
        as sentences, whereas previous bAbI models processed all word tokens from all support sentences sequentially.
        It's unclear how much of an advantage this prior knowledge provides, since period punctuation also unambiguously
        delineates sentences for the token-by-token processing models.  The final state of the sentence-processing-LSTM
        is considered to be an object.  Similar to visual QA, a separate LSTM produced a question embedding, which was
        appended to each object pair as input to the RN. Our model was trained on the joint version of bAbI (all 20
        tasks simultaneously), using the full dataset of 10K examples per task

        For the bAbI task, each of the 20 sentences in the support set was processed through a 32 unit LSTM to produce
        an object.  For the RN,g was a four-layer MLP consisting of 256 units per layer.  For f , we used a three-layer
        MLP consisting of 256, 512, and 159 units, where the final layer was a linear layer that produced logits for a
        softmax over the answer vocabulary.  A separate LSTM with 32 units was used to process the question.
        The softmax output was optimized with a cross-entropy loss function using the Adam optimizer with a learning
        rate of 2e-4 .
        """

        super().__init__()
        self._vocab_size = vocab_size
        self._story_maxlen = story_maxlen
        self._query_maxlen = query_maxlen
        self._max_sentences = max_sentences

        # placeholders
        story = Input((self._story_maxlen,))
        question = Input((self._query_maxlen, 1))

        pred = self.create_network(story, question)

        optimizer = Adam(lr=2e-4)

        self._model = Model(inputs=[story, question], outputs=pred)
        self._model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def create_network(self, story, question):

        drop_out = 0.5
        hidden = 256
        lstm_units = 32

        question_encoded = LSTM(lstm_units)(question)

        responses = []
        for i in range(story.shape[1]):

            # input_i = story
            input_i = Lambda(lambda x: x[:, i])(story)
            input_i = Reshape((self._story_maxlen, 1))(input_i)
            story_encoded = LSTM(lstm_units)(input_i)
            response = concatenate([story_encoded, question_encoded])

            # # add the match matrix with the second input vector sequence
            g_response = Dense(hidden, activation='relu', input_dim=lstm_units)(response)
            g_response = Dense(hidden, activation='relu')(g_response)
            g_response = Dense(hidden, activation='relu')(g_response)
            g_response = Dense(hidden, activation='relu')(g_response)
            responses.append(g_response)

        responses = add(responses)

        f_response = Dense(hidden, activation='relu', input_dim=hidden)(responses)
        f_response = Dense(512, activation='relu')(f_response)
        f_response = Dropout(drop_out)(f_response)
        return Dense(self._vocab_size, activation='softmax')(f_response)



class RNAgent(BaseKerasAgent):

    @staticmethod
    def add_cmdline_args(parser):
        BaseKerasAgent.add_cmdline_args(parser)

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RNAgent'
        self.opt = opt
        self.episode_done = True

        self.__create_network__(opt)

    def __create_network__(self, opt):
        statement_size = 8
        self._vocab_size = len(self._dictionary)
        self._story_max_len = 1 * statement_size
        self._query_max_len = 1 * statement_size
        self._max_sentences = 20

        self._model = RN(self._vocab_size, self._query_max_len, self._story_max_len, self._max_sentences)

    def _rank_candidates_(self, obs):

        iterator = 0
        history = obs['text'].split('\n')
        inputs_train = np.zeros((self._max_sentences, self._story_max_len))
        for story in history[:len(history) - 1][len(history) - self._max_sentences if len(history) > self._max_sentences else 0:]:

            history_vec = self._dictionary.txt2vec(story)[:self._story_max_len]
            inputs_train[iterator][:len(history_vec)] = history_vec
            iterator += 1

        queries_train = np.array([0] * self._query_max_len)
        query_vec = self._dictionary.txt2vec(history[len(history) - 1])
        queries_train[len(queries_train) - len(query_vec):] = query_vec
        queries_train = np.array([queries_train])

        inputs_train = np.reshape(inputs_train, (1, self._story_max_len, len(inputs_train)))
        queries_train = np.reshape(queries_train, (len(queries_train), self._query_max_len, 1))

        if 'labels' in obs:
            answers_train = np.array([0.] * self._vocab_size)
            answers_train[self._dictionary.txt2vec(obs['labels'][0])[0]] = 1
            answers_train = np.array([answers_train])

            self._model.train(inputs_train, queries_train, answers_train)
            return [self._dictionary[np.argmax(answers_train)]]

        else:
            predicted = self._model.predict(inputs_train, queries_train)
            return [self._dictionary[int(index)] for index in np.argsort(predicted[0])[::-1][:5]]
