from keras import Input, Model, Sequential
from keras.layers import Embedding, Dropout, dot, Activation, Permute, add, concatenate, SimpleRNN, Dense, LSTM

from pypagai.models.base import KerasModel


class N2NMemory(KerasModel):

    def __init__(self, model_cfg):

        super().__init__(model_cfg)
        """
        Keras implementation of: 
        
        - Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
          "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
          http://arxiv.org/abs/1502.05698

        - Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
          "End-To-End Memory Networks",
          http://arxiv.org/abs/1503.08895

        original source: https://github.com/keras-team/keras/blob/master/examples/babi_memnn.py
        
        Description of network:        
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

        # placeholders
        input_sequence = Input((self._story_maxlen, ))
        question = Input((self._query_maxlen,))

        answer = self.create_network(input_sequence, question, model_cfg)

        # build the final model
        self._model = Model([input_sequence, question], answer)
        self._model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['dropout'] = 0.3
        config['activation'] = 'softmax'
        config['samples'] = 32
        config['embedding'] = 64

        return config

    def create_network(self, input_sequence, question, cfg):

        drop_out = cfg['dropout']
        activation = cfg['activation']
        samples = cfg['samples']
        embedding = cfg['embedding']

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
        answer = LSTM(samples)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(drop_out)(answer)
        answer = Dense(self._vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary

        return Activation(activation)(answer)
