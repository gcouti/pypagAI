from keras import Model, Input
from keras.layers import RNN, Embedding, Dropout, RepeatVector, add, Dense

from pypagai.models.base import TensorFlowModel


class RNNModel(TensorFlowModel):

    ALIAS = "rnn"

    """
    Keras implementation of RNN.

    Original source: https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py

    Trains two recurrent neural networks based upon a story and a question.
    The resulting merged vector is then queried to answer a range of bAbI tasks.
    The results are comparable to those for an LSTM model provided in Weston et al.:
    "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
    http://arxiv.org/abs/1502.05698

    Task Number                  | FB LSTM Baseline | Keras QA
    ---                          | ---              | ---
    QA1 - Single Supporting Fact | 50               | 100.0
    QA2 - Two Supporting Facts   | 20               | 50.0
    QA3 - Three Supporting Facts | 20               | 20.5
    QA4 - Two Arg. Relations     | 61               | 62.9
    QA5 - Three Arg. Relations   | 70               | 61.9
    QA6 - yes/No Questions       | 48               | 50.7
    QA7 - Counting               | 49               | 78.9
    QA8 - Lists/Sets             | 45               | 77.2
    QA9 - Simple Negation        | 64               | 64.0
    QA10 - Indefinite Knowledge  | 44               | 47.7
    QA11 - Basic Coreference     | 72               | 74.9
    QA12 - Conjunction           | 74               | 76.4
    QA13 - Compound Coreference  | 94               | 94.4
    QA14 - Time Reasoning        | 27               | 34.8
    QA15 - Basic Deduction       | 21               | 32.4
    QA16 - Basic Induction       | 23               | 50.6
    QA17 - Positional Reasoning  | 51               | 49.1
    QA18 - Size Reasoning        | 52               | 90.8
    QA19 - Path Finding          | 8                | 9.0
    QA20 - Agent's Motivations   | 91               | 90.7

    For the resources related to the bAbI project, refer to: https://research.facebook.com/researchers/1543934539189348

    # Notes
    - With default word, sentence, and query vector sizes, the GRU model achieves:
      - 100% test accuracy on QA1 in 20 epochs (2 seconds per epoch on CPU)
      - 50% test accuracy on QA2 in 20 epochs (16 seconds per epoch on CPU)

    In comparison, the Facebook paper achieves 50% and 20% for the LSTM baseline.
    - The task does not traditionally parse the question separately. This likely
      improves accuracy and is a good example of merging two RNNs.
    - The word vector embeddings are not shared between the story and question RNNs.
    - See how the accuracy changes given 10,000 training samples (en-10k) instead
      of only 1000. 1000 was used in order to be comparable to the original paper.
    - Experiment with GRU, LSTM, and JZS1-3 as they give subtly different results.
    - The length and noise (i.e. 'useless' story components) impact the ability for
      LSTMs / GRUs to provide the correct answer. Given only the supporting facts,
      these RNNs can achieve 100% accuracy on many tasks. Memory networks and neural
      networks that use attentional processes can efficiently search through this
      noise to find the relevant statements, improving performance substantially.
      This becomes especially obvious on QA2 and QA3, both far longer than QA1.
    """

    def __init__(self, arg_parser):
        super().__init__(arg_parser)
        args = arg_parser.add_argument_group(__name__)
        args.add_argument('--hidden', type=int, default=32)

        args = arg_parser.parse()

        EMBED_HIDDEN_SIZE = args.hidden

        story_maxlen = self._story_maxlen
        query_maxlen = self._query_maxlen
        vocab_size = self._vocab_size

        sentence = Input(shape=(story_maxlen,), dtype='int32')
        encoded_sentence = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
        encoded_sentence = Dropout(0.3)(encoded_sentence)

        question = Input(shape=(query_maxlen,), dtype='int32')
        encoded_question = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
        encoded_question = Dropout(0.3)(encoded_question)
        encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
        encoded_question = RepeatVector(story_maxlen)(encoded_question)

        merged = add([encoded_sentence, encoded_question])
        merged = RNN(EMBED_HIDDEN_SIZE)(merged)
        merged = Dropout(0.3)(merged)
        preds = Dense(vocab_size, activation='softmax')(merged)

        model = Model([sentence, question], preds)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
