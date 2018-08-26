import tensorflow as tf
from pypagai.models.base import TensorFlowModel


class DMN(TensorFlowModel):
    """
        Dynamic Memory Networks (March 2016 Version - https://arxiv.org/abs/1603.01417)

        Improved End-To-End version.


        Inspired on: https://github.com/patrickachase/dynamic-memory-networks/blob/master/python/dynamic_memory_network.py
        https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_basic.py
    """

    @staticmethod
    def default_config():
        config = TensorFlowModel.default_config()
        config['hidden'] = 50

        return config

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        """

        :param model_cfg:
        """

        self._story = tf.placeholder("float", [None, self._story_maxlen], name='story')
        self._question = tf.placeholder("float", [None, self._query_maxlen], name='question')
        self._answer = tf.placeholder("int64", [None, 1], name='answer')

        # Parameters
        learning_rate = 0.01

        # Construct model
        self._model = self.neural_net()

        # Define loss and optimizer
        labels = tf.reshape(self._answer, [-1, ])
        self._loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._model, labels=labels))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train_op = self._optimizer.minimize(self._loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        arg = tf.argmax(self._model, 1)
        self._accuracy, _ = tf.metrics.accuracy(arg, labels)

    def neural_net(self):

        l1 = tf.layers.dense(self._story, 128)
        l2 = tf.layers.dense(l1, 32)

        return tf.layers.dense(l2, self._vocab_size)
