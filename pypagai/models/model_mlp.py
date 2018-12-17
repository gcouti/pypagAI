import numpy as np
import tensorflow as tf

from pypagai.models.base import TensorFlowModel


def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.variable_scope('gradient_noise'):
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn)


class MLP(TensorFlowModel):
    """
        Dynamic Memory Networks (March 2016 Version - https://arxiv.org/abs/1603.01417)

        Improved End-To-End version.

        Inspired on:
        https://www.oreilly.com/ideas/question-answering-with-tensorflow
        https://github.com/patrickachase/dynamic-memory-networks/blob/master/python/dynamic_memory_network.py
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

        self._cfg = model_cfg

    def _create_network_(self):
        self._answer = tf.placeholder(tf.float64, [None, 1], name='answer')
        self._story = tf.placeholder(tf.float64, [None, self._story_maxlen, self._sentences_maxlen], name='story')
        self._question = tf.placeholder(tf.float64, [None, self._query_maxlen], name='question')

        # Parameters
        learning_rate = 0.01

        # Construct model
        self._model = self.neural_net()

        # Define loss and optimizer
        labels = tf.reshape(self._answer, [-1, ])
        self._loss_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._model, labels=labels))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train_op = self._optimizer.minimize(self._loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        self._accuracy, _ = tf.metrics.accuracy(tf.argmax(self._model, 1), labels)

    def neural_net(self):
        ###############
        # Input Module
        ###############

        # Parameters
        learning_rate = 0.1
        num_steps = 500
        batch_size = 128
        display_step = 100

        # Network Parameters
        n_hidden_1 = 256  # 1st layer number of neurons
        n_hidden_2 = 256  # 2nd layer number of neurons
        num_input = 784  # MNIST data input (img shape: 28*28)
        num_classes = 10  # MNIST total classes (0-9 digits)

        # tf Graph input
        X = tf.placeholder("float", [None, num_input])
        Y = tf.placeholder("float", [None, num_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        # Create model
        def neural_net(x):
            # Hidden fully connected layer with 256 neurons
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            # Hidden fully connected layer with 256 neurons
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            # Output fully connected layer with a neuron for each class
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        # Construct model
        logits = neural_net(X)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            for step in range(1, num_steps + 1):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

            print("Optimization Finished!")

            return logits
