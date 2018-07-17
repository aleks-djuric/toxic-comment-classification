# A tensorflow implementation of an lstm model for toxic comment classification
# By Aleksandar Djuric

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell

from wordvectors import FastTextWrapper
from datetime import datetime
from time import time
from os import path

class TCCLSTMModel:
    """The toxic comment classification lstm model"""

    def __init__(self, config, is_training):
        self._config = config
        self._is_training = is_training
        self.n_steps = config.num_steps
        self.n_inputs = config.emb_size
        self.n_classes = config.num_classes
        self.n_hidden = config.hidden_size
        self.learning_rate = config.learning_rate

        self._build_rnn_graph()


    def _build_rnn_graph(self):
        input_ = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs],
            name = 'input')
        labels = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')

        with tf.name_scope("LSTM") as scope:
            def single_cell():
                return DropoutWrapper(GRUCell(self.n_hidden),
                    output_keep_prob=self._config.keep_prob)

            multi_layer_cell = MultiRNNCell(
                [single_cell() for _ in range(self._config.num_layers)])
            initial_state = multi_layer_cell.zero_state(self._config.batch_size, tf.float32)

        output, state = tf.nn.dynamic_rnn(
            multi_layer_cell, input_, dtype=tf.float32)
        output_flattened = tf.reshape(output, [-1, self.n_hidden])

        with tf.name_scope("softmax") as scope:
            with tf.variable_scope("softmax_params"):
                softmax_w = tf.get_variable("softmax_w", [self.n_hidden, self.n_classes])
                softmax_b = tf.get_variable("softmax_b", [self.n_classes])
            logits = tf.nn.xw_plus_b(output_flattened, softmax_w, softmax_b)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits, name="sigmoid")

        # Minimize loss using Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss)

        # Make predictions
        correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(logits)), labels)

        # Accuracy of individual classes being correct
        self.acc_of_class = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        # Accuracy of entire label being correct
        all_labels_true = tf.reduce_min(
            tf.cast(correct_prediction, tf.float32), 1)
        self.acc_of_label = tf.reduce_mean(all_labels_true)


    def fit(self, input_, labels):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            batches = range(0, tf.shape(input_)[0], self._config.batch_size)
            n_batches = tf.shape(input_)[0] // self._config.batch_size + 1
            for epoch in range(self._config.max_epoch):
                for batch, start in enumerate(batches):
                    end = start + self._config.batch_size
                    batch_x = input_[start:end]
                    batch_y = labels[start:end]

                    sess.run(self.optimizer,
                        feed_dict={input_: batch_x, labels: batch_y})

                    if batch%10000 == 0:
                        acc_class = sess.run(self.acc_of_class,
                            feed_dict={input_: batch_x, labels: batch_y})
                        acc_label = sess.run(self.acc_of_label,
                            feed_dict={input_: batch_x, labels: batch_y})
                        print("Epoch: {}, Batch number: {}".format(epoch, batch))
                        print("Class accuracy: {0:.2f}".format(acc_class))
                        print("Label accuracy: {0:.2f}".format(acc_label))
                        print("{0:.2f} % done".format(100 * (batch*epoch) /
                            (n_batches*self._config.max_epoch)))
                        print("__________\n")

            ts = time()
            st = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
            saver.save(sess, "../data/models/" + st + ".ckpt")


    def validate(self, input_):
        with tf.Session() as sess:
            print("")


    def predict(self, input_):
        with tf.Session() as sess:
            print("")


def main():
    # Get train data
    print("Loading train data...")
    train_comments_embedded_path = "../data/train_comments_embedded.npy"
    train_labels_path = "../data/train_labels.npy"
    if path.isfile(train_comments_embedded_path) and path.isfile(train_labels_path):
        comments_embedded = np.load(train_comments_embedded_path)
        labels = np.load(train_labels_path)
    else:
        train_data = pd.read_csv("../data/train_clean.csv")
        train_data = train_data.dropna(axis=0, how='any')
        labels = train_data.drop(
            ['Unnamed: 0','id','comment_text'], axis=1).values
        comments = train_data['comment_text'].values

        # Lookup word vector embeddings for train comments
        wv = FastTextWrapper()
        wv.load_model()
        comments_embedded = wv.get_embeddings(comments)

        # Save embeddings and labels
        comments_embedded = np.array(comments_embedded)
        np.save(train_comments_embedded_path, comments_embedded)
        labels = np.array(labels)
        np.save(train_labels_path, labels)


    # Split into training and validation data
    train_valid_split = np.random.rand(len(comments_embedded)) < 0.80
    train_x = comments_embedded[train_valid_split]
    train_y = labels[train_valid_split]
    valid_x = comments_embedded[~train_valid_split]
    valid_y = labels[~train_valid_split]

    # Initialize model
    print("Initializing model...")
    config = SmallConfig()
    model = TCCLSTMModel(config, is_training=True)

    # Train model
    print("Training model...")
    model.fit(train_x, train_y)

    # Validate model
    print("Validating model...")
    model.validate(valid_x, valid_y)

    # Get test data
    print("Loading test data...")
    test_data = pd.read_csv("../data/test_clean.csv")
    comments = train_data['comment_text'].values

    # Lookup word vector embeddings for test comments
    comments_embedded = wv.get_embeddings(comments)

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(comments_embedded)

    # Create submission file
    print("Creating submission file...")
    columns = ['id','toxic','severe_toxic','obscene','threat','insult',
        'identity_hate']
    results = pd.DataFrame(
        np.column_stack([test_data['id'].values, predictions]),columns=columns)
    results.to_csv("../data/lstm_submission.csv", index=False)


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 256
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 0.5
    lr_decay = 0.5
    batch_size = 16
    emb_size = 100
    num_classes = 6


class LargeConfig(object):
    """Large config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 1024
    max_epoch = 10
    max_max_epoch = 45
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 16
    emb_size = 100
    num_classes = 6


if __name__ == '__main__':
    main()
