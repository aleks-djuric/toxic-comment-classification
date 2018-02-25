# By Aleksandar Djuric
# Created February 14, 2018

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell

from wordvectors import FastTextWrapper

class TCCLSTMModel:
    """The toxic comment classification lstm model"""

    def __init__(self, config, is_training):
        self._config = config
        self._is_training = is_training
        self.n_steps = config.num_steps
        self.n_inputs = config.emb_size
        self.n_classes = config.num_classes
        self.n_hidden = config.hidden_size

        self._build_rnn_graph()


    def _build_rnn_graph():
        input_ = tf.placeholder(tf.float32, [None, self.n_steps, self.emb_size],
            name = 'input')
        labels = tf.placeholder(tf.int64, [None, self.n_classes], name='labels')

        with tf.name_scope("LSTM") as scope:
            def single_cell():
                return DropoutWrapper(GRUCell(self.n_hidden),
                    output_keep_prob=self._config.keep_prob)

            cell = MultiRNNCell(
                [single_cell() for _ in range(self._config.num_layers)])
            initial_state = cell.zero_state(self._config.batch_size, tf.float32)

        output, state = tf.nn.dynamic_rnn(cell, input_, dtype=tf.float32)
        output_flattened = tf.reshape(output, [-1, n_hidden])

        with tf.name_scope("softmax") as scope:
            with tf.variable_scope("softmax_params"):
                softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
                softmax_b = tf.get_variable("softmax_b", [num_classes])
            logits = tf.nn.xw_plus_b(output_flattened, softmax_w, softmax_b)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits, labels, name="sigmoid")

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    def fit(self, input_, labels):


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
    batch_size = 20
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
    batch_size = 20
    emb_size = 100
    num_classes = 6


def main():
    # Get train data
    print("Loading data...")
    train_data = pd.read_csv("../data/train_clean.csv")
    labels = train_data.drop(
        ['Unnamed: 0','id','comment_text'], axis=1).values
    comments = train_data['comment_text'].values

    # Lookup word vector embeddings for comments
    wv = FastTextWrapper()
    wv.load_model()
    comments_embedded =[]
    print("Looking up embeddings for comments...")
    for comment in comments:
        comments_embedded.append(wv.lookup_embeddings(comment))

    # Split into training and validation data
    train_valid_split = np.random.rand(len(comments_embedded)) < 0.80
    train_x = comments_embedded[train_test_split]
    train_y = labels[train_test_split]
    valid_x = comments_embedded[~train_test_split]
    valid_y = labels[~train_test_split]

    config = SmallConfig()
    model = TCCLSTMModel(config, is_training=True)


if __name__ == '__main__':
    main()
