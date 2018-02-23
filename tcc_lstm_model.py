# By Aleksandar Djuric
# Created February 14, 2018

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib.rnn import LSTMCell
from wordvectors import FTWordVectors

TRAIN_DATA_FILE = "../data/train_clean.csv"
TEST_DATA_FILE = "../data/test_clean.csv"

class TCCLSTMInput(object):
    """Model input configuration"""

    def __init__(self):
        train_data = pd.read_csv(TRAIN_DATA_FILE)
        comments = train_data['comment_text'].values

        self.input = self._embedding_lookup(comments)
        self.targets = train_data.drop(
            ['Unnamed: 0','id','comment_text'], axis=1).values


class TCCLSTMModel:
    """The toxic comment classification lstm model"""

    def __init__(self, input_, config, is_training):
        self.input = input_
        self.config = config
        self.is_training = is_training


    def build_rnn_graph():

        cell = tf.contrib.rnn.LSTMBlockCell(
            self._num_hidden, forget_bias=0.0, state_is_tuple=True,
            reuse=not is_training)

        dropout_cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=self.dropout_keep_prob)


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20


class LargeConfig(object):
  """Large config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 800
  max_epoch = 10
  max_max_epoch = 45
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


def main():

    config = {

    }
    # LSTM model parameters
    embed_size = 50
    max_seq_len = 400
    batchSize = 24
    lstmUnits = 64
    numClasses = 7
    iterations = 10000


    # labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    # input_data = tf.placeholder(tf.float32, [batchSize, max_seq_len])
    #
    # data = tf.Variable(tf.zeros([batchSize, max_seq_len, embed_size], dtype='float32'))
    # data = tf.nn.embedding_lookup(word_vectors, input_data)
    #
    # lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    # value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    #with tf.Session() as sess:

if __name__ == '__main__':
    main()
