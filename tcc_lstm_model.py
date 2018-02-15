# By Aleksandar Djuric
# Created February 14, 2018


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib.rnn import LSTMCell

class ToxicCommentClassification:
    
    def __init__(self, data, target, dropout, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize


def main():

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
