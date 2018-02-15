import nltk, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load
word_list = np.load('../data/word_list.npy').tolist()
word_vectors = np.load('../data/word_vectors.npy')
print("Loaded word embeddings.")

train_data = np.load('../data/train_data.npy')
test_data = np.load('../data/test_data.npy')
print("Loaded train and test data.")

train_labels = np.load('../data/train_labels.npy')
test_labels = np.load('../data/test_labels.npy')
print("Loaded train and test data.")

# LSTM Model
embed_size = 50
max_seq_len = 400
batchSize = 24
lstmUnits = 64
numClasses = 7
iterations = 10000

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.float32, [batchSize, max_seq_len])

data = tf.Variable(tf.zeros([batchSize, max_seq_len, embed_size], dtype='float32'))
data = tf.nn.embedding_lookup(word_vectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

#with tf.Session() as sess:
