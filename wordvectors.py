# Experimenting with fasttext

import os
import re
import fasttext as ft
import pandas as pd
import numpy as np

from sklearn.neighbors import BallTree
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class WordVectors:

    def __init__(self):
        # Check if a word vector model exists in data directory
        if(not os.path.isfile("../data/ft_model.bin")):
            train_data, test_data = load_data()
            self.word_vector_model = train_word_vectors(train_data, test_data)
        else:
            self.word_vector_model = ft.load_model("../data/ft_model.bin", encoding='utf-8')

        # Open saved model and retrieve word list + vectors
        with open("../data/ft_model.vec", 'r') as f:
            vectors = []
            words = []
            for line in f:
                line = line.split()
                words.append(line[0])
                vectors.append(np.array([float(i) for i in line[1:]]))
        self.vectors = np.array(vectors[1:])
        self.words = np.array(words[1:])

    def load_data(self):
        train_data = pd.read_csv("../data/train.csv")
        test_data = pd.read_csv("../data/test.csv")
        # Drop NaN values from data
        train_data = train_data.dropna(axis=0, how='any')
        test_data = test_data.dropna(axis=0, how='any')
        return train_data, test_data


    def clean_string(self, str, remove_stopwords=True, stem_words=False):
        # Remove special characters
        strip_special_chars = re.compile('[^A-Za-z0-9 ]+')
        str = strip_special_chars.sub('', str.lower())
        # Split string into words
        words = str.split()
        # Optionally remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            words = [w for w in words if w not in stop_words]
        # Optionally stem word_list
        if stem_words:
            stemmer = SnowballStemmer('english')
            words = [stemmer.stem(word) for word in words]

        return ' '.join(words)


    def train_word_vectors(self, train_data, test_data):
        # Clean comments of special characters and stopwords
        train_data_clean = np.array(
            [clean_string(comment) for comment in train_data['comment_text']])
        np.save('../data/train_comments_clean', train_data_clean)
        test_data_clean = np.array(
            [clean_string(comment) for comment in test_data['comment_text']])
        np.save('../data/test_comments_clean', test_data_clean)
        # Combine all text
        clean_data = np.concatenate((train_data_clean,test_data_clean),axis=0)
        # Save to a text file
        np.savetxt("../data/wv_train_data.txt", clean_data, fmt='%s')
        # Create word vector model
        model = ft.skipgram("../data/wv_train_data.txt", "../data/ft_model")

        return model


    def find_nn(self, query_word, num_neighbors):
        # Translate query word to 1x100 vector
        query = np.array(self.word_vector_model[query_word])
        query = np.reshape(query,(1,100))
        # Create nearest neighbours tree
        bt = BallTree(self.vectors)
        dist, idx = bt.query(query, k=num_neighbors)
        dist = dist[0]
        idx = idx[0]
        # Return nearest neighbours and euclidean distance
        nns = []
        for i in range(num_neighbors):
            nns.append((self.words[idx[i]],dist[i]))
        return nns
