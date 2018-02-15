# By Aleksandar Djuric
# Created February 14, 2018

import fasttext as ft
import numpy as np

from sklearn.neighbors import BallTree

class FTWordVectors:

    def __init__(self):
        self.model
        self.vectors
        self.words


    def create_embeddings(self):
        # Open model and retrieve word list + vectors
        with open("../data/ft_model.vec", 'r') as f:
            vectors = []
            words = []
            for line in f:
                line = line.split()
                words.append(line[0])
                vectors.append(np.array([float(i) for i in line[1:]]))
        self.vectors = np.array(vectors[1:])
        self.words = np.array(words[1:])


    def create(self, train_file):
        print("Creating fasttext new model...")
        self.model = ft.skipgram(train_file, "../data/ft_model")
        self.create_embeddings()
        print("Done")


    def load(self):
        self.model = ft.load_model("../data/ft_model.bin", encoding='utf-8')
        self.create_embeddings()
        print("Fasttext model loaded")


    def find_nn(self, query_word, num_neighbors):
        # Translate query word to 1x100 vector
        query = np.array(self.model[query_word])
        query = np.reshape(query,(1,100))
        # Create nearest neighbours tree
        self.bt = BallTree(self.vectors)
        dist, idx = bt.query(query, k=num_neighbors)
        dist = dist[0]
        idx = idx[0]
        # Return nearest neighbours and euclidean distance
        nns = []
        for i in range(num_neighbors):
            nns.append((self.words[idx[i]],dist[i]))
        return nns
