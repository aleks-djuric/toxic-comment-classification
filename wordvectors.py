# By Aleksandar Djuric
# Created February 14, 2018
# Last edited February 22, 2018

import fasttext as ft
import numpy as np
import pickle

from sklearn.neighbors import BallTree

class FastTextWrapper:
    """A wrapper class for the fasttext word vector library"""

    def __init__(self, data_dir="../data"):
        self._ft_dir = data_dir + "/fasttext/"
        self._nearestNeighborTree = None

        self.model = None
        self.word_vectors = None
        self.vocab = None


    def train_model(self, train_txt, vector_len=100):
        """Trains a new fasttext model"""

        print("Creating new fasttext model...")
        output_path = self._ft_dir + "ft_model"
        self.model = ft.skipgram(train_txt, output_path, dim=vector_len)
        self._retrieve_word_embeddings()
        print("Done")


    def load_model(self):
        """Loads a previously created fasttext model"""

        print("Loading fasttext model...")
        path = self._ft_dir + "ft_model.bin"
        self.model = ft.load_model(path, encoding='utf-8')
        self._retrieve_word_embeddings()
        print("Done")


    def _retrieve_word_embeddings(self):
        """Retrives vocab of embedded words and their vectors"""

        # Open model and retrieve word list + vectors
        path = self._ft_dir + "ft_model.vec"
        with open(path, 'r') as f:
            self.word_vectors = []
            self.vocab = []
            for line in f:
                line = line.split()
                self.vocab.append(line[0])
                self.word_vectors.append(np.array([float(i) for i in line[1:]]))
            self.word_vectors = self.word_vectors[1:]
            self.vocab = self.vocab[1:]


    def create_embeddings_dict(self, file_name='ft_embeddings_dict'):
        """Creates a dictionary of words and their
           corresponding vector embeddings"""

        if(self.model == None):
            print("Please load or create a model")
            return

        embeddings_dict = dict(zip(self.vocab, self.word_vectors))
        self._save_obj(embeddings_dict, file_name)


    def _save_obj(self, obj, name):
        """Saves a python object using the pickle library"""

        with open(self._ft_dir + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    def get_word_vector(self, word):
        """Returns word vectors for in and out-of-vocab words"""

        if(self.model == None):
            print("Please load or create a model")
            return

        return self.model[word]


    def lookup_embeddings(self, input_string):
        """Returns string in word vector form"""

        if(self.model == None):
            print("Please load or create a model")
            return

        string_embedded = []
        for word in input_string:
            string_embedded.append(self.model[word])

        return string_embedded


    def find_nn(self, query_word, num_neighbors=10):
        """Returns nearest neighbours and their distances
           for a given query word"""

        # Create nearest neighbors tree from word_vectors
        if self._nearestNeighborTree == None:
            self._create_nn_tree()

        # Translate query word to 1x100 vector
        query = np.array(self.model[query_word])
        query = np.reshape(query,(1,100))

        # Query ball tree for nearest neighbors
        dist, idx = self._nearestNeighborTree.query(query, k=num_neighbors)
        dist = dist[0]
        idx = idx[0]

        # Return nearest neighbors and euclidean distance
        nns = []
        for i in range(num_neighbors):
            nns.append((self.vocab[idx[i]], dist[i]))
        return nns


    def _create_nn_tree(self):
        """Creates a k-nearest neighbor ball tree"""

        self._nearestNeighborTree = BallTree(self.word_vectors)


    def _load_obj(self, name):
        """Loads a saved python object using the pickle library"""

        with open(self._ft_dir + name + '.pkl', 'rb') as f:
            return pickle.load(f)
