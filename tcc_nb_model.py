# By Aleksandar Djuric
# Created on February 23, 2018
# Last edited February 23, 2018

import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk import FreqDist

class TCCNBModel:
    """The toxic comment classification lstm model"""

    def __init__(self, vocab_size=10000):
        self._model = OneVsRestClassifier(MultinomialNB())
        self._vocab_size = vocab_size
        self._vocab = None

    def fit(self, input_, labels):
        print("\tCreating vocab...")
        self._vocab = self._create_vocab(input_)
        print("\tBuilding feature counts...")
        cv = CountVectorizer(vocabulary=self._vocab)
        feature_counts = cv.transform(input_)
        print("\tFitting model...")
        self._model.fit(feature_counts, labels)


    def predict(self, input_):
        print("\tBuilding feature counts...")
        cv = CountVectorizer(vocabulary=self._vocab)
        feature_counts = cv.transform(input_)
        print("\tPredicting...")
        return self._model.predict_proba(feature_counts)


    def _create_vocab(self, input_):
        word_dist = FreqDist()
        for comment in input_:
            word_dist.update(comment.split())
        vocab = [word for word,count in word_dist.most_common(self._vocab_size)]
        return dict(zip(vocab, range(self._vocab_size)))


def main():
    # Get train data
    print("Loading train data...")
    train_data = pd.read_csv("../data/train_clean.csv")
    print(train_data.isnull().sum())
    train_data = train_data.dropna(axis=0, how='any')
    train_comments = train_data['comment_text'].values
    labels = train_data.drop(
        ['Unnamed: 0','id','comment_text'], axis=1).values

    # Train model
    print("Training naive bayes model...")
    model = TCCNBModel()
    model.fit(train_comments, labels)

    # Get test data
    print("Loading test data...")
    test_data = pd.read_csv("../data/test_clean.csv")
    print(test_data.isnull().sum())
    test_comments = test_data['comment_text'].values

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_comments)

    # Create submission file
    print("Creating submission file...")
    columns = ['id','toxic','severe_toxic','obscene','threat','insult',
        'identity_hate']
    results = pd.DataFrame(
        np.column_stack([test_data['id'].values, predictions]),columns=columns)
    results.to_csv("../data/nb_submission.csv", index=False)

    print("Done")


if __name__ == '__main__':
    main()
