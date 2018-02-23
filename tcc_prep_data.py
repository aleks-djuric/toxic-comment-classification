# By Aleksandar Djuric
# Created on February 15, 2018

import nltk, re
import pandas as pd
import numpy as np
import pickle

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from wordvectors import FTWordVectors

global wv_model

def clean_string(str, remove_stopwords=True, stem_words=False):
    # Remove special characters
    strip_special_chars = re.compile('[^A-Za-z ]+')
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


def clean_data(train_file, test_file):
    print("Cleaning data...")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    # Drop NaN values from data
    train_data = train_data.dropna(axis=0, how='any')
    test_data = test_data.dropna(axis=0, how='any')
    # Clean comments of special characters and stopwords
    train_comments_clean = np.array(
        [clean_string(comment) for comment in train_data['comment_text']])
    test_comments_clean = np.array(
        [clean_string(comment) for comment in test_data['comment_text']])
    # Replace comments with cleaned comments
    train_data['comment_text'] = train_comments_clean
    test_data['comment_text'] = test_comments_clean
    # Save cleaned data
    train_data.to_csv('../data/train_clean.csv')
    test_data.to_csv('../data/test_clean.csv')
    print("Done")


def prep_wv_train_data():
    train_data = pd.read_csv('../data/train_clean.csv')
    test_data = pd.read_csv('../data/test_clean.csv')
    # Combine into single text document
    clean_data = np.concatenate(
        (train_data['comment_text'],test_data['comment_text']),axis=0)
    # Save to a text file
    np.savetxt("../data/wv_train_data.txt", clean_data, fmt='%s')





def main():

    # Get word vector model
    wv_model = FTWordVectors()
    if():
        clean_data("../data/train.csv", "../data/test.csv")
    if():
        prep_wv_train_data()
    if():
        wv_model.train("../data/wv_train_data.txt")
    else:
        wv_model.load()

    wv_model.create_embeddings();
    save_obj(wv_model.embeddings, 'word_embeddings')


if __name__ == '__main__':
    main()
