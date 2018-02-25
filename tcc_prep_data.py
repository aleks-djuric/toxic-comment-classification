# By Aleksandar Djuric
# Created on February 15, 2018
# Last edited February 23, 2018

import nltk, re
import pandas as pd
import numpy as np
import pickle

from wordvectors import FastTextWrapper
from datacleaner import DataCleaner

# Load test and train data
print("Loading data...")
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

# Drop NaN values from data
train_data = train_data.dropna(axis=0, how='any')
test_data = test_data.dropna(axis=0, how='any')

# Clean comments of special characters and stopwords
print("Cleaning special characters and stopwords...")
dc = DataCleaner()
train_comments_clean = np.array(
    [dc.clean_string(comment, stem_words=True)
    for comment in train_data['comment_text']])
test_comments_clean = np.array(
    [dc.clean_string(comment, stem_words=True)
    for comment in test_data['comment_text']])

# Replace comments with cleaned comments
train_data['comment_text'] = train_comments_clean
test_data['comment_text'] = test_comments_clean

# Save cleaned data
print("Saving cleaned data...")
clean_train_file = "../data/train_stemmed.csv"
clean_test_file = "../data/test_stemmed.csv"
train_data.to_csv(clean_train_file)
test_data.to_csv(clean_test_file)

raise SystemExit(0)

# Combine into single text document
print("Creating fasttext training txt...")
clean_data = np.concatenate(
    (train_data['comment_text'], test_data['comment_text']),axis=0)
# Save to a text file
np.savetxt("../data/ft_train_data.txt", clean_data, fmt='%s')

# Create fasttext model
wv = FastTextWrapper("../data")
wv.train_model("../data/ft_train_data.txt", vector_len=100)
