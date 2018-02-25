# By Aleksandar Djuric
# Created February 22, 2018
# Last edited February 22, 2018

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

class DataCleaner(object):

    def clean_string(self, string_, char_to_keep='[^A-Za-z ]+',
        remove_stopwords=True, stem_words=False):

        # Remove special characters
        strip_special_chars = re.compile(char_to_keep)
        string_ = strip_special_chars.sub('', string_.lower())

        # Split string into words
        words = string_.split()

        # Optionally remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            words = [w for w in words if w not in stop_words]

        # Optionally stem word_list
        if stem_words:
            stemmer = SnowballStemmer('english')
            words = [stemmer.stem(word) for word in words]

        return ' '.join(words)
