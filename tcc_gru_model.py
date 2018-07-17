# Pytorch implementation of a GRU RNN model for toxic comment classification
# By Aleksandar Djuric

import os
import h5py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.utils.data import Dataset, DataLoader

from wordvectors import FastTextWrapper
from datetime import datetime
from time import time
from itertools import accumulate


class ToxicCommentClassifier(nn.Module):
    """A gru rnn model for toxic comment classification """

    def __init__(self, config):
        super(ToxicCommentClassifier, self).__init__()

        self.num_classes = config.num_classes
        self.hidden_size = config.hidden_size
        self.emb_size = config.emb_size
        self.num_layers = config.num_layers
        self.drop_prob = config.drop_prob

        self.gru = nn.GRU(self.emb_size, self.hidden_size, num_layers=self.num_layers)
        self.dropout = nn.Dropout(self.drop_prob)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def init_hidden(self, batch_size):
        return autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))

    def forward(self, batch, batch_size, seq_lengths):
        packed_batch = nn.utils.rnn.pack_padded_sequence(
            batch, seq_lengths, batch_first=True)
        hidden = self.init_hidden(batch_size)
        packed_output, _ = self.gru(packed_batch, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output[-1])
        output = self.linear(output)

        return output


class ToxicCommentsDataset(Dataset):

    def __init__(self, data_path, dataset_path, training=True, train_embeddings=False):

        # Check if embedded vectors file exists, otherwise create it
        if os.path.isfile(dataset_path):
            print("Loading data from npy file...")
            file = h5py.File(dataset_path, 'r')
            self.comments = file.get('comments').value
            self.comm_lens = file.get('comment_lengths').value
            self.word_vectors = file.get('word_vectors').value
            if training:
                self.labels = file.get('labels').value

        else:
            print("Loading data from csv...")
            data = pd.read_csv(data_path)
            # Drop incomplete rows
            data = data.dropna(axis=0, how='any')
            # Get the six label categories
            if training:
                self.labels = data.drop(
                    ['Unnamed: 0','id','comment_text'], axis=1).values
            # Get the comments
            comments = data['comment_text'].values

            # Helper class to create word vector embeddings using FastText
            wv = FastTextWrapper()

            # Create a new word vector model or load a previously created model
            if train_embeddings:
                wv.train_model('../data/wv_train_data.txt', vector_len=100)
            else:
                wv.load_model()

            self.comments, self.comm_lens = wv.get_embeddings(comments, max_seq_len=250)
            self.word_vectors = np.stack(wv.word_vectors)

            # Save data to h5py file
            print("Saving data to h5py file...")

            file = h5py.File(dataset_path, 'w')
            file.create_dataset('comments', data=self.comments, dtype=np.int64)
            file.create_dataset('comment_lengths', data=self.comm_lens, dtype=np.int16)
            file.create_dataset('word_vectors', data=self.word_vectors, dtype=np.float32)
            if training:
                file.create_dataset('labels', data=self.labels)
            file.close()

    def __getitem__(self, index):
        return self.comments[index], self.comm_lens[index], self.labels[index]

    def __len__(self):
        return len(self.comments)


def train(model, train_data, config):

    dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.step_size, gamma=config.lr_decay)
    embed = nn.Embedding.from_pretrained(torch.FloatTensor(train_data.word_vectors))

    for epoch in range(config.epochs):
        for i, (comments, comment_lengths, labels) in enumerate(dataloader):
            # Get sorted indices of comment_lengths
            sorted_idx = np.argsort(comment_lengths).tolist()[::-1]
            comment_lengths = comment_lengths[sorted_idx]
            embedded_comments = embed(torch.LongTensor(comments[sorted_idx]))
            labels = labels[sorted_idx]

            output = model(packed_comments, config., comment_lengths)
            print(output.size())
            print(labels.size())
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % 100) == 0:
                print(output)
                print("Batch {:d}: {:.10f}".format(i, loss))


def validate():
    pass


def test():
    pass


def main():

    model_path = "../data/models/gru_model.pt"

    if not os.path.isfile(model_path):
        print("Creating training dataset...")
        train_data = ToxicCommentsDataset(data_path="../data/train_clean.csv",
                                          dataset_path="../data/train_dataset.hdf5")

        print("Initializing model...")
        config = SmallConfig()
        model = ToxicCommentClassifier(config)

        print("Training model...")
        train(model, train_data, config)
        torch.save(model, model_path)

    else:
        print("Loading model...")
        model = torch.load(model_path)


    print("Creating testing dataset...")
    test_data = ToxicCommentsDataset(data_path="../data/test_clean.csv",
                                     dateset_path="../data/test_dataset.hdf5",
                                     training=False)

    print("Testing model...")
    test(model, test_data, config)

    # Create submission file
    print("Creating submission file...")
    columns = ['id','toxic','severe_toxic','obscene','threat','insult',
        'identity_hate']
    results = pd.DataFrame(
        np.column_stack([test_data['id'].values, predictions]),columns=columns)
    results.to_csv("../data/lstm_submission.csv", index=False)


class SmallConfig(object):
    """Small config."""
    num_classes = 6
    hidden_size = 256
    emb_size = 100
    seq_len = 250
    drop_prob = 0.2
    batch_size = 16
    num_layers = 1
    lr = 0.1
    step_size = 1
    lr_decay = 0.8
    epochs = 10


class LargeConfig(object):
    """Large config."""
    num_classes = 6
    hidden_size = 512
    emb_size = 100
    seq_len = 250
    drop_prob = 0.2
    batch_size = 32
    num_layers = 2
    lr = 0.1
    step_size = 1
    lr_decay = 0.8
    epochs = 20


if __name__ == '__main__':
    main()
