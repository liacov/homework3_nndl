import os
import re
import json
import torch
import itertools
import numpy as np
from torch import nn
from time import time
from modules.dataset import OneHotEncode


class Network(nn.Module):

    # Constructor
    def __init__(
        self, vocab_size, embedding_dim, hidden_units, layers_num,
        hidden_type='LSTM', trained_embeddings=None, freeze_embeddings=False,
        dropout_prob=0.3
    ):
        # Call parent constructor
        super().__init__()

                # Initialize hidden layer class
        rnn = self.get_recurrent(hidden_type)

        # Define recurrent layer
        self.rnn = rnn(
            # Define size of the one-hot-encoded input
            input_size=embedding_dim,
            # Define size of a single recurrent hidden layer
            hidden_size=hidden_units,
            # Define number of stacked recurrent hidden layers
            num_layers=layers_num,
            # Set dropout probability
            dropout=dropout_prob,
            # Set batch size as first dimension
            batch_first=True
        )

        # Define output layer
        self.out = nn.Linear(hidden_units, vocab_size)
        # self.act = nn.LogSoftmax(dim=0)

        # Save architecture
        self.input_size = embedding_dim
        self.hidden_units = hidden_units
        self.layers_num = layers_num
        self.hidden_type = hidden_type
        self.dropout_prob = dropout_prob

        # Case pretrained embedding layer has been defined
        if trained_embeddings != None:
            # Load embedding layer from pretrained
            self.embed = nn.Embedding.from_pretrained(
                embeddings=trained_embeddings,
                freeze=freeze_embeddings
            )

            # Check that given pre-trained embeddings and given sizes match
            if (vocab_size, embedding_dim) != trained_embeddings.shape:
                # Raise new exception
                raise ValueError(' '.join([
                    'Error: given pre-trained embeddings',
                    'have shape {}'.format(trained_embeddings.shape),
                    'while the expected shape',
                    'is {}'.format((vocab_size, embedding_dim))
                ]))

        # Otherwise, initialize new embedding layer
        else:
            # Define embedding layer
            self.embed = nn.Embedding(vocab_size, embedding_dim)

        # Store vocabulary size
        self.vocab_size = vocab_size
        # Store embedding dimension
        self.embedding_dim = embedding_dim

    # Retrieve hidden layer class
    def get_recurrent(self, hidden_type):

        # Case hidden type is LSTM
        if hidden_type == 'LSTM':
            return nn.LSTM

        # Case hidden type is GRU
        if hidden_type == 'GRU':
            return nn.GRU


    def forward(self, x, state=None):

        # Go through embedding layer first
        x = self.embed(x)
        # Recurrent (hidden) layer output
        x, state = self.rnn(x, state)
        # Decision (linear) layer output
        x = self.out(x)
        # Apply LogSoftmax
        # x = self.act(x)
        # Return both new x and state
        return x, state

    def train_batch(self, batch, loss_fn, optimizer):

        # Extract target and input
        target = batch[:, -1]
        input = batch[:, :-1]
        # Eventually clear previous recorded gradients
        optimizer.zero_grad()
        # Make forward pass
        output, _ = self(input)
        # Evaluate loss only for last output
        loss = loss_fn(output[:, -1, :], target)
        # Backward pass
        loss.backward()
        # Update
        optimizer.step()
        # Return average batch loss
        return float(loss.data)

    def test_batch(self, batch, loss_fn):

        # Extract target and input
        target = batch[:, -1]
        input = batch[:, :-1]
        # Make forward pass
        output, _ = self(input)
        # Evaluate loss only for last output
        loss = loss_fn(output[:, -1, :], target)
        # Return average batch loss
        return float(loss.data)
