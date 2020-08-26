from time import time
from torch import nn
import numpy as np
import itertools
import torch
import json
import re
import os


class Network(nn.Module):

    # Constructor
    def __init__(
        self, vocab_size, embedding_dim, hidden_units, layers_num,
        hidden_type='LSTM', trained_embeddings=None, freeze_embeddings=False,
        dropout_prob=0.3
    ):
        """ Constructor

        Args
        vocab_size (int)                    Number of words in vocabulary
        embedding_dim (int)                 Number of attributes in each
                                            word vector
        input_size (int)                    Number of features in a single
                                            input vector
        hidden_units (int)                  Number of hidden units in a single
                                            recurrent hidden layer
        layers_num (int)                    Number of stacked hidden layers
        hidden_type (str)                   Type of hidden layer: LST or GRU
        trained_embeddings (torch.Float)    Pre-trained embeddings tensor
        freeze_embeddings (bool)            Wether embeddings have to be
                                            trained or not
        dropout_prob (float)                Probability for dropout, must be
                                            between [0, 1]

        Raise
        (ValueError)                        In case hidden type is not LSTM
                                            or GRU
        (ValueError)                        In case given embedding shape and
                                            pretrained embeddings does not
                                            coincide
        """
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
        self.out = nn.Linear(hidden_units, embedding_dim)

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
        """ Retrieve hidden layer class
        Args
        hidden_type (str)       Name of hidden layer class
        Return
        (nn.Module)             Recurrent layer class
        Raise
        (ValueError)            In case hidden type is not valid
        """
        # Case hidden type is LSTM
        if hidden_type == 'LSTM':
            return nn.LSTM

        # Case hidden type is GRU
        if hidden_type == 'GRU':
            return nn.GRU

        # Otherwise: raise error
        raise ValueError(' '.join([
            'Error: given recurrent neural network type can be',
            'either LSTM or GRU: {} given instead'.format(hidden_type)
        ]))

    def forward(self, x, state=None):
        """ Make forward step

        Args
        x (torch.Tensor)        Tensor containing input vectors with size
                                (batch size, window size, number of features)

        state (?)               TODO

        Return
        (torch.Tensor)          Predicted new values (shifted input vector)
        (?)                     TODO
        """
        # Go through embedding layer first
        x = self.embed(x)
        # Recurrent (hidden) layer output
        x, state = self.rnn(x, state)
        # Decision (linear) layer output
        x = self.out(x)
        # Return both new x and state
        return x, state

    def train_batch(self, batch, loss_fn, optimizer):
        """ Train batch of input data

        Args
        batch (torch.Tensor)        Float tensor representing input data
        loss_fn (nn.Module)         Loss function, used to compute train loss
        optimizer (nn.optimizer)    Optimizer used to find out best weights

        Return
        (float)                     Mean loss
        """
        # Take out target variable (last word of the sentence)
        target = batch[:, -1]
        # Remove the target variable from the input tensor
        input = batch[:, :-1]

        # Eventually clear previous recorded gradients
        optimizer.zero_grad()
        # Make forward pass
        output, _ = self(input)

        # Evaluate loss only for last output
        loss = loss_fn(output[:, -1, :], self.embed(target))
        # Backward pass
        loss.backward()
        # Update
        optimizer.step()
        # Return average batch loss
        return float(loss.data)

    def test_batch(self, batch, loss_fn):
        """ Test batch of input data

        Args
        batch (torch.Tensor)        Float tensor representing input data
        loss_fn (nn.Module)         Loss function, used to compute train loss

        Return
        (float)                     Mean loss
        """
        # Take out target variable (last character) from characters window
        target = batch[:, -1]
        # Remove the target variable from the input tensor
        input = batch[:, :-1]
        # Make forward pass
        output, _ = self(input)
        # Evaluate loss only for last output
        loss = loss_fn(output[:, -1, :], self.embed(target))
        # Return average batch loss
        return float(loss.data)
