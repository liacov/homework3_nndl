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
        # Take out target variable (last character) from characters window
        target = batch[:, -1]
        # Remove the target variable from the input tensor
        input = batch[:, :-1]

        # Eventually clear previous recorded gradients
        optimizer.zero_grad()
        # Make forward pass
        output, _ = self(input)

        # Evaluate loss only for last output
        loss = loss_fn(output, target)
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
        loss = loss_fn(output, target)
        # Return average batch loss
        return float(loss.data)


def save_epochs(path, params=list(), train_losses=list(), train_times=list(), test_losses=list(), test_times=list()):
    """ Save epochs as .json

    Args
    params (list)           List of parameters combinations
    train_losses (list)     List of mean loss per train epoch
    train_times (list)      List of time taken per train epoch
    test_losses (list)      List of mean loss per test epoch
    test_times (list)       List of time taken per test epoch
    params_path (list)      Path to output file

    Raise
    (OSError)               In case saving file was not possible
    """
    # Open output file
    with open(path, 'w') as file:
        # Write json file
        json.dump({
            # Save parameters name and value as strings
            'params': [
                {str(kw): str(arg) for kw, arg in params[i]}
                for i in range(len(params))
            ],
            # Save lists
            'train_losses': train_losses,
            'train_times': train_times,
            'test_losses': test_losses,
            'test_times': test_times
        }, file)


# Utility: load epochs from disk
def load_params(path):
    """ Load epochs from .json file

    Args
    path (str)              Path to file holding epochs values

    Return
    (list)                  List of parameters combinations
    (list)                  List of mean loss per train epoch
    (list)                  List of time taken per train epoch
    (list)                  List of mean loss per test epoch
    (list)                  List of time taken per test epoch

    Raise
    (FileNotFoundError)         In case file does not exists
    (OSError)                   In case it was not possible opening file
    """
    # Initialize parameters list
    params = list()
    # Initialize lists of training losses and times
    train_losses, train_times = list(), list()
    # Initialize lists of test losses and times
    test_losses, test_times = list(), list()
    # Open input file
    with open(path, 'r') as file:
        # Load json file
        epochs_dict = json.load(file)
        # Retrieve lists of values
        params = epochs_dict.get('params', list())
        train_losses = epochs_dict.get('train_losses', list())
        train_times = epochs_dict.get('train_times', list())
        test_losses = epochs_dict.get('test_losses', list())
        test_times = epochs_dict.get('test_times', list())
    # Return retrieved values
    return params, train_losses, train_times, test_losses, test_times


# Utility: make training and testing
def train_test_epochs(
    net, train_dl, test_dl, loss_fn, optimizer, num_epochs, save_after,
    net_path='', epochs_path='', verbose=True, device=torch.device('cpu')
):
    """ Load, train and test a network

    Args
    net (nn.Module)             Network to be trained and tested
    train_dl (DataLoader)       Train examples iterator
    test_dl (DataLoader)        Test exapmples iterator
    loss_fn (?)                 Function used to compute loss
    optimizer (toch.optim)      Optimizer updating network weight
    num_epochs (int)            Number of epochs to loop through
    save_after (int)            Number of epochs after which a checkupoint
                                must be saved
    net_path (str)              Path to network weights file (.pth)
    epochs_path (str)           Path to epochs values file (.tsv)
    verbose (bool)              Wether to show verbose output or not
    device (torch.device)       Device holding network weights

    Return
    (list)      Train losses (mean per epoch)
    (list)      Test losses (mean per epoch)
    (list)      Train time (per epoch)
    (list)      Test time (per epoch)
    """
    # If file path is set, load network from file
    if os.path.isfile(net_path):
        # Load pretrained weigths and optimizer state
        net = net.from_file(path=net_path, optimizer=optimizer)
    # Move network to device
    net.to(device)

    # Initialize train and test loss (per epoch)
    train_losses, test_losses = list(), list()
    # Initialize train and test times (per epoch)
    train_times, test_times = list(), list()

    # Eventually load epochs from file
    if os.path.isfile(epochs_path):
        # Load epochs values
        _, train_losses, train_times, test_losses, test_times = load_params(
            epochs_path=epochs_path
        )

    # Loop through each epoch
    for e in num_epochs:
        # Set network in training mode
        net.train()
        # Train the network, retrieve mean loss and total time for current epoch
        train_loss, train_time = net.train_epoch(
            train_dl=train_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        # Store train loss
        train_losses.append(train_loss)
        # Store train time
        train_times.append(train_time)

        # Set network in evaluation mode
        net.eval()
        # Disable gradient computation
        with torch.no_grad():
            # Test the network, retrieve mean loss and total time for current epoch
            test_loss, test_time = net.test_epoch(
                test_dl=test_dl,
                loss_fn=loss_fn,
                device=device
            )
            # Store test loss
            test_losses.append(test_loss)
            # Store test time
            test_times.append(test_time)

        # Check if current epoch is a checkpoint
        if (e + 1) % save_after == 0:
            # Verbose output
            if verbose:
                print('Epoch nr. {:d}'.format(e + 1))
                print('Train loss (mean) {:.3f}'.format(train_loss), end='')
                print('in {:.0f} seconds'.format(train_time))
                print('Test loss (mean) {:.3f}'.format(test_loss), end='')
                print('in {:.0f} seconds'.format(test_time))

            # Save model weights
            net.to_file(path=net_path, optimizer=optimizer)
            # Save epochs train and test values
            save_params(
                params=list(),
                train_losses=train_losses,
                train_times=train_times,
                test_losses=test_losses,
                test_times=test_times,
                epochs_path=epochs_path
            )
