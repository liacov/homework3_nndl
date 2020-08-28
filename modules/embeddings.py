import re
import numpy as np


def from_file(path, words=None):

    # Intitialize words embeddings (word: vector)
    embeddings = dict()
    # Open file
    with open(path, 'r') as file:
        # Loop through each line
        for line in file:
            # Clean line from newline characters
            line = re.sub(r'[\n\r]+', '', line)
            # Split line according to spaces
            line = re.split(r'\s+', line)
            # Get word (first item in line) and vetcor (other items)
            word, vector = line[0], [float(v) for v in line[1:]]
            # If words is not empty and current word is not required
            if words and (word not in words):
                # Skip iteration
                continue
            # Otherwise, save embedding
            embeddings.setdefault(word, vector)
    # Return either list of words and vectors
    return embeddings

def gaussian_sampling(mean, std, dim, words):

    # Initialize embeddings
    embeddings = dict()
    # Loop through each word in text
    for word in words:
        # Initialize random vector
        vector = np.random.normal(loc=mean, scale=std, size=(dim, )).tolist()
        # Store vector word vector
        embeddings.setdefault(word, vector)
    return embeddings
