import os
import re
import torch
import random
import unidecode as ud
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Mobydick(Dataset):

    # Constructor
    def __init__(self, file_path, min_len=4, transform=None):

        # Load data
        with open(file_path, 'r') as file:
            text = file.read()

        ## Text preprocessing
        # Remove non-unicode characters
        text = ud.unidecode(text)
        # Lowarcase
        text = text.lower()
        # Remove single newlines
        text = re.sub(r'(?<!\n)\n', ' ', text)
        # Remove punctuation and numbers
        text = re.sub(r'[\*\=\/\d]', '', text)
        # Remove underscores
        text = re.sub('_', '', text)
        # Remove symbols between words
        text = re.sub(r'(?<=\D)[-]+(?=(\D))', ' ', text)
        # Remove double spaces
        text = re.sub(r'[\t ]+', ' ', text)

        # Split text into sentences
        sentences = list(re.findall(r'([^\.\!\?\n]+[\.\!\?]+["]{,1}[ ]{,1})', text))

        # Split sentences according to words
        words = []
        sentences_clean = []
        for sentence in sentences:
            # Split sentence into words according to punctuation
            tokens = list(re.split(r'([ \n\:\;\"\,\(\)\.\!\?])', sentence))
            # Remove useless characters
            tokens = [w for w in tokens if re.search('[^| ]', w)]
            # filter short sentences
            if len(tokens) >= min_len:
                # Substitute entire sentence with splited one
                sentences_clean.append(tokens)
                # Save words
                words.extend(tokens)

        # Store words
        self.words = set(words)
        # Store sentences
        self.sentences = sentences_clean
        # Store sentences transformation pipeline
        self.transform = transform

    def __len__(self):

        return len(self.sentences)

    def __getitem__(self, i):

        # Case index i is greater or equal than number of sequences
        if i >= len(self.sentences):
            # Raise key error
            raise IndexError('Chosen index exceeds sentences indices')

        # Case transform is not set
        if self.transform is None:
            # Just return i-th sentence
            return self.sentences[i]

        # Otherwise, transform it and return transformation result
        return self.transform(self.sentences[i])


class Bible(Dataset):

    # Constructor
    def __init__(self, file_path, min_len=4, transform=None):

        # Load data
        with open(file_path, 'r') as file:
            text = file.read()

        ## Text preprocessing
        # Remove non-unicode characters
        text = ud.unidecode(text)
        # Lowercase
        text = text.lower()
        # Remove single newlines
        text = re.sub(r'(?<!\n)\n', ' ', text)
        # Remove undesired punctuation and numbers
        text = re.sub(r'[\*\=\/\d]', '', text)
        # Remove undesired symbols between words
        text = re.sub(r'(?<=\D)[-]+(?=(\D))', ' ', text)
        # Remove double spaces
        text = re.sub(r'[\t ]+', ' ', text)

        # Split text into sentences
        sentences = list(re.findall(r'([^\.\!\?\n]+[\.\!\?]+["]{,1}[ ]{,1})', text))

        # Split sentences according to words
        words = []
        sentences_clean = []
        for sentence in sentences:
            # Split sentence into words according to punctuation
            tokens = list(re.split(r'([ \n\:\;\"\,\(\)\.\!\?])', sentence))
            # Remove useless characters
            tokens = [w for w in tokens if re.search('[^| ]', w)]
            # Remove punctuation at the beginning of sentences
            if tokens[0] == ':' and  tokens[1] == ':':
                tokens = tokens[2:]
            elif tokens[0] == ':' and  tokens[1]!= ':':
                tokens = tokens[1:]
            # Filter short sentences
            if len(tokens) >= min_len:
                # Substitute entire sentence with splited one
                sentences_clean.append(tokens)
                # Save words
                words.extend(tokens)

        # Store words
        self.words = set(words)
        # Store sentences
        self.sentences = sentences_clean
        # Store sentences transformation pipeline
        self.transform = transform

    def __len__(self):

        return len(self.sentences)

    def __getitem__(self, i):

        # Case index i is greater or equal than number of sequences
        if i >= len(self.sentences):
            # Raise key error
            raise IndexError('Chosen index exceeds sentences indices')

        # Case transform is not set
        if self.transform is None:
            # Just return i-th sentence
            return self.sentences[i]

        # Otherwise, transform it and return transformation result
        return self.transform(self.sentences[i])


def split_train_test(dataset, train_prc=0.8):

    # Define dataset length
    n = len(dataset)
    # Define number of training dataset indices
    m = round(train_prc * n)
    # Split datasets in two
    return torch.utils.data.random_split(dataset, [m, n - m])


class WordToIndex(object):

    # Constructor
    def __init__(self, words):

        # Store list of words
        self.words = set(words)
        # Define mapping from words to integers
        self.encoder = {e: i for i, e in enumerate(words)}
        # Define mapping from integers to words
        self.decoder = {i: e for i, e in enumerate(words)}

    # Return word as its index
    def __call__(self, sentence):
        # Make list of labels from words
        labels = [self.encoder[w] for w in sentence if w in self.words]
        # Return list of labels
        return labels

    # Return vector of indices as its corresponding words
    def reverse(self, sentence):
        # Make list of words from labels
        words = [self.decoder[i] for i in sentence if i in self.decoder.keys()]
        # retrun list of words
        return words


class RandomCrop(object):

    # Constructor
    def __init__(self, crop_len):
        # Store crop length
        self.crop_len = crop_len

    def __call__(self, sentence):

        # Check for compatibility of crop length with sentence length
        if len(sentence) < self.crop_len:
            # Raise new index error
            raise IndexError(' '.join([
                'Error: given crop length is {:d}'.format(self.crop_len),
                'while current sentence length is {:d}:'.format(len(sentence)),
                'crop length must be smaller or equal than sentence length'
            ]))

        # Define start annd end index of crop window, at random
        i = random.randint(0, len(sentence) - self.crop_len)
        j = i + self.crop_len
        # Take a subset of coriginal sentence
        sentence = sentence[i:j]
        # Return subset
        return sentence


class ToTensor(object):

    def __call__(self, sentence):

        return torch.tensor(sentence).float()
