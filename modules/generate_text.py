import numpy as np
from torch import  nn
from modules.dataset import *


def generate_word(net_out, words, w2i, X, classification):
    # Initialize softmax
    softmax = nn.Softmax(dim=1)
    if classification:
        # Compute probabilities
        prob = softmax(net_out[:, -1, :])
        # Pick most probable index
        next_index = torch.argmax(prob).item()
        # Retrieve most probable word
        next_word = w2i.decode([next_index])[0]
    else:
        # Retrieve the index of the embedding closest to the net output
        distances = np.linalg.norm((X - net_out[:, -1, :].to('cpu').numpy()[0]),
                    axis = 1)
        #print(distances)
        closest_index = np.argmin(distances)
        # Retrieve the chosen word text
        next_word = words[closest_index]
        # Add to the seed sentence
    return next_word

def generate_text(net, seed, n, device, words, w2i, X = None,
                classification = True):

    # Evaluation mode
    net.eval()
    # Print initial seed
    print(seed, end=' ', flush=True)

    with torch.no_grad():
        ## Find initial state of the RNN
        # Transform words in the corresponding indices
        seed_encoded = torch.tensor(w2i(seed.lower()))
        # Reshape: batch-like shape
        seed_encoded = torch.reshape(seed_encoded, (1, -1))
        # Move to the selected device
        seed_encoded = seed_encoded.to(device)
        # Forward step
        net_out, net_state = net(seed_encoded)
        # Generate next word
        next_word = generate_word(net_out, words, w2i, X, classification)
        # Add to seed
        seed += ' ' + next_word
        ## Generate n words
        for i in range(n):
            # Transform words in the corresponding indices
            seed_encoded = torch.tensor(w2i(seed.lower()))
            # Reshape: batch-like shape
            seed_encoded = torch.reshape(seed_encoded, (1, -1))
            # Move to the selected device
            seed_encoded = seed_encoded.to(device)
            # Forward step
            net_out, net_state = net(seed_encoded, net_state)
            # Generate next word
            next_word = generate_word(net_out, words, w2i, X, classification)
            # Add to seed
            seed += ' ' + next_word
            # Print the current result
            print(next_word, end=' ', flush=True)

    return seed
