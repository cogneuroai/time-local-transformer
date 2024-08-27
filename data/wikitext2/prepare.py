import os
import numpy as np
import torch
import zipfile
import requests
import io
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm
from collections import Counter
import pickle

# specify the url and filename
url = 'https://wikitext.smerity.com/wikitext-2-v1.zip'
filename = 'wikitext-2-v1.zip'

# download the wikitext-2 dataset
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(path='data/wikitext2/')

# Specify the data file paths (train, test, valid)
train_data_path = 'data/wikitext2/wikitext-2/wiki.train.tokens'
test_data_path = 'data/wikitext2/wikitext-2/wiki.test.tokens'
valid_data_path = 'data/wikitext2/wikitext-2/wiki.valid.tokens'

# Function to yield tokens from the dataset
def yield_tokens(data_iter):
    for text in data_iter:
        yield text.split()  # Split on whitespace

# Build vocabulary with manual handling of special tokens
def build_vocab(data_path):
    print(f"Building vocabulary from {data_path}...")

    # Initialize a counter to collect all tokens
    counter = Counter()
    with open(data_path, 'r') as f:
        for line in f:
            counter.update(line.split())  # Split on whitespace

    # Create a Vocab object from the counter, adding special tokens
    vocab = Vocab(counter)

    # Create a dictionary for token to index mapping
    token_to_index = {token: idx for idx, token in enumerate(vocab.itos)}

    return vocab, token_to_index

def process_data(file_path, save_file, vocab, token_to_index):
    print(f"Processing {file_path}...")

    # Open the file and read the content
    with open(file_path, 'r') as f:
        data = f.read()

    # Tokenize and numericalize the data with progress
    tokens = data.split()  # Split on whitespace
    ids = []
    unk_index = vocab['<unk>']  # Get the index for '<unk>' token once

    # Debug: Print the first few tokens and their indices
    #print(f"First few tokens before numericalization: {tokens[:10]}")

    for token in tqdm(tokens, desc="Tokenizing and numericalizing"):
        token_idx = token_to_index.get(token, unk_index)
        ids.append(token_idx)
        # Debug: Print the token and its assigned index
        #if len(ids) < 100:  # Limit to the first 10 tokens for clarity
        #print(f"Token: {token}, Index: {token_idx}")

    print(f"{file_path} has {len(ids):,} tokens")

    # Save the tokenized data
    ids_array = np.array(ids, dtype=np.uint16)
    ids_array.tofile(save_file)
    print(f"Saved {save_file}")

def decode_train_bin(save_file, token_to_index):
    # Reverse the token_to_index dictionary to create an index_to_token dictionary
    index_to_token = {idx: token for token, idx in token_to_index.items()}

    # Load the binary file
    data = np.fromfile(save_file, dtype=np.uint16)

    # Decode the binary file back to tokens
    decoded_tokens = [index_to_token.get(idx, '<unk>') for idx in data]

    # Print out a sample of the decoded tokens
    print(f"\nDecoded tokens from {save_file}:")
    print(" ".join(decoded_tokens[:100]))  # Print the first 100 tokens

# Build vocabulary using the training data
vocab, token_to_index = build_vocab(train_data_path)
vocab_size = len(vocab)
with open('data/wikitext2/vocab_size.pkl', 'wb') as f:
    pickle.dump(vocab_size, f)

# Save the entire vocabulary as a dictionary
with open('data/wikitext2/vocab.pkl', 'wb') as f:
    pickle.dump(token_to_index, f)

# Process and save the tokenized train, test, and valid data
process_data(train_data_path, 'data/wikitext2/train.bin', vocab, token_to_index)
process_data(test_data_path, 'data/wikitext2/test.bin', vocab, token_to_index)
process_data(valid_data_path, 'data/wikitext2/val.bin', vocab, token_to_index)

decode_train_bin('data/wikitext2/train.bin', token_to_index)

print(f"Total vocabulary size: {len(vocab):,}")