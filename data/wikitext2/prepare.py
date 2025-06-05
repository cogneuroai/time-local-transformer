import requests
# Choose your tokenizer:
# Option 1: tiktoken
#import tiktoken
# Option 2: HuggingFace transformers
from transformers import GPT2Tokenizer

import numpy as np
import zipfile
import io
import pickle
import math

# specify the url and filename
url = 'https://wikitext.smerity.com/wikitext-2-v1.zip'
filename = 'wikitext-2-v1.zip'
#
## download the wikitext-2 dataset
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(path='data/wikitext2/')

# specify the data file paths (train, test, valid)
train_data_path = 'data/wikitext2/wikitext-2/wiki.train.tokens'
test_data_path = 'data/wikitext2/wikitext-2/wiki.test.tokens'
valid_data_path = 'data/wikitext2/wikitext-2/wiki.valid.tokens'

# Choose your tokenizer initialization:
# Option 1: tiktoken
#enc = tiktoken.get_encoding("gpt2")
#vocab_size = enc.n_vocab
# Option 2: HuggingFace
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
vocab_size = len(tokenizer)

# Function to round up to nearest multiple (e.g., 64 or 128 for GPU efficiency)
def round_up_vocab_size(vocab_size, multiple=64):
    return math.ceil(vocab_size / multiple) * multiple

# Calculate the rounded-up vocabulary size
rounded_vocab_size = round_up_vocab_size(vocab_size)
print(f"Rounded vocabulary size for efficiency: {rounded_vocab_size}")

# define a function to load, tokenize and save data
vocab_set = set()
def process_data(file_path, save_file):
    global vocab_set
    with open(file_path, 'r') as f:
        data = f.read()

    # Choose your tokenization method:
    # Option 1: tiktoken
    #ids = enc.encode_ordinary(data)
    # Option 2: HuggingFace
    ids = tokenizer.encode(data, add_special_tokens=False)
    
    vocab_set.update(ids)
    print(f"{file_path} has {len(ids):,} tokens")

    # Print first 3000 tokens to a debug file
    #debug_file = save_file + '.debug.txt'
    #with open(debug_file, 'w') as f:
    #    f.write(','.join(map(str, ids[:3000])))
        
    # export to bin files
    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(save_file)

with open('data/wikitext2/vocab_size.pkl', 'wb') as f:
    pickle.dump(rounded_vocab_size, f)

# process and save the tokenized train, test, valid data
process_data(train_data_path, 'data/wikitext2/train.bin')
process_data(test_data_path, 'data/wikitext2/test.bin')
process_data(valid_data_path, 'data/wikitext2/val.bin')

print(f"Original vocabulary size: {vocab_size}") 
print(f"Rounded vocabulary size for efficiency: {rounded_vocab_size}") 

