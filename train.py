import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
import pickle
from tqdm import tqdm
import wandb

from gaver import Gaver
from post_F import Post as Post_F

torch.manual_seed(2077)
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Time-Local Transformer model.")

    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of token embeddings')
    parser.add_argument('--ntau', type=int, default=32, help='Number of filters for sith model')
    parser.add_argument('--tau_max', type=int, default=32, help='Center of the temporal receptive field for the last taustar produced')
    parser.add_argument('--tau_min', type=int, default=1, help='Center of the temporal receptive field for the first taustar produced')
    parser.add_argument('--k', type=int, default=8, help='Accuracy of the inverse reconstruction')
    parser.add_argument('--g', type=float, default=0.0, help='Scaling factor of the output of the module')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size of RNNs for each embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=6e-5, help='Learning rate for optimizer')
    parser.add_argument('--data_dir', type=str, default='data/wikitext2', help='Directory containing data files')
    parser.add_argument('--vocab_size_path', type=str, default='vocab_size.pkl', help='Path to the vocabulary size pickle file')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='Path to the vocabulary dictionary for decoding')
    parser.add_argument('--n_tokens_log', type=int, default=None, help='Number of tokens to process before logging loss')
    parser.add_argument('--n_tokens_eval', type=int, default=None, help='Number of tokens to go through when evaluating.')
    parser.add_argument('--wandb_log', action='store_true', help='Enable logging to W&B')
    parser.add_argument('--wandb_group_name', type=str, default='testing', help='name for wandb group')
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--no_test", action="store_true", help="no test set for this dataset")
    parser.add_argument('--model_type', type=str, default='rnn', choices=['rnn', 'lstm', 'gru', 'sith'], help='Type of model to use (rnn, lstm, gru, or sith)')
    parser.add_argument('--eval_on_log', action='store_true', help='Evaluate immediatley after n_tokens_log'),
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], default='cuda', help='Device to run the model on'),
    parser.add_argument('--save_every_epoch', action='store_true', help='Save a checkpoint after every epoch'),
    parser.add_argument('--n_accum_steps', type=int, default=1, help='Number of accumulation steps for gradient updates')
    parser.add_argument('--compile', action='store_true', help='Compile the model using torch.compile()')
    parser.add_argument('--detach_on_accum', action='store_true', help='Detach hidden states only after n_accum_steps')
    parser.add_argument('--inverse_method', type=str, default='post', choices=['euler', 'gaver', 'cme', 'post', 'post_F'], help='Method to use for inverse reconstruction')
    parser.add_argument('--decode', action='store_true', help='Decode tokens during training')
    parser.add_argument('--sample_during_training', action='store_true', help='Sample during training')
    parser.add_argument('--wandb_entity_name', type=str, required=True, help='Specify the entity name for wandb logging')

    return parser.parse_args()

def get_batch(data_path, batch_size, mode, single_pass=True, n_tokens_eval=None, disable_tqdm=False):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    total_elements = len(data)
    if n_tokens_eval:
        total_elements = n_tokens_eval  # Adjust total elements if evaluation limit is set

    # Calculate the size of each segment based on the batch size
    segment_length = total_elements // batch_size
    offsets = np.arange(batch_size) * segment_length

    # Initialize pointers for each segment
    pointers = np.zeros(batch_size, dtype=int)

    # Initialize tqdm for visual progress tracking
    progress_bar = tqdm(total=segment_length - 1, desc=f'{mode} Processing', disable=disable_tqdm)

    while True:
        x_batch = []
        y_batch = []

        # Collect one token from each segment
        for j in range(batch_size):
            if pointers[j] < segment_length - 1:  # Ensure there's a next token to fetch
                start_idx = offsets[j] + pointers[j]
                next_idx = start_idx + 1
                x_batch.append(data[start_idx])
                y_batch.append(data[next_idx])
                pointers[j] += 1

        if not x_batch or not y_batch:  # If any batch is empty, break the loop (end of segment)
            break

        # Convert lists to tensors and add batch dimension
        x_batch = torch.tensor(x_batch, dtype=torch.int64).unsqueeze(1)
        y_batch = torch.tensor(y_batch, dtype=torch.int64).unsqueeze(1)

        yield x_batch, y_batch, np.sum(pointers), total_elements

        # Update tqdm progress bar
        progress_bar.update(1)

        if single_pass and np.all(pointers >= segment_length - 1):
            break

    progress_bar.close()

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.c_fc    = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class BlockRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_heads, model_type):
        super(BlockRNN, self).__init__()
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if model_type == 'rnn':
            self.rnns = nn.ModuleList([nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True) for _ in range(embedding_dim)])
        elif model_type == 'lstm':
            self.rnns = nn.ModuleList([nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True) for _ in range(embedding_dim)])
        elif model_type == 'gru':
            self.rnns = nn.ModuleList([nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True) for _ in range(embedding_dim)])
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)
        self.mlp = MLP(embedding_dim)
        self.layer_norm_1 = LayerNorm(embedding_dim, bias=False)
        self.layer_norm_2 = LayerNorm(embedding_dim, bias=False)
        self.layer_norm_3 = LayerNorm(embedding_dim, bias=False)

    def forward(self, current_input, h0s):
        rnn_outputs = []

        hxs = []
        for i, rnn in enumerate(self.rnns):
            rnn_input = current_input[:, i].unsqueeze(-1)
            output, hx = rnn(rnn_input, h0s[i])
            hxs.append(hx)
            rnn_outputs.append(output.squeeze(1))  # Take the last output from each RNN

        rnn_outputs = torch.stack(rnn_outputs, dim=2)  # Shape: [batch_size, hidden_size, embedding_dim]

        ## Attention mechanism

        query = self.layer_norm_1(current_input).unsqueeze(1) # Shape: [batch_size, 1, embedding_dim] for compatibility with multihead_attention

        skip_connection = current_input.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]

        kv = self.layer_norm_2(rnn_outputs)
        kv = torch.cat((query, kv), dim=1) # self attention

        attn_output, _ = self.multihead_attention(query, kv, kv)

        # skip connection 1
        attn_output = attn_output + skip_connection
        # skip connection 2
        attn_output = attn_output + self.mlp(self.layer_norm_3(attn_output))  # Skip connection 2

        attn_output = attn_output.squeeze(1)  # Shape: [batch_size, embedding_dim]
        return attn_output, hxs

class BlockSITH(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ntau, n_heads, tau_max, g, k, inverse_method):
        super(BlockSITH, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.inverse_method = inverse_method
        #if inverse_method == 'euler':
        #    self.sith = Euler(tau_min=1, tau_max=tau_max, n_taus=ntau, max_fn_evals=k, g=g, batch_first=True)
        if inverse_method == 'gaver':
            self.sith = Gaver(tau_min=1, tau_max=tau_max, n_taus=ntau, max_fn_evals=k, g=g, batch_first=True)
        elif inverse_method == 'post_F':
            self.sith = Post_F(tau_min=1, tau_max=tau_max, n_taus=ntau, k=k, g=g, batch_first=True)
        #elif inverse_method == 'post':
        #    self.sith = Post(tau_min=1, tau_max=tau_max, n_taus=ntau, k=k, g=g, batch_first=True)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)
        self.embedding_dim = embedding_dim
        self.mlp = MLP(embedding_dim)
        self.layer_norm_1 = LayerNorm(embedding_dim, bias=False)
        self.layer_norm_2 = LayerNorm(embedding_dim, bias=False)
        self.layer_norm_3 = LayerNorm(embedding_dim, bias=False)

    def forward(self, current_input, h0):

        if self.inverse_method == 'post_F':
            current_input = current_input.unsqueeze(0)
            output, hs = self.sith(fs=current_input, F=h0)
            output = output.squeeze(0)
            current_input = current_input.squeeze(0)
        else:
            current_input = current_input - current_input.min(dim=-1, keepdim=True)[0]
            output, hs = self.sith(f=current_input, hx=h0)

        
        output = output.float().transpose(1,2) # Shape: [batch_size, ntau, embedding_dim]

        ## Attention mechanism

        query = self.layer_norm_1(current_input).unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]

        skip_connection = current_input.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]

        kv = self.layer_norm_2(output)
        kv = torch.cat((query, kv), dim=1) # self attention # Shape: [batch_size, ntau+1, embedding_dim]

        attn_output, _ = self.multihead_attention(query, kv, kv) # Shape: [batch_size, 1, embedding_dim]

        # skip connection 1
        attn_output = attn_output + skip_connection
        # skip connection 2
        attn_output = attn_output + self.mlp(self.layer_norm_3(attn_output))

        attn_output = attn_output.squeeze(1)  # Shape: [batch_size, embedding_dim]
        return attn_output, hs
    
class TimeLocalTransformer(nn.Module):
    def __init__(self, vocab_size, args):
        super(TimeLocalTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, args.embedding_dim)
        self.model_type = args.model_type
        self.embedding_dim = args.embedding_dim
        self.detach_on_accum = args.detach_on_accum
        self.n_accum_steps = args.n_accum_steps
        if args.model_type == 'sith':
            self.blocks = nn.ModuleList([BlockSITH(vocab_size, args.embedding_dim, args.ntau, args.n_heads, args.tau_max, args.g, args.k, args.inverse_method) for _ in range(args.n_layers)])
        else:
            self.blocks = nn.ModuleList([BlockRNN(vocab_size, args.embedding_dim, args.hidden_size, args.n_heads, args.model_type) for _ in range(args.n_layers)])

        self.llm_head = nn.Linear(args.embedding_dim, vocab_size)

    def forward(self, current_input, h0s):
        x = current_input
        hxs = []
        for block, h0 in zip(self.blocks, h0s):
            x, hx = block(x, h0)
            hxs.append(hx)
        logits = self.llm_head(x)
        return logits, hxs

def decode_token(token_idx, vocab_dict):
    if token_idx.dim() > 0:
        token_idx = token_idx[128]
    return vocab_dict.get(token_idx.item())

def sample_model(model, initial_sentence, vocab_dict, args, max_length=50):
    model.eval()  # Set model to evaluation mode

    # Tokenize the initial sentence using the provided vocab_dict
    #vocab_dict = {idx: token for token, idx in vocab_dict.items()}
    vocab_dict = {token: idx for idx, token in vocab_dict.items()}
    tokens = [vocab_dict.get(word, vocab_dict['<unk>']) for word in initial_sentence.split()]
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(args.device)  # Shape: (1, seq_len)
    
    generated_tokens = tokens.tolist()[0]  # Initialize generated tokens list

    hx = [None for _ in range(args.n_layers)]  # Initialize hidden states for each layer
    for _ in range(max_length - len(generated_tokens)):
        with torch.no_grad():
            # Get the model output for the current sequence of tokens
            output, hx = model(tokens, hx)
            
            # Get the index of the last token (from the output)
            last_token_logits = output[:, -1, :]
            last_token_idx = last_token_logits.argmax(dim=-1).item()
            
            # Add the last token index to the generated tokens
            generated_tokens.append(last_token_idx)
            
            # Prepare the new input sequence (including the newly generated token)
            tokens = torch.LongTensor(generated_tokens).unsqueeze(0).to(args.device)
    
    # Decode the generated tokens back to words using the provided vocab_dict
    decoded_sentence = ' '.join(decode_token(torch.tensor([token_idx]), vocab_dict) for token_idx in generated_tokens)
    
    return decoded_sentence


def train_model(model, train_loader, criterion, optimizer, device, n_tokens_log, wandb_log, eval_on_log, n_accum_steps, n_layers, vocab_dict=None):
    model.train()
    total_loss = 0.0
    token_count = 0
    running_tokens = 0
    accum_loss = 0.0
    step = False

    if model.model_type == 'sith':
        hxs = [None for _ in range(n_layers)]
    else:
        hxs = [[None] * model.embedding_dim for _ in range(n_layers)]

    optimizer.zero_grad()
    for step, (x_batch, y_batch, _, total_elements) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        total_elements = total_elements

        # Get the current input and target token
        current_input = model.embedding(x_batch).squeeze(1)  # Shape: [batch_size, embedding_dim]
        current_target = y_batch.squeeze(1)

        # Forward pass
        logits, hxs = model(current_input, hxs)

        if not model.detach_on_accum:
            hxs = tree_map(lambda x: x.detach(), hxs)

        # Compute loss
        loss = criterion(logits.view(-1, logits.size(-1)), current_target)
        accum_loss += loss
        total_loss += loss.item() * x_batch.size(0)  # Multiply loss by batch size for proper averaging later
        token_count += x_batch.size(0)
        running_tokens += x_batch.size(0)

        ## for super debugging by decoding the tokens
        ###############################################################
        if vocab_dict:
            #with open('./decoded_tokens', 'w') as f:
            #    f.write("")
            current_target = decode_token(current_target, vocab_dict)
            predicted_token_idx = torch.argmax(logits, dim=-1)
            predicted_token = decode_token(predicted_token_idx, vocab_dict)
            ###############################################################
            #print(f"Target: {current_target}, Predicted: {predicted_token}, Loss: {loss.item():.4f}")
            with open('./decoded_tokens.txt', 'a') as f:
                f.write(f"Target: {current_target}, Predicted: {predicted_token}, Loss: {loss.item():.4f}\n")
            ###############################################################


        # Backpropagation and update
        if not model.detach_on_accum:
            loss.backward()

        if (step+1) % n_accum_steps == 0:
            #print('stepping optimizer', step+1)
            if model.detach_on_accum:
                hxs = tree_map(lambda x: x.detach(), hxs)
                accum_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            accum_loss = 0.0

        average_loss = total_loss / token_count
        # Logging and printing
        stop = False
        if n_tokens_log and token_count >= n_tokens_log:
            if wandb_log:
                wandb.log({"train_loss": average_loss, "train_ppl": np.exp(average_loss), "running_train_tokens": running_tokens})
            print(f"\ntrain_loss: {average_loss:.4f}, train_ppl: {np.exp(average_loss):.4f}, running_tokens: {running_tokens}")
            total_loss = 0  # Reset total loss after logging
            token_count = 0  # Reset token count after logging

            if eval_on_log:
                return running_tokens, total_elements, average_loss, np.exp(average_loss), stop

    stop = True
    if wandb_log:
        wandb.log({"train_loss": average_loss, "train_ppl": np.exp(average_loss), "running_train_tokens": running_tokens})
    print(f"\ntrain_loss: {average_loss:.4f}, train_ppl: {np.exp(average_loss):.4f}, running_tokens: {running_tokens}")
    return running_tokens, total_elements, average_loss, np.exp(average_loss), stop
    

@torch.no_grad()
def evaluate_model(model, val_loader, criterion, device, wandb_log, mode, n_layers):
    model.eval()
    total_loss = 0
    token_count = 0

    if model.model_type == 'sith':
        hxs = [None for _ in range(n_layers)]
    else:
        hxs = [[None] * model.embedding_dim for _ in range(n_layers)]
    
    for step, (x_batch, y_batch, _, _) in enumerate(val_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Get the current input and target token
        current_input = model.embedding(x_batch).squeeze(1)  # Shape: [batch_size, embedding_dim]
        current_target = y_batch.squeeze(1)

        # Forward pass for a single token
        logits, hxs = model(current_input, hxs)

        if not model.detach_on_accum:
            hxs = tree_map(lambda x: x.detach(), hxs)
        
        # Compute loss for current token
        loss = criterion(logits.view(-1, logits.size(-1)), current_target)
        total_loss += loss.item() * x_batch.size(0)
        token_count += x_batch.size(0)

        if (step+1) % model.n_accum_steps == 0:
            #print('stepping optimizer', step+1)
            if model.detach_on_accum:
                hxs = tree_map(lambda x: x.detach(), hxs)

    # Calculate average loss after all batches
    average_loss = total_loss / token_count

    if wandb_log:
        wandb.log({f"{mode}_loss": average_loss, f"{mode}_ppl": np.exp(average_loss)})

    print(f"\n{mode}_loss: {average_loss:.4f}, {mode}_ppl: {np.exp(average_loss):.4f}")
    print(f"{mode} token count: {token_count}")
    
    # Reset total loss and token count
    total_loss = 0
    token_count = 0

    return average_loss, np.exp(average_loss)

def save_checkpoint(epoch, model, optimizer, train_loss, train_ppl, val_loss, val_ppl, test_loss, test_ppl, running_tokens, min_train_loss, min_train_ppl, min_val_loss, min_val_ppl, min_test_loss, min_test_ppl, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_ppl': train_ppl,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'test_loss': test_loss,
        'test_ppl': test_ppl,
        'running_train_tokens': running_tokens,
        'min_train_loss': min_train_loss,
        'min_train_ppl': min_train_ppl,
        'min_val_loss': min_val_loss,
        'min_val_ppl': min_val_ppl,
        'min_test_loss': min_test_loss,
        'min_test_ppl': min_test_ppl
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'ckpt.pt'))
    print(f"Saved checkpoint to {checkpoint_dir}")

def evaluate_and_log(model, data_path, criterion, device, wandb_log, mode, args):
    loader = get_batch(data_path, batch_size=args.batch_size, mode=mode, n_tokens_eval=args.n_tokens_eval, disable_tqdm=args.no_tqdm)
    loss, ppl = evaluate_model(model, loader, criterion, device, wandb_log, mode, args.n_layers)
    print('\n' + '='*10)
    return loss, ppl

def main_training_loop(model, train_data_path, val_data_path, test_data_path, model_id, args, vocab_dict=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.model_type == 'sith':
        checkpoint_dir = f'checkpoints/{args.model_type}_nlayers{args.n_layers}_ntau{args.ntau}_taumax{args.tau_max}_g{args.g}_k{args.k}_id{model_id}/'
    else:
        checkpoint_dir = f'checkpoints/{args.model_type}_nlayers{args.n_layers}_hs{args.hidden_size}_id{model_id}/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    min_train_loss = float('inf')
    min_val_loss = float('inf')
    min_test_loss = float('inf')

    for epoch in range(args.epochs):
        if args.sample_during_training:
            initial_sentence = "The quick brown fox"
            generated_sentence = sample_model(model, initial_sentence, vocab_dict, args)
            print(f"Generated sentence: {generated_sentence}")
        print('\n' + '='*80)
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        train_loader = get_batch(train_data_path, batch_size=args.batch_size, mode='train', disable_tqdm=args.no_tqdm)
        running_tokens, _, train_loss, train_ppl, _ = train_model(
            model, train_loader, criterion, optimizer, args.device, args.n_tokens_log, args.wandb_log, args.eval_on_log, args.n_accum_steps, args.n_layers, vocab_dict=vocab_dict
        )

        # Update minimum train metrics
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            if args.wandb_log:
                wandb.log({"min_train_loss": min_train_loss, "min_train_ppl": np.exp(min_train_loss)})

        val_loss, val_ppl = evaluate_and_log(model, val_data_path, criterion, args.device, args.wandb_log, 'val', args)
        # Update minimum val metrics
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if args.wandb_log:
                wandb.log({"min_val_loss": min_val_loss, "min_val_ppl": np.exp(min_val_loss)})

        if not args.no_test:
            test_loss, test_ppl = evaluate_and_log(model, test_data_path, criterion, args.device, args.wandb_log, 'test', args)
            # Update minimum test metrics
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                if args.wandb_log:
                    wandb.log({"min_test_loss": min_test_loss, "min_test_ppl": np.exp(min_test_loss)})
                ## only save the best model
                save_checkpoint(epoch, model, optimizer, train_loss, train_ppl, val_loss, val_ppl, test_loss, test_ppl, 
                                running_tokens, min_train_loss, np.exp(min_train_loss), min_val_loss, np.exp(min_val_loss), min_test_loss, np.exp(min_test_loss), checkpoint_dir
                                )
        # Example usage within the main_training_loop

        if args.save_every_epoch:
            save_checkpoint(epoch, model, optimizer, train_loss, train_ppl, val_loss, val_ppl, test_loss, test_ppl, 
                            running_tokens, min_train_loss, np.exp(min_train_loss), min_val_loss, np.exp(min_val_loss), min_test_loss, np.exp(min_test_loss), checkpoint_dir
                            )
        
        if args.wandb_log:
            wandb.log({"epoch": epoch+1})

def main():
    args = parse_args()

    if args.wandb_log:
        wandb.init(project="time-local-transformer", entity=args.entity, group=args.wandb_group_name, config=vars(args))
        
    if args.wandb_log:
        if 'SLURM_JOB_ID' in os.environ:
            wandb.config.update({'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            wandb.config.update({'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID')})

    vocab_size_path = os.path.join(args.data_dir, args.vocab_size_path)
    with open(vocab_size_path, 'rb') as f:
        vocab_size = pickle.load(f)
    
    if args.decode or args.sample_during_training:
        vocab_dict_path = os.path.join(args.data_dir, args.vocab_path)
        with open(vocab_dict_path, 'rb') as f:
            vocab_dict = pickle.load(f)
        vocab_dict = {idx: token for token, idx in vocab_dict.items()}
    else: 
        vocab_dict = None
    
    model_id = np.random.randint(0, 10000)
    if args.wandb_log:
        wandb.config.update({'model_id': model_id})

    # Print arguments in a formatted way
    print('\n' + '='*80 + '\n')
    print("Configuration Arguments:")
    print('\n' + '='*80)

    # Model type
    print(f"{'Model type':>30}: {args.model_type}")
    
    # Specific arguments for RNN/LSTM/GRU or SITH
    if args.model_type in ['rnn', 'lstm', 'gru']:
        print(f"{'RNN/LSTM/GRU arguments':>30}:")
        print(f"{'Hidden size':>30}: {args.hidden_size}")
    else:
        print(f"{'SITH arguments':>30}:")
        print(f"{'tau_max':>30}: {args.tau_max}")
        print(f"{'tau_min':>30}: {args.tau_min}")
        print(f"{'ntau':>30}: {args.ntau}")
        print(f"{'g':>30}: {args.g}")
        print(f"{'k':>30}: {args.k}")
    
    # Model configuration
    print(f"\n{'Model configuration':>30}:")
    print(f"{'Vocab size':>30}: {vocab_size}")
    print(f"{'Embedding dimension':>30}: {args.embedding_dim}")
    print(f"{'Number of heads':>30}: {args.n_heads}")
    print(f"{'Number of layers':>30}: {args.n_layers}")

    # Other arguments
    print(f"\n{'Other arguments':>30}:")
    print(f"{'Model ID':>30}: {model_id}")
    print(f"{'Device':>30}: {args.device}")
    print(f"{'Batch size':>30}: {args.batch_size}")
    print(f"{'Epochs':>30}: {args.epochs}")
    print(f"{'Learning rate':>30}: {args.learning_rate}")
    print(f"{'Data directory':>30}: {args.data_dir}")
    print(f"{'Vocab size path':>30}: {args.vocab_size_path}")
    print(f"{'Vocab path':>30}: {args.vocab_path}")
    print(f"{'Tokens log interval':>30}: {args.n_tokens_log}")
    print(f"{'Tokens eval':>30}: {args.n_tokens_eval}")
    print(f"{'Wandb logging':>30}: {args.wandb_log}")
    print(f"{'Wandb group name':>30}: {args.wandb_group_name}")
    print(f"{'Disable tqdm':>30}: {args.no_tqdm}")
    print(f"{'Eval each train log':>30}: {args.eval_on_log}")
    print(f"{'No test set':>30}: {args.no_test}")
    print(f"{'Gradient accumulation steps':>30}: {args.n_accum_steps}")
    print(f"{'Detach on accumulation':>30}: {args.detach_on_accum}")
    print(f"{'Compile model':>30}: {args.compile}")

    print('\n' + '='*80)
    print(f"Logging every {args.n_tokens_log} tokens in the train set")
    if args.n_tokens_eval:
        print(f"Evaluating on the first {args.n_tokens_eval} tokens of val and test set")
    else :
        print(f"Evaluating on the full val and test set")
    print(f"GLHF")
    print('\n' + '='*80 + '\n')

    # Instantiate Model
    model = TimeLocalTransformer(vocab_size, args).to(args.device)
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)


    # Train Model
    train_data_path = os.path.join(args.data_dir, 'train.bin')
    val_data_path = os.path.join(args.data_dir, 'val.bin')
    if args.no_test:
        test_data_path = None
    else:
        test_data_path = os.path.join(args.data_dir, 'test.bin')
    main_training_loop(model, train_data_path, val_data_path, test_data_path, model_id, args, vocab_dict)

if __name__ == "__main__":
    main()
