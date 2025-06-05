import argparse
import datetime
import json
import os
import pickle
from contextlib import nullcontext

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
import click
from typing import Optional
from dataclasses import dataclass
#from isith_sequential import iSITH
from isith import iSITH
from gaver_cell import GaverCell
from gaver_cell_F import GaverCellF
from post_fornberg_F import PostFornbergCell
import math
import time
import inspect


def ddp_setup():
    # Only setup DDP for CUDA devices
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl")


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=config.n_embd, 
            num_heads=config.n_head, 
            batch_first=True
        )
        self.config = config
    
    def forward(self, input_sequence, rnn_output):
        actual_batch_size = input_sequence.shape[0]

        if self.config.model_type == 'gaver_cell_F':
            self.hidden_size_check = int(self.config.k * self.config.hidden_size)
        elif self.config.model_type == 'post_fornberg_F':
            self.hidden_size_check = int(self.config.k + self.config.hidden_size)
        else:
            self.hidden_size_check = self.config.hidden_size
        
        flat_hidden_units = einops.rearrange(
            rnn_output,
            'b t e h -> (b t) h e',
            b=actual_batch_size,
            t=self.config.block_size,
            e=self.config.n_embd,
            h=self.hidden_size_check
        )
        flat_query = einops.rearrange(
            input_sequence,
            'b t e -> (b t) 1 e',
            b=actual_batch_size,
            t=self.config.block_size,
            e=self.config.n_embd
        )
        flat_kv = torch.cat([flat_hidden_units, flat_query], dim=1)
        attn_output, _ = self.multihead_attention(flat_query, flat_kv, flat_kv)
        
        return einops.rearrange(
            attn_output, 
            '(b t) 1 e -> b t e',
            b=actual_batch_size, 
            t=self.config.block_size, 
            e=self.config.n_embd
        )

class BlockRNN(nn.Module):
    def __init__(self, config):
        super(BlockRNN, self).__init__()
        self.config = config
        
        if config.model_type == 'isith':
            self.temporal = iSITH(
                ntau=config.hidden_size,
                tau_min=1,
                tau_max=config.block_size,
                buff_max=config.block_size,
                dt=1.0,
                k=config.k,
                g=0.0,
                delta_pulse=config.delta_pulse,
                gaussian=config.gaussian,
                F=config.F,
                device=int(os.environ["LOCAL_RANK"])
            )

        elif config.model_type == 'rnn':
            self.temporal = nn.ModuleList([
                nn.RNNCell(
                    input_size=1, 
                    hidden_size=config.hidden_size
                ) for _ in range(config.n_embd)
            ])
            # Weight_ih: (n_embd, hidden_size, 1)
            self.weight_ih = nn.Parameter(torch.stack([cell.weight_ih for cell in self.temporal]))
            # Bias_ih: (n_embd, hidden_size)
            self.bias_ih = nn.Parameter(torch.stack([cell.bias_ih for cell in self.temporal]))
            # Weight_hh: (n_embd, hidden_size, hidden_size)
            self.weight_hh = nn.Parameter(torch.stack([cell.weight_hh for cell in self.temporal]))
            # Bias_hh: (n_embd, hidden_size)
            self.bias_hh = nn.Parameter(torch.stack([cell.bias_hh for cell in self.temporal]))
            del self.temporal

        elif config.model_type == 'lstm':
            self.temporal = nn.ModuleList([
                nn.LSTMCell(
                    input_size=1,
                    hidden_size=config.hidden_size
                ) for _ in range(config.n_embd)
            ])
            # LSTMCell parameters are pre-stacked for the 4 gates.
            # Shape: (n_embd, 4 * hidden_size, 1)
            self.weight_ih = nn.Parameter(torch.stack([cell.weight_ih for cell in self.temporal]))
            # Shape: (n_embd, 4 * hidden_size, hidden_size)
            self.weight_hh = nn.Parameter(torch.stack([cell.weight_hh for cell in self.temporal]))
            # Shape: (n_embd, 4 * hidden_size)
            self.bias_ih = nn.Parameter(torch.stack([cell.bias_ih for cell in self.temporal]))
            # Shape: (n_embd, 4 * hidden_size)
            self.bias_hh = nn.Parameter(torch.stack([cell.bias_hh for cell in self.temporal]))

            del self.temporal

        elif config.model_type =='gaver_cell':
            self.temporal = GaverCell(
                    tau_min=1,
                    tau_max=config.block_size,
                    n_taus=config.hidden_size,
                    fn_evals=int(config.k),
                    g=0.0
                )

        elif config.model_type =='gaver_cell_F':
            self.temporal = GaverCellF(
                    tau_min=1,
                    tau_max=config.block_size,
                    n_taus=config.hidden_size,
                    fn_evals=int(config.k),
                    g=0.0
                )

        elif config.model_type =='post_fornberg_F':
            self.temporal = PostFornbergCell(
                    tau_min=1,
                    tau_max=config.block_size,
                    n_taus=config.hidden_size,
                    k=int(config.k),
                    g=0.0
                )

        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
            
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        actual_batch_size = x.shape[0]
        if self.config.model_type == 'isith':
            x = einops.rearrange(
                x,
                'b t e -> b t e',
                b=actual_batch_size,
                t=self.config.block_size,
                e=self.config.n_embd
            )

            x_reshaped = einops.rearrange(
                x, 
                'b t e -> b 1 e t',
                b=actual_batch_size,
                t=self.config.block_size,
                e=self.config.n_embd
            )
            
            isith_output = self.temporal(x_reshaped)
            isith_output = einops.rearrange(
                isith_output,
                'b h e t -> b h e t',
                b=actual_batch_size,
                h=self.config.hidden_size,
                e=self.config.n_embd,
                t=self.config.block_size
            )
            
            rnn_output = einops.rearrange(
                isith_output,
                'b h e t -> b t e h',
                b=actual_batch_size,
                t=self.config.block_size,
                e=self.config.n_embd,
                h=self.config.hidden_size
            )
            
        elif self.config.model_type == 'gaver_cell':
            hidden_state = None
            outputs = []
            
            for t in range(self.config.block_size):
                if self.config.truncated_bptt and hidden_state is not None:
                    hidden_state = hidden_state.detach()
                
                input_t = einops.rearrange(
                    x[:, t, :], 
                    'b e -> b e', 
                    b=actual_batch_size, 
                    e=self.config.n_embd
                )
                
                til_f_t, hidden_state = self.temporal(input_t, hidden_state)
                outputs.append(einops.rearrange(til_f_t.bfloat16(), 'b e h -> b e h'))
            
            rnn_output = einops.rearrange(
                torch.stack(outputs),
                't b e h -> b t e h',
                b=actual_batch_size, 
                t=self.config.block_size, 
                e=self.config.n_embd, 
                h=self.config.hidden_size
            )

        elif self.config.model_type == 'gaver_cell_F':
            hidden_state = None
            outputs = []
            
            for t in range(self.config.block_size):
                if self.config.truncated_bptt and hidden_state is not None:
                    hidden_state = hidden_state.detach()
                
                input_t = einops.rearrange(
                    x[:, t, :], 
                    'b e -> b e', 
                    b=actual_batch_size, 
                    e=self.config.n_embd
                )
                
                hidden_state = self.temporal(input_t, hidden_state)
                outputs.append(einops.rearrange(hidden_state.bfloat16(), 'b e h k -> b e (h k)', b=actual_batch_size, e=self.config.n_embd, k=int(self.config.k), h=self.config.hidden_size))

            
            rnn_output = einops.rearrange(
                torch.stack(outputs),
                't b e x -> b t e x',
                b=actual_batch_size, 
                t=self.config.block_size, 
                e=self.config.n_embd, 
                x=int(self.config.k * self.config.hidden_size)
            )

        elif self.config.model_type == 'post_fornberg_F':
            hidden_state = None
            outputs = []
            
            for t in range(self.config.block_size):
                if self.config.truncated_bptt and hidden_state is not None:
                    hidden_state = hidden_state.detach()
                
                input_t = einops.rearrange(
                    x[:, t, :], 
                    'b e -> b e', 
                    b=actual_batch_size, 
                    e=self.config.n_embd
                )
                
                hidden_state = self.temporal(input_t, hidden_state)
                outputs.append(einops.rearrange(hidden_state.bfloat16(), 'b e x -> b e x', b=actual_batch_size, e=self.config.n_embd, x=int(self.config.k + self.config.hidden_size)))

            
            rnn_output = einops.rearrange(
                torch.stack(outputs),
                't b e x -> b t e x',
                b=actual_batch_size, 
                t=self.config.block_size, 
                e=self.config.n_embd, 
                x=int(self.config.k + self.config.hidden_size)
            )
            
        elif self.config.model_type == 'rnn':

            # Initialize hidden state as a single tensor
            # Shape: (batch_size, n_embd, hidden_size)
            hidden_state = torch.zeros(actual_batch_size, self.config.n_embd, self.config.hidden_size, device=x.device)

            outputs = [] # To store hidden state output of each time step

            # Loop over time steps (this loop is usually necessary for RNNs)
            for t in range(self.config.block_size):
                if self.config.truncated_bptt:
                    # Detach hidden state for truncated BPTT
                    hidden_state = hidden_state.detach()

                # Get input for current time step: (batch_size, n_embd)
                input_t = x[:, t, :]
                #print('input_t', input_t.shape)
                # Reshape input for einsum: (batch_size, n_embd, 1) <- adds the 'input_size=1' dim
                input_t_reshaped = einops.rearrange(input_t, 'b e -> b e 1')

                # --- Vectorized RNN Cell Computation ---
                # Input to hidden term: W_ih @ x_t + b_ih
                # 'b e i, e h i -> b e h' computes batch-wise dot products for each embedding dim 'e'
                # - b: batch_size
                # - e: n_embd
                # - i: input_size (1)
                # - h: hidden_size
                input_gate = torch.einsum('b e i, e h i -> b e h', input_t_reshaped, self.weight_ih) + self.bias_ih.unsqueeze(0) # Add broadcastable bias

                # Hidden to hidden term: W_hh @ h_{t-1} + b_hh
                # 'b e j, e h j -> b e h' computes batch-wise matrix-vector products for each embedding dim 'e'
                # - b: batch_size
                # - e: n_embd
                # - j: hidden_size (previous hidden state dim)
                # - h: hidden_size (output hidden state dim)
                hidden_gate = torch.einsum('b e j, e h j -> b e h', hidden_state, self.weight_hh) + self.bias_hh.unsqueeze(0) # Add broadcastable bias

                # Compute new hidden state
                hidden_state = torch.tanh(input_gate + hidden_gate)
                # hidden_state shape: (batch_size, n_embd, hidden_size)

                # Store the output (which is the hidden state in basic RNN)
                outputs.append(hidden_state)
                # ----------------------------------------

            # Stack outputs along the time dimension
            # List of tensors [(b, e, h), (b, e, h), ...] -> (b, t, e, h)
            rnn_output = torch.stack(outputs, dim=1)

        elif self.config.model_type == 'lstm':

            # Initialize hidden state (h) and cell state (c) as single tensors
            # Shape: (batch_size, n_embd, hidden_size)
            h_t = torch.zeros(actual_batch_size, self.config.n_embd, self.config.hidden_size, device=x.device)
            c_t = torch.zeros(actual_batch_size, self.config.n_embd, self.config.hidden_size, device=x.device)

            outputs = [] # To store hidden state output (h_t) of each time step

            # Loop over time steps
            for t in range(self.config.block_size):
                if self.config.truncated_bptt:
                    # Detach both hidden and cell states for truncated BPTT
                    h_t = h_t.detach()
                    c_t = c_t.detach()

                # Get input for current time step: (batch_size, n_embd)
                input_t = x[:, t, :]
                # Reshape input for einsum: (batch_size, n_embd, 1)
                input_t_reshaped = einops.rearrange(input_t, 'b e -> b e 1')

                # --- Vectorized LSTM Cell Computation ---
                # Calculate combined gates = (W_ih @ x_t + b_ih) + (W_hh @ h_{t-1} + b_hh)
                # Note: The 'h' dimension in einsum output corresponds to 4*hidden_size here

                # Input term: einsum('b e i, e hi -> b e h', ...)
                # - b: batch, e: n_embd, i: input_size=1
                # - e: n_embd, h: 4*hidden_size, i: input_size=1  (weight_ih)
                # -> b: batch, e: n_embd, h: 4*hidden_size
                input_gate_term = torch.einsum('b e i, e hi -> b e h', input_t_reshaped, self.weight_ih) + self.bias_ih.unsqueeze(0)

                # Hidden term: einsum('b e j, e hj -> b e h', ...)
                # - b: batch, e: n_embd, j: hidden_size (h_t)
                # - e: n_embd, h: 4*hidden_size, j: hidden_size (weight_hh)
                # -> b: batch, e: n_embd, h: 4*hidden_size
                hidden_gate_term = torch.einsum('b e j, e hj -> b e h', h_t, self.weight_hh) + self.bias_hh.unsqueeze(0)

                # Combined gates before activation
                # Shape: (batch_size, n_embd, 4 * hidden_size)
                gates = input_gate_term + hidden_gate_term

                # Split the combined gates into i, f, g, o
                # Each chunk will have shape (batch_size, n_embd, hidden_size)
                i_raw, f_raw, g_raw, o_raw = torch.split(gates, self.config.hidden_size, dim=-1)

                # Apply activations
                i_t = torch.sigmoid(i_raw)
                f_t = torch.sigmoid(f_raw)
                g_t = torch.tanh(g_raw) # Cell gate uses tanh
                o_t = torch.sigmoid(o_raw)

                # Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
                c_t = f_t * c_t + i_t * g_t

                # Update hidden state: h_t = o_t * tanh(c_t)
                h_t = o_t * torch.tanh(c_t)
                # h_t shape: (batch_size, n_embd, hidden_size)
                # c_t shape: (batch_size, n_embd, hidden_size)

                # Store the hidden state output for this timestep
                outputs.append(h_t)
                # ----------------------------------------

            # Stack outputs along the time dimension
            # List of tensors [(b, e, h), ...] -> (b, t, e, h)
            rnn_output = torch.stack(outputs, dim=1)
            # lstm_output shape: (batch_size, sequence_length, n_embd, hidden_size)

        # Apply attention and MLP
        x = x + self.attn(self.ln_1(x), rnn_output)
        x = x + self.mlp(self.ln_2(x))

        return x

class TimeLocalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #print(config.vocab_size)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([BlockRNN(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)

    def forward(self, idx, targets=None, return_logits=False):

        x = self.transformer.wte(idx)

        for i, block in enumerate(self.transformer.h):
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]).float() # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        original_seq_length = self.config.block_size
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= original_seq_length else idx[:, -original_seq_length:]
            
            # Temporarily update sequence length to match current input
            self.config.block_size = idx_cond.size(1)
            
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, idx_cond)
            
            # restore original sequence length
            self.config.block_size = original_seq_length
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: DataLoader,
        save_every: int,
        save_snapshot_path: str,
        load_snapshot_path: str,
        resume_dir: str,
        args: argparse.Namespace,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.save_every = save_every
        self.epochs_run = 0
        self.iter_counter = 0 
        self.save_snapshot_path = save_snapshot_path
        self.load_snapshot_path = load_snapshot_path
        self.eval_every = args.eval_every
        self.ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.args = args
        self.logfile = args.logfile
        self.resume_dir = resume_dir

        model.to(self.gpu_id)
        if self.args.compile:
            print0("Compiling the model... (takes a ~minute)")
            unoptimized_model = model
            torch.set_float32_matmul_precision('high')
            self.model = torch.compile(model)
        else: self.model = model
            
        if resume_dir:
            print0(f"Resuming from {resume_dir}")
            self._load_snapshot(load_snapshot_path)
        
        if self.gpu_id == 0 and args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                group=args.wandb_group,
                config=vars(args),
            )

        model = DDP(self.model, device_ids=[self.gpu_id])
        self.model = model
        raw_model = model.module  
        self.optimizer = raw_model.configure_optimizers(weight_decay=self.args.weight_decay,
                                        learning_rate=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2),
                                        device_type=self.args.device_type, zero_stage=self.args.zero_stage)
        
        if self.args.dtype == 'bfloat16':
            self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        else:
            self.dtype = self.args.dtype
        
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = torch.amp.autocast(device_type=self.args.device_type, dtype=ptdtype) if (self.args.device_type == "cuda") else nullcontext()
        print0(f"Using {self.dtype} for training")

        # Add tracking for best losses
        self.best_val_loss = float('inf')
        self.best_test_loss = float('inf')
        self.should_save_snapshot = False  # Add tracking variable

        total_params = sum(p.numel() for p in model.parameters())
        if self.gpu_id == 0 and self.args.use_wandb:
            wandb.run.summary['total_parameters'] = total_params


    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.iter_counter = snapshot.get("ITER_COUNTER", 0)  # Load iter_counter with default 0
        # Load best losses if they exist in the snapshot
        self.best_val_loss = snapshot.get("BEST_VAL_LOSS", float('inf'))
        self.best_test_loss = snapshot.get("BEST_TEST_LOSS", float('inf'))
        print0(f"Resuming training from snapshot at Epoch {self.epochs_run}, Iteration {self.iter_counter}")
        print0(f"Loaded best val loss: {self.best_val_loss:.6f}")
        print0(f"Loaded best test loss: {self.best_test_loss:.6f}")
        print0(f"Loaded iter counter: {self.iter_counter}")
    

    # learning rate decay scheduler (cosine with warmup)
    def _get_lr(self) -> float:
        min_lr = self.args.learning_rate * self.args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if self.iter_counter < self.args.warmup_iters:
            return self.args.learning_rate * (self.iter_counter+1) / self.args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if self.iter_counter > self.args.lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.iter_counter - self.args.warmup_iters) / (self.args.lr_decay_iters - self.args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (self.args.learning_rate - min_lr)


    def _run_batch(self, source, targets):
        t0 = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        with self.ctx:
            _, loss = self.model(source, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        if self.args.decay_lr:
            lr = self._get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.optimizer.step()

        # Gather and average loss across all devices
        loss_tensor = torch.tensor([loss.item()], device=self.gpu_id)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        loss_tensor /= self.ddp_world_size

        if self.args.device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        tokens_per_second = self.ddp_world_size * self.args.batch_size * self.args.sequence_length / (t1-t0)
        estimated_training_time = (self.args.max_iters * (t1-t0)) / 3600  # Total training time in hours
        # Log metrics only on primary GPU
        if self.gpu_id == 0:
            metrics = {
                'train/loss': loss_tensor.item(),
                'train/perplexity': math.exp(loss_tensor.item()),
                'train/grad_norm': norm,
                'train/lr': lr,
                'train/tokens_per_second': tokens_per_second,
                'train/estimated_training_time': estimated_training_time,
                'train/iter': self.iter_counter,
                'train/epoch': self.epochs_run,
            }
            
            print(f"iter {self.iter_counter:4d}/{self.args.max_iters} | train loss {loss_tensor.item():.6f} | "
                  f"ppl {math.exp(loss_tensor.item()):.4f} | norm {norm:.4f} | lr {lr:.2e} | "
                  f"({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
            
            # Log to file
            if self.logfile:
                with open(self.logfile, "a") as f:
                    f.write(f"s:{self.iter_counter} trl:{loss_tensor.item():.6f}\n")
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log(metrics, step=self.iter_counter)

    @torch.no_grad()
    def _run_eval(self, data_loader, split_name):
        self.model.eval()
        running_loss = torch.tensor(0.0, device=self.gpu_id)
        total_batches = torch.tensor(0, device=self.gpu_id)
        
        for source, targets in data_loader:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            with self.ctx:
                _, loss = self.model(source, targets)
            running_loss += loss.item()
            total_batches += 1

        # Gather and average metrics across all devices
        torch.distributed.all_reduce(running_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_batches, op=torch.distributed.ReduceOp.SUM)
        
        if self.args.device_type == "cuda":
            torch.cuda.synchronize()
        
        final_loss = running_loss.item() / total_batches.item()
        
        # Log metrics only on primary GPU
        if self.gpu_id == 0:
            metrics = {
                f'{split_name}/loss': final_loss,
                f'{split_name}/perplexity': math.exp(final_loss),
            }
            
            # Log to file
            if self.logfile:
                with open(self.logfile, "a") as f:
                    f.write(f"s:{self.iter_counter} {split_name[:2]}l:{final_loss:.6f}\n")

            print0(f"{split_name:5s} loss {final_loss:.6f} | ppl {math.exp(final_loss):.4f}")
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log(metrics, step=self.iter_counter)

        # Track best losses and save if validation improves
        if split_name == 'val':
            if final_loss < self.best_val_loss:
                self.best_val_loss = final_loss
                # Log best validation metrics to wandb
                if self.gpu_id == 0 and self.args.use_wandb:
                    wandb.log({
                        'best_val/loss': self.best_val_loss,
                        'best_val/perplexity': math.exp(self.best_val_loss)
                    }, step=self.iter_counter)
                self.should_save_snapshot = True
        elif split_name == 'test':
            if final_loss < self.best_test_loss:
                self.best_test_loss = final_loss
                # Log best test metrics to wandb
                if self.gpu_id == 0 and self.args.use_wandb:
                    wandb.log({
                        'best_test/loss': self.best_test_loss,
                        'best_test/perplexity': math.exp(self.best_test_loss)
                    }, step=self.iter_counter)
        
        self.model.train()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {self.args.iters_per_epoch}")
        self.epochs_run = epoch
        self.train_data.sampler.set_epoch(epoch)
        
        for source, targets in self.train_data:
            if self.eval_every > 0 and self.iter_counter % self.eval_every == 0:
                if self.val_data:
                    self._run_eval(self.val_data, 'val')
                if self.test_data:
                    self._run_eval(self.test_data, 'test')
                # Check if we should save after evals are complete
                if self.gpu_id == 0 and self.iter_counter > 0:  # don't save on first eval pass
                    if self.args.save_every:
                        self._save_snapshot(epoch)
                    elif self.should_save_snapshot:
                        self._save_snapshot(epoch)
                    self.should_save_snapshot = False  # Reset the flag
                with open(self.logfile, "a") as f:
                    f.write("-"*20+"\n")
            
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
            
            self.iter_counter += 1

    def _save_snapshot(self, epoch):
        model_state = self.model.module.state_dict()
        snapshot = {
            "MODEL_STATE": model_state,
            "EPOCHS_RUN": epoch,
            "ITER_COUNTER": self.iter_counter,
            "BEST_VAL_LOSS": self.best_val_loss,
            "BEST_TEST_LOSS": self.best_test_loss,
        }
        torch.save(snapshot, self.save_snapshot_path)
        print0(f"snapshot saved at {self.save_snapshot_path}")
        
        with open(self.logfile, "a") as f:
            f.write(f"*:{self.best_val_loss:.6f}\n")

    def train(self, max_epochs: int):
        try:
            for epoch in range(self.epochs_run, max_epochs):
                self._run_epoch(epoch)
            if self.val_data:
                self._run_eval(self.val_data, 'val')
            if self.test_data:
                self._run_eval(self.test_data, 'test')
        finally:
            # Cleanup wandb on primary GPU
            if self.gpu_id == 0 and self.args.use_wandb:
                wandb.finish()

@dataclass
class TimeLocalConfig:
    version: str = "3.1"
    block_size: int = 8192
    batch_size: int = 1
    vocab_size: int = 128256
    model_type: str = "isith"
    k: float = 50
    truncated_bptt: bool = False
    hidden_size: int = 6
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    norm_eps: float = 1e-5
    delta_pulse: bool = False
    gaussian: bool = False
    F: bool = False
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        assert self.n_embd % self.n_head == 0

class MyDataset(Dataset):
    def __init__(self, dataset_name, split, seq_length):
        data_path = os.path.join('data', dataset_name, f'{split}.bin')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.seq_length = seq_length
        self.debug_counter = 0  # Add counter for debugging

        print0(f"{split} dataset loaded: {len(self.data)} tokens, sequence length: {seq_length}")

    def __len__(self):
        return (len(self.data) // self.seq_length) - (1 if len(self.data) % self.seq_length == 0 else 0)

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        source = self.data[start_idx:start_idx + self.seq_length].astype(np.int64)
        target = self.data[start_idx + 1:start_idx + 1 + self.seq_length].astype(np.int64)
        return torch.tensor(source), torch.tensor(target)


def load_train_objs(config):
    train_set = MyDataset(config.dataset_name, 'train', config.sequence_length)
    
    # Try to load validation and test sets, but continue if not found
    val_set = None
    test_set = None
    try:
        val_set = MyDataset(config.dataset_name, 'val', config.sequence_length)
    except FileNotFoundError:
        print0("Validation set not found, skipping validation")
    
    try:
        test_set = MyDataset(config.dataset_name, 'test', config.sequence_length)
    except FileNotFoundError:
        print0("Test set not found, skipping test evaluation")

    # Check for vocab_size.pkl file
    vocab_size_path = os.path.join('data', config.dataset_name, 'vocab_size.pkl')
    if os.path.exists(vocab_size_path):
        with open(vocab_size_path, 'rb') as f:
            vocab_size = pickle.load(f)
            config.vocab_size = vocab_size  # Update args with actual vocab size
            print0(f"Loaded vocabulary size from {vocab_size_path}: {config.vocab_size}")
    else:
        print0(f"Using default vocabulary size: {config.vocab_size}")

    model_args = dict(
        version="tlocalv2",
        block_size=config.sequence_length,
        vocab_size=config.vocab_size,
        batch_size=config.batch_size,
        n_layer=config.n_layers,
        n_head=config.n_heads,
        n_embd=config.embedding_dim,
        hidden_size=config.hidden_size,
        model_type=config.model_type,  
        k=config.k,
        truncated_bptt=config.truncated_bptt,
        ffn_dim_multiplier=config.ffn_dim_multiplier,
        multiple_of=config.multiple_of,
        norm_eps=config.norm_eps,
        delta_pulse=config.delta_pulse,
        gaussian=config.gaussian,
        F=config.F,
    )

    config = TimeLocalConfig(**model_args)
    model = TimeLocalTransformer(config)

    return train_set, val_set, test_set, model
    

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )

@dataclass
class TrainingConfig:
    # Model Architecture
    model_type: str
    k: float
    truncated_bptt: bool
    vocab_size: int
    hidden_size: int
    embedding_dim: int
    ffn_dim_multiplier: float
    n_heads: int
    n_layers: int
    sequence_length: int
    multiple_of: int
    norm_eps: float
    dropout: float
    flash: bool
    # Training Configuration
    tokens_per_iter: int
    total_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    beta1: float
    beta2: float
    init_std: float
    # Learning Rate Schedule
    decay_lr: bool
    warmup_iters: int
    learning_rate_decay_frac: float
    # Optimization & Performance
    compile: bool
    zero_stage: int
    dtype: str
    device_type: str
    # Logging & Checkpointing
    save_every: bool
    eval_every: int
    eval_batches: int
    # Data & Output Paths
    dataset_name: str
    output_dir: str
    # Weights & Biases Integration
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    wandb_group: Optional[str]
    delta_pulse: Optional[bool] = False
    gaussian: Optional[bool] = False
    F: Optional[bool] = False
    # Add logfile path
    # Runtime attributes (set after initialization)
    logfile: Optional[str] = None
    run_dir: Optional[str] = None
    resume_dir: Optional[str] = None
    save_snapshot_path: Optional[str] = None
    load_snapshot_path: Optional[str] = None
    max_iters: Optional[int] = None
    lr_decay_iters: Optional[int] = None
    iters_per_epoch: Optional[int] = None
    slurm_job_id: Optional[str] = None
    slurm_job_name: Optional[str] = None
    slurm_array_job_id: Optional[str] = None
    slurm_array_task_id: Optional[str] = None
    batch_size: Optional[int] = None

@click.command()
@click.option('--delta_pulse', is_flag=True, help='Use delta pulse filters')
@click.option('--gaussian', is_flag=True, help='Use gaussian filters')
@click.option('--f', is_flag=True, help='Use F filters')
@click.option('--truncated_bptt', is_flag=True, help='Use truncated BPTT (limit to 1 timestep)')
@click.option('--model_type', type=click.Choice(['lstm', 'rnn', 'gaver_cell', 'gaver_cell_F', 'isith', 'post_fornberg_F']), default='rnn',
              help='Type of temporal processing model to use')
@click.option('--k', type=float, default=50, help='k parameter for iSITH model')
@click.option('--vocab_size', type=int, default=50257, help='Vocabulary size')
@click.option('--embedding_dim', type=int, default=768, help='Dimension of embeddings')
@click.option('--hidden_size', type=int, default=None, help='Number of filters if iSITH, or hidden states if RNN')
@click.option('--ffn_dim_multiplier', type=float, default=1.3, help='Multiplier for hidden dim')
@click.option('--n_heads', type=int, default=12, help='Number of attention heads')
@click.option('--n_layers', type=int, default=12, help='Number of transformer layers')
@click.option('--sequence_length', type=int, default=1024, help='Maximum sequence length')
@click.option('--multiple_of', type=int, default=256, help='Hidden size will be a multiple of this')
@click.option('--norm_eps', type=float, default=1e-5, help='Normalization epsilon')
@click.option('--dropout', type=float, default=0.0, help='Dropout rate (0 good for pre-training)')
@click.option('--flash', is_flag=True, help='Use Flash Attention for faster training')
@click.option('--tokens_per_iter', type=int, default=1024, help='Number of tokens per iteration')
@click.option('--total_epochs', type=int, required=True, help='Total number of training epochs')
@click.option('--learning_rate', type=float, default=6e-4, help='Learning rate')
@click.option('--weight_decay', type=float, default=0.1, help='Weight decay coefficient')
@click.option('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold')
@click.option('--beta1', type=float, default=0.9, help='Adam beta1')
@click.option('--beta2', type=float, default=0.95, help='Adam beta2')
@click.option('--init_std', type=float, default=0.02, help='Standard deviation for weight initialization')
@click.option('--decay_lr', is_flag=True, default=True, help='Whether to decay learning rate')
@click.option('--warmup_iters', type=int, default=700, help='Number of iterations for warmup')
@click.option('--learning_rate_decay_frac', type=float, default=0.0, help='Learning rate decay fraction')
@click.option('--compile', is_flag=True, help='Use torch.compile to optimize the model')
@click.option('--zero_stage', type=int, default=0, help='ZeRO optimizer stage (0/1/2/3)')
@click.option('--dtype', type=click.Choice(['float16', 'bfloat16', 'float32']), default='bfloat16',
              help='Data type for training')
@click.option('--device_type', type=str, default='cuda', help='Device to use (cuda/cpu)')
@click.option('--save_every', type=bool, default=False, help='Save checkpoint every epoch (True), or just when val loss improves (False)')
@click.option('--eval_every', type=int, default=0, help='Evaluate model every N batches')
@click.option('--eval_batches', type=int, default=-1, help='Number of batches to evaluate (-1 for full epoch)')
@click.option('--dataset_name', type=str, required=True, help='Name of the dataset (will load from data/{dataset_name})')
@click.option('--output_dir', type=str, default='outputs', help='Base directory for outputs')
@click.option('--use_wandb', is_flag=True, help='Enable wandb logging')
@click.option('--resume_dir', type=str, default=None, help='Directory to resume from')
@click.option('--wandb_project', type=str, default='timelocalformer', help='WandB project name')
@click.option('--wandb_entity', type=str, default=None, help='WandB entity name')
@click.option('--wandb_run_name', type=str, default=None, help='WandB run name')
@click.option('--wandb_group', type=str, default=None, help='WandB group name')
def main(**kwargs):
    if 'f' in kwargs:
        kwargs['F'] = kwargs.pop('f')
    args = TrainingConfig(**kwargs)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert args.tokens_per_iter % args.sequence_length == 0
    args.batch_size = (args.tokens_per_iter // args.sequence_length) * ddp_world_size
    print0(f"calculated batch_size for {args.tokens_per_iter} tokens per iter, {args.sequence_length} sequence length, {ddp_world_size} GPUs: {args.batch_size}")
    
    # Determine base directory structure
    job_id = os.environ.get('SLURM_JOB_ID') 
    if args.use_wandb:
        sub_dir = f"wandb/{args.wandb_group}" if args.wandb_group else "wandb_no_group_name"
    else:
        sub_dir = "offline"
    
    if job_id:
        args.base_dir = os.path.join(args.output_dir, args.dataset_name, sub_dir, job_id)
    else:
        args.base_dir = os.path.join(args.output_dir, args.dataset_name, sub_dir)
    
    # resume directory must be in teh form wandb_group / timestamp
    # this allows you to make a new group and resume runs from other groups
    if args.resume_dir:
        # Load from original snapshot
        args.load_snapshot_path = os.path.join(args.output_dir, args.dataset_name, args.resume_dir, 'snapshot.pt')
        if not os.path.exists(args.load_snapshot_path):
            raise ValueError(f"Snapshot not found at: {args.load_snapshot_path}")
        
        # Create new directory for resumed run
        args.run_dir = os.path.join(args.base_dir, timestamp)
        args.save_snapshot_path = os.path.join(args.run_dir, 'snapshot.pt')
        args.logfile = os.path.join(args.run_dir, 'main.log')
        
        # Create the run directory
        os.makedirs(args.run_dir, exist_ok=True)
        
        if int(os.environ.get("RANK", 0)) == 0:
            with open(os.path.join(args.run_dir, 'resumed_from.txt'), 'w') as f:
                f.write(f"Resumed from: {os.path.join(args.base_dir, args.resume_dir)}\n\n")
    else:
        # Fresh run
        args.run_dir = os.path.join(args.base_dir, timestamp)
        args.logfile = os.path.join(args.run_dir, 'main.log')
        args.save_snapshot_path = os.path.join(args.run_dir, 'snapshot.pt')
        args.load_snapshot_path = None
        # Create the run directory
        if int(os.environ.get("RANK", 0)) == 0:
            os.makedirs(args.run_dir, exist_ok=True)

    slurm_vars = {
        # Job identification
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'job_name': os.environ.get('SLURM_JOB_NAME'),
        # Array specific information
        'array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
        'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
    }
    # Create empty logfile
    if int(os.environ.get("RANK", 0)) == 0:
        if args.logfile:
            with open(args.logfile, "w") as f:
                pass
        if any(slurm_vars.values()):
            args.slurm_job_id = slurm_vars['job_id']
            args.slurm_job_name = slurm_vars['job_name']
            args.slurm_array_job_id = slurm_vars['array_job_id']
            args.slurm_array_task_id = slurm_vars['array_task_id']
            slurm_dict = {k: v for k, v in slurm_vars.items() if v is not None} 
            with open(os.path.join(args.run_dir, 'slurm_vars.json'), 'w') as f:
                json.dump(slurm_dict, f, indent=4)
            args.wandb_run_name = f"{args.model_type}-{args.sequence_length}-{args.hidden_size}-{timestamp}-{args.slurm_job_id}-{args.slurm_array_job_id}-{args.slurm_array_task_id}"
        else:
            args.wandb_run_name = f"{args.model_type}-{args.sequence_length}-{args.hidden_size}-{timestamp}"
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ddp_setup()
    train_dataset, val_dataset, test_dataset, model = load_train_objs(args)
    train_data = prepare_dataloader(train_dataset, args.batch_size)
    
    # Only create dataloaders for available datasets
    val_data = prepare_dataloader(val_dataset, args.batch_size) if val_dataset else None
    test_data = prepare_dataloader(test_dataset, args.batch_size) if test_dataset else None

    # Calculate lr_decay_iters if not explicitly provided
    print0("train_data length (number of steps/iters per epoch): ", len(train_data))
    args.iters_per_epoch = len(train_data)
    args.max_iters = args.iters_per_epoch * args.total_epochs
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters
        print0(f"Setting lr_decay_iters to {args.lr_decay_iters} "
               f"({args.iters_per_epoch} iterations/epoch * {args.total_epochs} epochs)")
    else:
        args.lr_decay_iters = 600000
        print0(f"Warning: Could not determine dataset size. Using default lr_decay_iters={args.lr_decay_iters}")

    trainer = Trainer(model, train_data, val_data, test_data, args.save_every, args.save_snapshot_path, args.load_snapshot_path, args.resume_dir, args)
    
    # After trainer initialization, save wandb config if using wandb
    if args.use_wandb and int(os.environ.get("RANK", 0)) == 0:
        wandb_config = {
            'project': wandb.run.project,
            'entity': wandb.run.entity,
            'group': wandb.run.group,
            'run_name': wandb.run.name,
            'run_id': wandb.run.id,
            'run_url': wandb.run.get_url() if wandb.run else None
        }
        with open(os.path.join(args.run_dir, 'wandb_config.json'), 'w') as f:
            json.dump(wandb_config, f, indent=4)
    
    trainer.train(args.total_epochs)
    destroy_process_group()

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    main()

