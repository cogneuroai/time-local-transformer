# time-local-transformer

Human language processing, characterized by its ability to capture long-range dependencies in sequential inputs, operates under the constraints of limited working memory. In contrast, state-of-the-art transformer models in artificial intelligence rely on access to the fixed context window, what deviates from the dynamic nature of human cognition.  Here we propose a novel approach to reconcile this disparity by integrating a computational model of working memory into the transformer architecture. This biologically-inspired modification constructs a time-local transformer, capable of learning complex dependencies without needing the full input history. Our findings demonstrate that this approach still preserves the capacity of transformers for effective sequence processing. This work is a step towards developing AI models that align more closely with the principles of human brain function, opening new avenues for understanding the neural underpinnings of language and cognition.

install requirements
```
python3 -m pip install torch tqdm wandb numpy click transformers
```

download and tokenize Wikitext2 dataset
```
python3 data/wikitext2/prepare.py
```

train RNN version with default parameters
```
torchrun --standalone --nproc_per_node 1 train.py --model_type=rnn
```

train Shift Register version with default parameters
```
torchrun --standalone --nproc_per_node 1 train.py --model_type=isith --delta_pulse
```

train Memory Timeline version with default parameters
```
torchrun --standalone --nproc_per_node 1 train.py --model_type=gaver_cell
```