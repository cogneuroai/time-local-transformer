# time-local-transformer

install requirements
```
python3 -m pip install torch torchtext==0.6 tqdm wandb
```

tokenize dataset
```
python3 data/wikitext2/prepare.py
```

train rnn version with default parameters
```
python3 train.py --model_type=rnn
```

train Memory Timeline version with default parameters
```
python3 train.py --model_type=sith --inverse_method=gaver
```