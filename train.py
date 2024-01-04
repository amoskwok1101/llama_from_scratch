from typing import Optional
import torch
import time
import json
# from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from model import ModelArgs,Transformer
with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
n = len(text)
train_data = text[:int(n*0.9)]
val_data = text[int(n*0.9):]
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
train_data = torch.tensor(encode(train_data), dtype=torch.long)
val_data = torch.tensor(encode(val_data), dtype=torch.long)

learning_rate = 1e-3
target_learning_rate = 1e-4
max_iters = 1000
batch_size=8
eval_interval = 200
block_size = 128
eval_iters = 100
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y,0,'train')
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
   
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(ModelArgs(vocab_size=vocab_size)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_lambda = lambda epoch: max(1 - epoch / (max_iters ), target_learning_rate / learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss= model(xb, yb,0,'train')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

