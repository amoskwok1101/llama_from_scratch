from typing import Optional
import torch
import time
import json
# from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from model import ModelArgs,Transformer

class Llama:
    def __init__(self, model:Transformer, tokenizer, model_args:ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    def build(device:str):
        model_args = ModelArgs()
        model = Transformer(model_args).to(device)
        
        return model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(ModelArgs(vocab_size=100)).to(device)
    total_len = 10
    tokens = torch.randint(0, 100, (1, 10)).to(device)
    output = []
    for cur_pos in tqdm(range(1,total_len)):
        with torch.no_grad():
            logits,_ = model(tokens[:,cur_pos-1:cur_pos],None, cur_pos,'infer')
            next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
            # print(next_token)
            # break
            # output.append(next_token.item())
