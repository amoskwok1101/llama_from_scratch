import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # This will be set when we do tokenization
    multiple_of: int = 256
    ffn_dim_multipler: Optional[int] = None
    norm_eps: float = 1e-5

    # Needed for kv cache
    max_seq_len: int = 512
    max_batch_size: int = 8

    device:str = 'cuda'

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float=10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"
    # Build the theta parameters
    # accroding to the formula theta_i = 10000^(-2(i-1)/dim) for i in [1, dim/2]
    # shape: (head_dim//2)
    theta_numerator = torch.arange(0, head_dim,2).float()
    theta = 1.0/(theta ** (theta_numerator / head_dim)).to(device)
    # Construct the position
    # Shape: (seq_len)
    m = torch.arange(seq_len,device=device)
    # Mulitply each theta by each position using the outer product
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    # each elem in freqs_complex = cos(m_p*theta_q) + i*sin(m_p*theta_q)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device:str) -> torch.Tensor:
    # (B,Seq_len,H,Head_dim) -> (B,Seq_len,H,Head_dim//2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1,2))
    # (Seq_len, Head_dim//2) -> (1,Seq_len,1,Head_dim//2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B,Seq_len,H,Head_dim//2) * (1,Seq_len,1,Head_dim//2) -> (B,Seq_len,H,Head_dim//2
    x_rotated = x_complex * freqs_complex
    # (B,Seq_len,H,Head_dim//2) -> (B,Seq_len,H,Head_dim//2,2)
    x_out = torch.view_as_real(x_rotated)
    # (B,Seq_len,H,Head_dim//2,2) -> (B,Seq_len,H,Head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (x[:,:,None,:]
                .expand(batch_size, seq_len, n_kv_heads,n_rep, head_dim)
                .reshape(batch_size, seq_len, n_rep*n_kv_heads, head_dim)
                )
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        

    def _norm(self, x):
        # (B, Seq_len, Dim) * (B,Seq_len,1) -> (B,Seq_len,Dim)
        # rsqrt = 1/sqrt
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    def forward(self, x):
        # (Dim) * (B,Seq_len,Dim) -> (B,Seq_len,Dim)
        return self._norm(x.float()).type_as(x) * self.weight

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        # number of heads for the kv
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number of head for query
        self.n_heads_q = args.n_heads
        # how many times the kv should be repeated to match query
        self.n_rep = self.n_heads_q//self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self,x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor,mode:str):
        batch_size, seq_len, dim = x.shape
        # (B,1,Dim) -> (B,1,H_Q*Head_dim)
        xq = self.wq(x)
        # (B,1,Dim) -> (B,H_KV*Head_dim)
        xk = self.wk(x)
        # (B,1,Dim) -> (B,H_KV*Head_dim)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # apply rotary embeddings to query and key
        xq = apply_rotary_embeddings(xq, freqs_complex,device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex,device=x.device)

        if mode!='train':
            # replace the entry in the cache for this new token
            self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
            self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

            # Retrieve the kv pairs from the cache
            keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
            values = self.cache_v[:batch_size, 0:start_pos+seq_len]
        else:
            keys = xk
            values = xv

        # Repeat the heads for the kv to match the query
        keys = repeat_kv(keys,self.n_rep)
        values = repeat_kv(values,self.n_rep)

        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2).to(x.device)
        values = values.transpose(1,2).to(x.device)
        # print device of keys and values
        
        scores = torch.matmul(xq, keys.transpose(2,3))/math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1).type_as(x)

        out = torch.matmul(scores, values)
        out = (out.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        out = self.wo(out)
        # (B,Seq_len,Dim)
        return out
        
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        
        hidden_dim = args.dim * 4 
        hidden_dim = int(2*hidden_dim/3) if args.ffn_dim_multipler is None else int(args.dim * args.ffn_dim_multipler)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim,bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim,bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish*x_V
        x = self.w2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else self.n_heads
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        # norm before ffn
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        # self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor,mode:str) -> torch.Tensor:
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex,mode)
        out = h+self.feed_forward(self.ffn_norm(h))
        return x

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size!=-1, "vocab_size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.token_embedding = nn.Embedding(self.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args))
        
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size,bias = False)
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads, 
                                                              self.args.max_seq_len*2, 
                                                              self.args.device)

    def forward(self, tokens:torch.Tensor,target:torch.Tensor,start_pos: int,mode:str):
        # (B,seq_len)
        batch_size, seq_len = tokens.shape
        # assert seq_len == 1, "Only one token at a time"
        # (B, seq_len) -> (B, seq_len, dim)
        h = self.token_embedding(tokens)

        # Retrieve the pairs (m, theta) corresponding to the position [start_pos, start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
        # print(freqs_complex)
        # Consecutively apply the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos,freqs_complex,mode)
        h = self.norm(h)
        logits = self.output(h).float()
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            # logits = logits.view(B*T, C)
            # targets = targets.view(B*T)
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))
        # print(logits.shape)
        return logits, loss
