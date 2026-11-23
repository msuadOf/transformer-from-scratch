"""
v5_deep_transformer.py —— 阶段 v5 修复版：加入残差连接 (Residual) + LayerNorm + FFN，6 层网络重回 8/8 (GPT 原型)
"""

import math
import torch
import torch.nn as nn
from utils import VOCAB_SIZE

LR = 0.005
STEPS = 250


def make_pad_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(size, device="cpu"):
    return torch.tril(torch.ones(1, 1, size, size, device=device)).bool()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model=32, n_heads=4, d_ff=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h, mask):
        B, L, _ = h.shape
        Q = self.W_q(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scores = (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)).masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)

        # 1. 残差连接 + LayerNorm
        h = self.norm1(h + self.W_o(ctx))
        # 2. FFN + 残差连接 + LayerNorm
        h = self.norm2(h + self.ffn(h))
        return h


class MyModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=32, n_heads=4, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        h = self.pe(self.emb(x) * math.sqrt(self.d_model))
        mask = make_subsequent_mask(L, device=x.device) & make_pad_mask(x)
        for layer in self.layers:
            h = layer(h, mask)
        return self.head(h)
