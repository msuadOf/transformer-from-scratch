"""
v4_masked_attention.py —— 阶段 v4：加上因果掩码 (Subsequent Mask) & 填充掩码 (Padding Mask)，单层通关 8/8
"""

import math
import torch
import torch.nn as nn
from utils import VOCAB_SIZE

LR = 0.02
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


class MyModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=32, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.head = nn.Linear(d_model, vocab_size)
        self.last_attn = None

    def forward(self, x):
        B, L = x.shape
        h = self.pe(self.emb(x) * math.sqrt(self.d_model))

        Q = self.W_q(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(h).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 修复：叠加 下三角因果掩码 & Padding 掩码
        mask = make_subsequent_mask(L, device=x.device) & make_pad_mask(x)
        scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        self.last_attn = attn.detach()

        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.head(self.act(self.W_o(ctx)))
