"""
v2_pos_encoding.py —— 阶段 v2：引入正余弦位置编码 (解决了 "我爱你/你爱我"，但长句仍受限于 1/t 死板平均)
"""

import math
import torch
import torch.nn as nn
from utils import VOCAB_SIZE

LR = 0.03
STEPS = 250


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
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=8):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        h = self.pe(self.emb(x) * math.sqrt(self.d_model))
        h = self.proj(h) * mask
        ctx = h.cumsum(dim=1) / mask.cumsum(dim=1).clamp(min=1.0)
        return self.head(ctx)
