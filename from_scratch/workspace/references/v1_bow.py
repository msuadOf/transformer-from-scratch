"""
v1_bow.py —— 阶段 v1：词袋前缀平均 (能看到整句前文，但无位置编码，死于 "I love you" vs "you love I")
"""

import torch
import torch.nn as nn
from utils import VOCAB_SIZE

LR = 0.03
STEPS = 250


class MyModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        h = self.proj(self.emb(x)) * mask
        # 前缀累积平均：位置 t 的特征 = 前 t 个词向量的算术平均
        ctx = h.cumsum(dim=1) / mask.cumsum(dim=1).clamp(min=1.0)
        return self.head(ctx)
