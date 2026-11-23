"""
v0_bigram.py —— 阶段 v0：极简查表模型 (只看当前最后 1 个词)
"""

import torch
import torch.nn as nn
from utils import VOCAB_SIZE

LR = 0.03
STEPS = 250


class MyModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.emb(x)
        return self.head(h)
