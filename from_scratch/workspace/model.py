"""
model.py —— 学习者迭代模型架构的主画布
从当前的 v0 (15行代码) 出发，跟随讲义一步步修改本文件，进化到 v6 完整 Transformer！
"""

import math
import torch
import torch.nn as nn
from utils import VOCAB_SIZE, SRC_VOCAB, TGT_VOCAB


class MyModel(nn.Module):
    """
    当前阶段：v0 极简查表模型 (只看当前最后 1 个词)
    在终端运行 ./run.sh 查看翻车现象，并在 train.py 或 forward 中手动 Debug！
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (Batch, Seq_len)
        h = self.emb(x)          # (Batch, Seq_len, d_model)
        logits = self.head(h)    # (Batch, Seq_len, vocab_size)
        return logits
