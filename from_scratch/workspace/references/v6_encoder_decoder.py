"""
v6_encoder_decoder.py —— 阶段 v6：拆分为双向 Encoder + 交叉注意力 Decoder 的完整 Transformer
"""

import math
import torch
import torch.nn as nn
from utils import SRC_VOCAB, TGT_VOCAB

LR = 0.005
STEPS = 300


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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.last_attn = None

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.W_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        self.last_attn = attn.detach()

        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(ctx)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=32, n_heads=4, d_ff=128):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        src = self.norm1(src + self.self_attn(src, src, src, src_mask))
        src = self.norm2(src + self.ffn(src))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model=32, n_heads=4, d_ff=128):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        tgt = self.norm1(tgt + self.self_attn(tgt, tgt, tgt, tgt_mask))
        tgt = self.norm2(tgt + self.cross_attn(tgt, memory, memory, src_mask))
        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt


class MyModel(nn.Module):
    def __init__(self, src_v=len(SRC_VOCAB), tgt_v=len(TGT_VOCAB), d_model=32, n_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_v, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_v, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, tgt_v)

    def encode(self, src, src_mask):
        h = self.pe(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            h = layer(h, src_mask)
        return h

    def decode(self, tgt, memory, src_mask, tgt_mask):
        h = self.pe(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            h = layer(h, memory, src_mask, tgt_mask)
        return self.head(h)

    def forward(self, src, tgt_in):
        src_mask = make_pad_mask(src)
        tgt_mask = make_pad_mask(tgt_in) & make_subsequent_mask(tgt_in.size(1), device=src.device)
        memory = self.encode(src, src_mask)
        return self.decode(tgt_in, memory, src_mask, tgt_mask)
