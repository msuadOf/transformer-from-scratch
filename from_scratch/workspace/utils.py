"""
utils.py —— 脏活累活工具库（学习者无需修改本文件）
负责：
1. 构建 8 句试金石中英翻译数据集与词表
2. 序列编号转换、Batch Padding 对齐
3. 格式化打印 8 句测试翻译的体检对比表
4. macOS 原生苹方字体 (PingFang.ttc) 注意力热力图绘制
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ================= 1. 试金石数据集 (8 句涵盖 4 大关卡) =================
RAW_DATA = [
    ("hello world", "你好世界"),
    ("I love you", "我爱你"),
    ("you love I", "你爱我"),
    ("I love deep learning", "我爱深度学习"),
    ("I love machine learning", "我爱机器学习"),
    ("you love deep learning", "你爱深度学习"),
    ("apple is sweet", "苹果很甜"),
    ("Transformer is all you need", "你需要的就是变形金刚"),
]

# ================= 2. 单序列词表与数据 (用于 v0 ~ v5) =================
VOCAB = {'<pad>': 0, '<sep>': 1, '<eos>': 2}
for _src, _tgt in RAW_DATA:
    for _w in _src.split():
        if _w not in VOCAB:
            VOCAB[_w] = len(VOCAB)
    for _c in _tgt:
        if _c not in VOCAB:
            VOCAB[_c] = len(VOCAB)

IDX2WORD = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def encode_prompt(src_text: str) -> torch.Tensor:
    """把英文句子转为带 <sep> 结尾的输入张量 (1, L)，方便手动 Debug 测试单句"""
    ids = [VOCAB[w] for w in src_text.split()] + [VOCAB['<sep>']]
    return torch.tensor([ids], dtype=torch.long)


def encode_full_sample(src_text: str, tgt_text: str) -> torch.Tensor:
    """把 '英文 + <sep> + 中文' 转为完整训练序列张量 (1, L)，方便 Debug 训练态注意力"""
    ids = [VOCAB[w] for w in src_text.split()] + [VOCAB['<sep>']] + [VOCAB[c] for c in tgt_text]
    return torch.tensor([ids], dtype=torch.long)


def build_single_seq_batch(data_list=RAW_DATA):
    """构建单序列训练 Batch: seq_in_batch (8, 16), seq_out_batch (8, 16)"""
    in_list, out_list = [], []
    for src, tgt in data_list:
        s_ids = [VOCAB[w] for w in src.split()]
        t_ids = [VOCAB[c] for c in tgt]
        seq_in = s_ids + [VOCAB['<sep>']] + t_ids
        # 英文部分标签填 0 (<pad>)，只从 <sep> 位置开始计算预测中文的 Loss
        seq_out = [0] * len(s_ids) + t_ids + [VOCAB['<eos>']]
        in_list.append(torch.tensor(seq_in, dtype=torch.long))
        out_list.append(torch.tensor(seq_out, dtype=torch.long))
    in_batch = nn.utils.rnn.pad_sequence(in_list, padding_value=0, batch_first=True)
    out_batch = nn.utils.rnn.pad_sequence(out_list, padding_value=0, batch_first=True)
    return in_batch, out_batch


# ================= 3. 双序列词表与数据 (用于 v6 Encoder-Decoder) =================
SRC_VOCAB = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
TGT_VOCAB = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
for _src, _tgt in RAW_DATA:
    for _w in _src.split():
        if _w not in SRC_VOCAB:
            SRC_VOCAB[_w] = len(SRC_VOCAB)
    for _c in _tgt:
        if _c not in TGT_VOCAB:
            TGT_VOCAB[_c] = len(TGT_VOCAB)

IDX2SRC = {v: k for k, v in SRC_VOCAB.items()}
IDX2TGT = {v: k for k, v in TGT_VOCAB.items()}


def build_enc_dec_batch(data_list=RAW_DATA):
    """构建 Encoder-Decoder 训练 Batch: src_batch, tgt_in_batch, tgt_out_batch"""
    src_list, tgt_list = [], []
    for src, tgt in data_list:
        s = torch.tensor([SRC_VOCAB[w] for w in src.split()], dtype=torch.long)
        t = torch.tensor([TGT_VOCAB['<sos>']] + [TGT_VOCAB[c] for c in tgt] + [TGT_VOCAB['<eos>']], dtype=torch.long)
        src_list.append(s)
        tgt_list.append(t)
    src_b = nn.utils.rnn.pad_sequence(src_list, padding_value=0, batch_first=True)
    tgt_b = nn.utils.rnn.pad_sequence(tgt_list, padding_value=0, batch_first=True)
    return src_b, tgt_b[:, :-1], tgt_b[:, 1:]


# ================= 4. 推理与格式化成绩单打印 =================
@torch.no_grad()
def evaluate_model(model: nn.Module, final_loss: float = None, max_gen: int = 12):
    """
    自动兼容单序列模型 (v0~v5) 与 Encoder-Decoder 模型 (v6，含 encode/decode 方法)
    逐句运行贪心自回归解码并打印对比表
    """
    model.eval()
    is_enc_dec = hasattr(model, "encode") and hasattr(model, "decode")
    correct = 0

    print("╔══════════════════════════════════════════════════════════════════════╗")
    if final_loss is not None:
        print(f"║ 📉 最终训练 Loss: {final_loss:<50.4f} ║")
        print("╟──────────────────────────────────────────────────────────────────────╢")

    for src, tgt in RAW_DATA:
        if not is_enc_dec:
            seq = [VOCAB[w] for w in src.split()] + [VOCAB['<sep>']]
            pred_chars = []
            for _ in range(max_gen):
                x = torch.tensor([seq], dtype=torch.long)
                logits = model(x)
                next_id = logits[0, -1].argmax().item()
                if next_id == VOCAB['<eos>']:
                    break
                seq.append(next_id)
                pred_chars.append(IDX2WORD[next_id])
            pred_str = "".join(pred_chars)
        else:
            s = torch.tensor([[SRC_VOCAB[w] for w in src.split()]], dtype=torch.long)
            src_mask = (s != 0).unsqueeze(1).unsqueeze(2)
            memory = model.encode(s, src_mask)
            ys = [TGT_VOCAB['<sos>']]
            for _ in range(max_gen):
                t = torch.tensor([ys], dtype=torch.long)
                tgt_mask = torch.tril(torch.ones(1, 1, t.size(1), t.size(1))).bool()
                out = model.decode(t, memory, src_mask, tgt_mask)
                next_id = out[0, -1].argmax().item()
                if next_id == TGT_VOCAB['<eos>']:
                    break
                ys.append(next_id)
            pred_str = "".join(IDX2TGT[i] for i in ys[1:])

        ok = (pred_str == tgt)
        if ok:
            correct += 1
        mark = "✅" if ok else "❌"
        print(f"║ {mark} {src:<28} -> 预测: {pred_str:<12} (标准: {tgt})")

    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"📊 综合通关率: {correct} / {len(RAW_DATA)}\n")
    return correct


# ================= 5. macOS 原生中文字体注意力热力图 =================
def plot_attention(attn_2d, x_tokens, y_tokens, title="Attention Weights"):
    """使用 macOS 系统自带 STHeiti Medium.ttc 绘制注意力权重矩阵热力图"""
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
    zh_font = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = [zh_font.get_name(), "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    if isinstance(attn_2d, torch.Tensor):
        attn_2d = attn_2d.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=110)
    sns.heatmap(
        attn_2d[:len(y_tokens), :len(x_tokens)],
        annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=x_tokens, yticklabels=y_tokens,
        vmin=0.0, vmax=1.0, ax=ax
    )
    ax.set_title(title, fontproperties=zh_font, fontsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(zh_font)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
    plt.tight_layout()
    plt.show()
