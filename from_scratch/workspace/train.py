"""
train.py —— 程序主入口与手动 Debug 试验台
在 macOS 终端直接运行 ./run.sh 即可启动本脚本。
你可以在训练循环内部或底部的「手动 Debug 实验区」自由添加 print() 或打断点排查问题！
"""

import torch
import torch.nn as nn
import torch.optim as optim

import model as user_model
from model import MyModel
from utils import (
    VOCAB, VOCAB_SIZE, SRC_VOCAB, TGT_VOCAB,
    encode_prompt, encode_full_sample,
    build_single_seq_batch, build_enc_dec_batch,
    evaluate_model, plot_attention
)


def get_default_hyperparams(net: nn.Module):
    """根据模型复杂度自动选择最合适的默认学习率与步数（也可在 model.py 中定义 LR / STEPS 覆盖）"""
    if hasattr(net, "encode") and hasattr(net, "decode"):
        default_lr, default_steps = 0.005, 300
    elif hasattr(net, "layers"):
        default_lr, default_steps = 0.005, 250
    elif hasattr(net, "W_q") or hasattr(net, "self_attn"):
        default_lr, default_steps = 0.02, 250
    else:
        default_lr, default_steps = 0.03, 250

    lr = getattr(user_model, "LR", default_lr)
    steps = getattr(user_model, "STEPS", default_steps)
    return lr, steps


def main():
    torch.manual_seed(42)

    net = MyModel()
    lr, steps = get_default_hyperparams(net)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 判断当前是单序列模型 (v0~v5) 还是 Encoder-Decoder 模型 (v6)
    is_enc_dec = hasattr(net, "encode") and hasattr(net, "decode")
    if not is_enc_dec:
        seq_in_batch, seq_out_batch = build_single_seq_batch()
    else:
        src_batch, tgt_in_batch, tgt_out_batch = build_enc_dec_batch()

    # ================= 1. 训练循环 =================
    net.train()
    for step in range(steps):
        optimizer.zero_grad()

        if not is_enc_dec:
            logits = net(seq_in_batch)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), seq_out_batch.reshape(-1))
        else:
            logits = net(src_batch, tgt_in_batch)
            loss = criterion(logits.reshape(-1, len(TGT_VOCAB)), tgt_out_batch.reshape(-1))

        loss.backward()

        # 🛠️ [Debug 探针点 1]：想看深层网络 (v5) 第 1 层梯度是否消失？可以在 step == 0 时打印：
        # if step == 0 and hasattr(net, "layers"):
        #     print("Step 0 第0层梯度范数:", net.layers[0].W_q.weight.grad.norm().item())
        #     print("Step 0 最后层梯度范数:", net.layers[-1].W_q.weight.grad.norm().item())

        optimizer.step()

    # ================= 2. 跑 8 句试金石翻译体检 =================
    evaluate_model(net, final_loss=loss.item())

    # ================= 3. 🛠️ 手动 Debug 实验区 =================
    # 跟随讲义中的【手动 Debug 指南】，可以在下面取消注释或自己写几行 print 验证猜想：

    # --- [排查 v0 翻车]：为什么所有句子都翻译成同一句话？对比两句不同英文在 <sep> 处的输出 ---
    # x1, x2 = encode_prompt("hello world"), encode_prompt("I love you")
    # print("v0 在 <sep> 位置输出的最大差值:", (net(x1)[:, -1] - net(x2)[:, -1]).abs().max().item())

    # --- [排查 v1 翻车]：为什么 "I love you" 和 "you love I" 输出一模一样？ ---
    # x1, x2 = encode_prompt("I love you"), encode_prompt("you love I")
    # print("v1 对两句话在 <sep> 处的预测分布差值:", (net(x1)[:, -1] - net(x2)[:, -1]).abs().max().item())

    # --- [排查 v3 翻车]：为什么训练 Loss 0.068 近乎满分，推理却 0/8？看看训练时 <sep> 的注意力在看哪！ ---
    # sample = encode_full_sample("I love you", "我爱你")  # ["I", "love", "you", "<sep>", "我", "爱", "你"]
    # _ = net(sample)
    # print("训练时位置 3 (<sep>) 的注意力权重分布:", net.last_attn[0, :, 3, :].mean(dim=0).round(decimals=3))


if __name__ == "__main__":
    main()
