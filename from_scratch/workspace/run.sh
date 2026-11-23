#!/bin/zsh
# macOS 本机实验运行脚本 (zsh)
set -e

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

cd "$SCRIPT_DIR"

resolve_stage_file() {
    case "$1" in
        v0)      echo "references/v0_bigram.py" ;;
        v1)      echo "references/v1_bow.py" ;;
        v2)      echo "references/v2_pos_encoding.py" ;;
        v3)      echo "references/v3_attention_unmasked.py" ;;
        v4)      echo "references/v4_masked_attention.py" ;;
        v5_bare) echo "references/v5_bare_6layers.py" ;;
        v5)      echo "references/v5_deep_transformer.py" ;;
        v6)      echo "references/v6_encoder_decoder.py" ;;
        *)
            echo "❌ 未知阶段: $1 (可选: v0, v1, v2, v3, v4, v5_bare, v5, v6)" >&2
            exit 1
            ;;
    esac
}

CMD="${1:-train}"

case "$CMD" in
    train|run)
        echo "🚀 正在运行 train.py (使用 $PYTHON_BIN) ..."
        "$PYTHON_BIN" train.py
        ;;
    diff)
        if [[ -z "$2" ]]; then
            echo "用法:"
            echo "  ./run.sh diff v2       # 对比你当前的 model.py 与阶段 v2 的代码差异"
            echo "  ./run.sh diff v2 v3    # 对比阶段 v2 升级到 v3 改了哪几行代码"
            exit 1
        fi
        if [[ -z "$3" ]]; then
            TARGET_FILE="$(resolve_stage_file "$2")"
            echo "🔍 对比 当前 model.py <---> $TARGET_FILE :"
            git diff --no-index --color=always model.py "$TARGET_FILE" || true
        else
            FILE_A="$(resolve_stage_file "$2")"
            FILE_B="$(resolve_stage_file "$3")"
            echo "🔍 对比阶段演进 $FILE_A ---> $FILE_B :"
            git diff --no-index --color=always "$FILE_A" "$FILE_B" || true
        fi
        ;;
    checkout)
        if [[ -z "$2" ]]; then
            echo "用法: ./run.sh checkout <v0|v1|v2|v3|v4|v5_bare|v5|v6>"
            exit 1
        fi
        TARGET_FILE="$(resolve_stage_file "$2")"
        cp model.py model.py.bak
        cp "$TARGET_FILE" model.py
        echo "✅ 已将 model.py 切换至 [$2] ($TARGET_FILE)！（原文件已自动备份为 model.py.bak）"
        ;;
    list|help|-h|--help)
        echo "🛠️  Transformer 从零进化工作台 (macOS)"
        echo "---------------------------------------------------------"
        echo "  ./run.sh                  运行 train.py 训练并测试当前 model.py"
        echo "  ./run.sh diff v2          对比当前 model.py 与参考阶段 v2 的差异"
        echo "  ./run.sh diff v2 v3       查看从 v2 进化到 v3 具体改了哪几行代码"
        echo "  ./run.sh checkout v3      将 model.py 切换为 v3 阶段代码 (自动备份 model.py.bak)"
        echo ""
        echo "📌 可用阶段代号: v0, v1, v2, v3, v4, v5_bare, v5, v6"
        ;;
    *)
        echo "❌ 未知命令: $CMD (运行 ./run.sh help 查看帮助)"
        exit 1
        ;;
esac
