#!/usr/bin/env bash
set -euo pipefail

SFT_ROOT="/home/work/Category_filesystem_V3/youwei.wang/sft"
SFT_VENV="$SFT_ROOT/.sft_venv"
LF_DIR="$SFT_ROOT/LLaMA-Factory"

BASE_PY="$(command -v python3.11 || command -v python3)"
UV="$HOME/.local/bin/uv"
PY="$SFT_VENV/bin/python"

if ! command -v "$UV" >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

rm -rf "$SFT_VENV"
"$UV" venv --python 3.11 --seed "$SFT_VENV"

# torch 先单独装一遍，避免 llama-factory 解析 extras 时拉错 cuda 轮子
"$UV" pip install --python "$PY" \
  "torch>=2.4,<2.6" "torchvision" "torchaudio" \
  --index-url https://download.pytorch.org/whl/cu124

# llama-factory 主体
"$UV" pip install --python "$PY" -e "${LF_DIR}[torch,metrics]"

# 数据脚本用
"$UV" pip install --python "$PY" pillow requests pandas tqdm

echo "venv ready: $SFT_VENV"
"$PY" -c "import torch, transformers, peft, trl, llamafactory; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__); \
print('peft', peft.__version__, 'trl', trl.__version__); \
print('llamafactory', llamafactory.__version__)"
