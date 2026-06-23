#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/0_config.sh" "${1:-${TASK_NAME:-}}"

UV="$HOME/.local/bin/uv"

if ! command -v "$UV" >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

rm -rf "$VENV_DIR"
# uv 内置按需下载 standalone CPython 3.11，宿主不需要预装 3.11
"$UV" venv --python 3.11 --seed "$VENV_DIR"

# torch 先单独装一遍，避免 llama-factory 解析 extras 时拉错 cuda 轮子
"$UV" pip install --python "$PY" \
  "torch>=2.4,<2.6" "torchvision" "torchaudio" \
  --index-url https://download.pytorch.org/whl/cu124

# llama-factory 主体
"$UV" pip install --python "$PY" -e "${LF_DIR}[torch,metrics]"

# 数据脚本用
"$UV" pip install --python "$PY" pillow requests pandas tqdm

echo "venv ready: $VENV_DIR"
"$PY" -c "import torch, transformers, peft, trl, llamafactory; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__); \
print('peft', peft.__version__, 'trl', trl.__version__); \
print('llamafactory', llamafactory.__version__)"
