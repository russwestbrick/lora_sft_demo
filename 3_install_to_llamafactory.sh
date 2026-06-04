#!/usr/bin/env bash
set -euo pipefail

# 自身位置 -> SFT_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LF_DIR="$SFT_ROOT/LLaMA-Factory"
DATA_DIR="$LF_DIR/data"
CFG_DIR="$LF_DIR/examples/train_lora"
MERGE_DIR="$LF_DIR/examples/merge_lora"

: "${MODEL_PATH:?need to: export MODEL_PATH=/abs/path/to/Qwen3-VL-8B-Instruct (see README)}"

# 1) 数据：把 train.json 的绝对路径注册到 dataset_info.json
DS_JSON="$SFT_ROOT/lora_sft_data/train.json"
test -f "$DS_JSON" || { echo "missing $DS_JSON, run 2_convert_csv_to_json.py first"; exit 1; }

python3 - "$DATA_DIR/dataset_info.json" "$DS_JSON" <<'PY'
import json, sys, pathlib
info_path = pathlib.Path(sys.argv[1])
ds_json   = sys.argv[2]
info = json.loads(info_path.read_text()) if info_path.exists() else {}
info["extract_attrs_sft"] = {
    "file_name": ds_json,
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "images":   "images",
        "system":   "system"
    },
    "tags": {
        "role_tag":      "role",
        "content_tag":   "content",
        "user_tag":      "user",
        "assistant_tag": "assistant"
    }
}
info_path.parent.mkdir(parents=True, exist_ok=True)
info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
print("registered dataset: extract_attrs_sft ->", ds_json)
PY

# 2) yaml：把模板里的 __MODEL_PATH__ 替换成 $MODEL_PATH，输出到 LLaMA-Factory 仓库内
mkdir -p "$CFG_DIR" "$MERGE_DIR"

# 安全分隔符 '|'，避免 MODEL_PATH 含 '/' 时打架
sed "s|__MODEL_PATH__|${MODEL_PATH}|g" \
  "$SCRIPT_DIR/configs/lora_sft.yaml" \
  > "$CFG_DIR/qwen3vl_8b_lora_sft.yaml"

sed "s|__MODEL_PATH__|${MODEL_PATH}|g" \
  "$SCRIPT_DIR/configs/export.yaml" \
  > "$MERGE_DIR/qwen3vl_8b_lora_merge.yaml"

echo "---- rendered train yaml (head) ----"
head -n 5 "$CFG_DIR/qwen3vl_8b_lora_sft.yaml"
echo "---- rendered export yaml (head) ----"
head -n 5 "$MERGE_DIR/qwen3vl_8b_lora_merge.yaml"

echo "done."
