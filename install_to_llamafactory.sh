#!/usr/bin/env bash
set -euo pipefail

SFT_ROOT="/home/work/Category_filesystem_V3/youwei.wang/sft"
LF_DIR="$SFT_ROOT/LLaMA-Factory"
DATA_DIR="$LF_DIR/data"
CFG_DIR="$LF_DIR/examples/train_lora"
MERGE_DIR="$LF_DIR/examples/merge_lora"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"   # = lora_sft 目录

# 1) 数据：直接通过绝对路径在 dataset_info.json 里指认
DS_JSON="$SFT_ROOT/lora_sft_data/train.json"
test -f "$DS_JSON" || { echo "missing $DS_JSON, run convert_csv_to_json.py first"; exit 1; }

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

# 2) yaml：拷到 LLaMA-Factory 仓库内
mkdir -p "$CFG_DIR" "$MERGE_DIR"
cp -v "$SRC_DIR/configs/lora_sft.yaml" "$CFG_DIR/qwen3vl_8b_lora_sft.yaml"
cp -v "$SRC_DIR/configs/export.yaml"   "$MERGE_DIR/qwen3vl_8b_lora_merge.yaml"

echo "done."
