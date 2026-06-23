#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/0_config.sh" "${1:-${TASK_NAME:-}}"

# 校验前置条件
test -d "$LF_DIR" || { echo "missing $LF_DIR, run: git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory"; exit 1; }
test -f "$TRAIN_JSON" || { echo "missing $TRAIN_JSON, run 2_convert_csv_to_json.py first"; exit 1; }
test -n "$MODEL_PATH" || { echo "MODEL_PATH is empty, edit 0_yaml_to_setting.py"; exit 1; }

DATA_INFO="$LF_DIR/data/dataset_info.json"

# 1) 数据：把 TRAIN_JSON 的绝对路径注册到 dataset_info.json
python3 - "$DATA_INFO" "$TRAIN_JSON" "$DATASET_NAME" <<'PY'
import json, sys, pathlib
info_path = pathlib.Path(sys.argv[1])
ds_json   = sys.argv[2]
ds_name   = sys.argv[3]
info = json.loads(info_path.read_text()) if info_path.exists() else {}
info[ds_name] = {
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
print(f"registered dataset: {ds_name} -> {ds_json}")
PY

# 2) yaml：把模板里的占位符替换为当前 task 的设置，输出到 LLaMA-Factory 仓库内
mkdir -p "$(dirname "$TRAIN_YAML_OUT")" "$(dirname "$EXPORT_YAML_OUT")"

python3 - "$TRAIN_YAML_TPL" "$TRAIN_YAML_OUT" "$EXPORT_YAML_TPL" "$EXPORT_YAML_OUT" <<'PY'
import os
import re
import sys
from pathlib import Path

train_tpl, train_out, export_tpl, export_out = [Path(arg) for arg in sys.argv[1:]]
replacements = {
    "__MODEL_PATH__": os.environ["MODEL_PATH"],
    "__SAVE_DIR__": os.environ["SAVE_DIR"],
    "__MERGED_DIR__": os.environ["MERGED_DIR"],
    "__DATASET_NAME__": os.environ["DATASET_NAME"],
    "__PER_DEVICE_TRAIN_BATCH_SIZE__": os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"],
    "__GRADIENT_ACCUMULATION_STEPS__": os.environ["GRADIENT_ACCUMULATION_STEPS"],
}

def render(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(key, value)
    unresolved = sorted(set(re.findall(r"__[A-Z0-9_]+__", text)))
    if unresolved:
        raise SystemExit(f"[error] unresolved placeholder(s) in {src}: {', '.join(unresolved)}")
    dst.write_text(text, encoding="utf-8")
    print(f"rendered {src} -> {dst}")

render(train_tpl, train_out)
render(export_tpl, export_out)
PY

echo "---- rendered train yaml (head) ----"
head -n 5 "$TRAIN_YAML_OUT"
echo "---- rendered export yaml (head) ----"
head -n 5 "$EXPORT_YAML_OUT"

echo "done."
