#!/usr/bin/env bash
# ===========================================================================
# 0_config.sh - task_name -> detail_config_dict
#
# - 配置集中写在同目录 0_yaml_to_setting.py 的 TASK_SETTINGS
# - 可以 `source 0_config.sh <task_name>` 把变量带入当前 shell
# - 也可以直接 `bash 0_config.sh <task_name>` 打印现场摘要
# - 不传 task_name 时使用 0_yaml_to_setting.py 里的 DEFAULT_TASK_NAME
# ===========================================================================
set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_SETTINGS_FILE="$WORK_DIR/0_yaml_to_setting.py"
TASK_NAME_ARG="${1:-${TASK_NAME:-}}"

TASK_EXPORTS="$(
  python3 - "$TASK_SETTINGS_FILE" "$TASK_NAME_ARG" "$WORK_DIR" <<'PY'
import runpy
import shlex
import sys
from pathlib import Path

settings_file = Path(sys.argv[1])
task_name = sys.argv[2].strip()
work_dir = Path(sys.argv[3]).resolve()

try:
    namespace = runpy.run_path(str(settings_file))
except Exception as exc:
    raise SystemExit(f"ERROR: failed to read task settings {settings_file}: {exc}")

root = namespace.get("TASK_SETTINGS")
if not isinstance(root, dict):
    raise SystemExit(f"ERROR: TASK_SETTINGS must be a dict in {settings_file}")

if not task_name:
    task_name = str(namespace.get("DEFAULT_TASK_NAME") or "").strip()
if not task_name:
    raise SystemExit(f"ERROR: task name is required and DEFAULT_TASK_NAME is empty in {settings_file}")
if task_name not in root:
    raise SystemExit(f"ERROR: task '{task_name}' not found in {settings_file}")

setting = root[task_name]
if not isinstance(setting, dict):
    raise SystemExit(f"ERROR: task '{task_name}' must be a dict")
if str(setting.get("_ready", 0)).strip() != "1":
    raise SystemExit(f"ERROR: task '{task_name}' is not ready; set _ready=1 in {settings_file}")


def required_str(name: str) -> str:
    value = str(setting.get(name) or "").strip()
    if not value:
        raise SystemExit(f"ERROR: task '{task_name}'.{name} is required")
    return value


def positive_int(name, default=None):
    raw = setting.get(name, default)
    try:
        value = int(raw)
    except Exception:
        raise SystemExit(f"ERROR: task '{task_name}'.{name} must be a positive integer")
    if value <= 0:
        raise SystemExit(f"ERROR: task '{task_name}'.{name} must be a positive integer")
    return value


def abs_path(value):
    p = Path(value)
    return p if p.is_absolute() else work_dir / p


data_dir = abs_path(str(setting.get("data_dir") or "train_data"))
csv_name = required_str("csv_name")
train_json_name = str(setting.get("train_json_name") or "train.json").strip()
image_dir_name = str(setting.get("image_dir_name") or "images").strip()
if not train_json_name or not image_dir_name:
    raise SystemExit(f"ERROR: task '{task_name}' train_json_name/image_dir_name cannot be empty")

lf_dir = abs_path(str(setting.get("llamafactory_dir") or "LLaMA-Factory"))
train_yaml_tpl = abs_path(required_str("train_yaml_template"))
export_yaml_tpl = abs_path(required_str("export_yaml_template"))
train_yaml_name = required_str("train_yaml_name")
export_yaml_name = required_str("export_yaml_name")

values = {
    "TASK_NAME": task_name,
    "WORK_DIR": str(work_dir),
    "TASK_SETTINGS_FILE": str(settings_file),
    "CSV_NAME": csv_name,
    "MODEL_PATH": required_str("model_path"),
    "IMG_CONCURRENCY": positive_int("img_concurrency", 16),
    "DATA_DIR": str(data_dir),
    "CSV_PATH": str(data_dir / csv_name),
    "TRAIN_JSON": str(data_dir / train_json_name),
    "IMG_DIR": str(data_dir / image_dir_name),
    "VENV_DIR": str(abs_path(str(setting.get("venv_dir") or ".sft_venv"))),
    "LF_DIR": str(lf_dir),
    "SAVE_DIR": str(abs_path(required_str("save_dir"))),
    "MERGED_DIR": str(abs_path(required_str("merged_dir"))),
    "TRAIN_YAML_TPL": str(train_yaml_tpl),
    "EXPORT_YAML_TPL": str(export_yaml_tpl),
    "TRAIN_YAML_OUT": str(lf_dir / "examples" / "train_lora" / train_yaml_name),
    "EXPORT_YAML_OUT": str(lf_dir / "examples" / "merge_lora" / export_yaml_name),
    "DATASET_NAME": required_str("dataset_name"),
    "PER_DEVICE_TRAIN_BATCH_SIZE": positive_int("per_device_train_batch_size", 1),
    "GRADIENT_ACCUMULATION_STEPS": positive_int("gradient_accumulation_steps", 1),
}
values["PY"] = str(Path(values["VENV_DIR"]) / "bin" / "python")
values["LF_CLI"] = str(Path(values["VENV_DIR"]) / "bin" / "llamafactory-cli")

for name, value in values.items():
    print(f"export {name}={shlex.quote(str(value))}")
PY
)" || {
  return 1 2>/dev/null || exit 1
}
eval "$TASK_EXPORTS"

mkdir -p "$DATA_DIR" "$IMG_DIR"

# ---- 直接执行（非 source）时打印现场摘要 -----------------------------------
# BASH_SOURCE[0] == $0 时代表本文件是直接被 bash 调起，而不是被 source
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  exists() { [ -e "$1" ] && echo yes || echo no; }
  echo "==== lora_sft config ===="
  echo "TASK_NAME       : $TASK_NAME"
  echo "WORK_DIR        : $WORK_DIR"
  echo "CSV_NAME        : $CSV_NAME"
  echo "CSV_PATH        : $CSV_PATH (exists? $(exists "$CSV_PATH"))"
  echo "TRAIN_JSON      : $TRAIN_JSON"
  echo "IMG_DIR         : $IMG_DIR"
  echo "IMG_CONCURRENCY : $IMG_CONCURRENCY"
  echo "MODEL_PATH      : $MODEL_PATH (exists? $(exists "$MODEL_PATH"))"
  echo "VENV_DIR        : $VENV_DIR (exists? $(exists "$VENV_DIR"))"
  echo "LF_DIR          : $LF_DIR (exists? $(exists "$LF_DIR"))"
  echo "SAVE_DIR        : $SAVE_DIR"
  echo "MERGED_DIR      : $MERGED_DIR"
  echo "DATASET_NAME    : $DATASET_NAME"
  echo "TRAIN_YAML_TPL  : $TRAIN_YAML_TPL (exists? $(exists "$TRAIN_YAML_TPL"))"
  echo "TRAIN_YAML_OUT  : $TRAIN_YAML_OUT"
  echo "EXPORT_YAML_OUT : $EXPORT_YAML_OUT"
  echo "PER_DEVICE_TRAIN_BATCH_SIZE    : $PER_DEVICE_TRAIN_BATCH_SIZE"
  echo "GRADIENT_ACCUMULATION_STEPS    : $GRADIENT_ACCUMULATION_STEPS"
  echo "========================="
  echo "tip: edit 0_yaml_to_setting.py and choose TASK_NAME to switch model/data/yaml"
fi
