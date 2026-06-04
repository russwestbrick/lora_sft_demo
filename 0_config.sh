#!/usr/bin/env bash
# ===========================================================================
# 0_config.sh - 唯一配置源
#
# - 可以 `source 0_config.sh` 把所有变量带入当前 shell
# - 也可以直接 `bash 0_config.sh` 跑一次，会打印现场摘要供肉眼复核
# - 步骤脚本（1_/2_/3_）会自动 source 本文件，所以即使没 source 也不影响后续
# ===========================================================================
set -euo pipefail

# ---- 用户可调（基本只改这三行） -------------------------------------------
: "${CSV_NAME:=training_data_extract_attributes_and_item_tags.csv}"
: "${MODEL_PATH:=/home/work/model_repo/Qwen3-VL-8B-Instruct}"
: "${IMG_CONCURRENCY:=16}"

# ---- 路径派生（不要写绝对字面量） ------------------------------------------
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 训练数据：CSV / train.json / images/ 全部统一在 train_data/
DATA_DIR="$WORK_DIR/train_data"
CSV_PATH="$DATA_DIR/$CSV_NAME"
TRAIN_JSON="$DATA_DIR/train.json"
IMG_DIR="$DATA_DIR/images"

# venv
VENV_DIR="$WORK_DIR/.sft_venv"
PY="$VENV_DIR/bin/python"
LF_CLI="$VENV_DIR/bin/llamafactory-cli"

# LLaMA-Factory
LF_DIR="$WORK_DIR/LLaMA-Factory"

# 训练 / 导出落点
SAVE_DIR="$WORK_DIR/saves/qwen3vl-8b/lora/sft"
MERGED_DIR="$WORK_DIR/saves/qwen3vl-8b/lora/merged"

# yaml 模板与渲染后的落点
TRAIN_YAML_TPL="$WORK_DIR/configs/lora_sft.yaml"
EXPORT_YAML_TPL="$WORK_DIR/configs/export.yaml"
TRAIN_YAML_OUT="$LF_DIR/examples/train_lora/qwen3vl_8b_lora_sft.yaml"
EXPORT_YAML_OUT="$LF_DIR/examples/merge_lora/qwen3vl_8b_lora_merge.yaml"

# 数据集在 dataset_info.json 中的名字
DATASET_NAME="extract_attrs_sft"

mkdir -p "$DATA_DIR" "$IMG_DIR"

export CSV_NAME MODEL_PATH IMG_CONCURRENCY
export WORK_DIR DATA_DIR CSV_PATH TRAIN_JSON IMG_DIR
export VENV_DIR PY LF_CLI LF_DIR
export SAVE_DIR MERGED_DIR
export TRAIN_YAML_TPL EXPORT_YAML_TPL TRAIN_YAML_OUT EXPORT_YAML_OUT
export DATASET_NAME

# ---- 直接执行（非 source）时打印现场摘要 -----------------------------------
# BASH_SOURCE[0] == $0 时代表本文件是直接被 bash 调起，而不是被 source
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  exists() { [ -e "$1" ] && echo yes || echo no; }
  echo "==== lora_sft config ===="
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
  echo "========================="
  echo "tip: 步骤脚本 1_/2_/3_ 会自动 source 本文件，无需手动 export"
fi
