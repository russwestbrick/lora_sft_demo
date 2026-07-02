#!/usr/bin/env bash
# sync_hf_configs.sh
# 将原始 HuggingFace 模型的配置/tokenizer 等非权重文件，迁移到 checkpoint 导出目录
# 用法: bash sync_hf_configs.sh <src_dir> <dst_dir> [--dry-run]

set -euo pipefail

SRC_DIR="${1:?用法: $0 <src_dir> <dst_dir> [--dry-run]}"
DST_DIR="${2:?用法: $0 <src_dir> <dst_dir> [--dry-run]}"
DRY_RUN="${3:-}"

# ── 要迁移的文件列表（按类别整理，不含权重文件） ──────────────────
FILES=(
  # 1. 模型配置
  "config.json"
  "generation_config.json"

  # 2. Tokenizer 相关
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
  "merges.txt"
  "special_tokens_map.json"
  "added_tokens.json"
  "tokenizer.model"                  # SentencePiece 模型（部分架构用）

  # 3. 预处理器 / 处理器配置
  "preprocessor_config.json"
  "processor_config.json"
  "video_preprocessor_config.json"
  "image_preprocessor_config.json"

  # 4. Chat 模板
  "chat_template.jinja"

)

# ── 前置检查 ──────────────────────────────────────────────────────
if [[ ! -d "$SRC_DIR" ]]; then
  echo "❌ 源目录不存在: $SRC_DIR" >&2; exit 1
fi
if [[ ! -d "$DST_DIR" ]]; then
  echo "❌ 目标目录不存在: $DST_DIR" >&2; exit 1
fi

# ── 执行迁移 ──────────────────────────────────────────────────────
copied=0
skipped=0
backed_up=0

echo "======================================"
echo "SRC : $SRC_DIR"
echo "DST : $DST_DIR"
echo "MODE: $( [[ "$DRY_RUN" == "--dry-run" ]] && echo '预览 (dry-run)' || echo '执行' )"
echo "======================================"
echo ""

for f in "${FILES[@]}"; do
  src_path="$SRC_DIR/$f"

  # 源文件不存在则跳过
  if [[ ! -f "$src_path" ]]; then
    echo "  SKIP  $f  (源文件不存在)"
    ((skipped++)) || true
    continue
  fi

  dst_path="$DST_DIR/$f"

  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    # dry-run 只打印
    if [[ -f "$dst_path" ]]; then
      echo "  COPY  $f  (目标已有 → 会先备份为 $f.bak)"
    else
      echo "  COPY  $f  (目标无此文件 → 新增)"
    fi
    ((copied++)) || true
    continue
  fi

  # 目标已存在 → 备份
  if [[ -f "$dst_path" ]]; then
    cp -v "$dst_path" "${dst_path}.bak"
    ((backed_up++)) || true
  fi

  # 复制
  cp -v "$src_path" "$dst_path"
  ((copied++)) || true
done

echo ""
echo "--------------------------------------"
echo "复制: $copied  |  备份: $backed_up  |  跳过: $skipped"
echo "--------------------------------------"
