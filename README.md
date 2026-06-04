# Qwen3-VL-8B-Instruct LoRA SFT 最小链路

在 AIS A100 上用 LLaMA-Factory `main` 分支对本地 `Qwen3-VL-8B-Instruct` 跑一次能跑通的 LoRA SFT 并导出合并 ckpt。

约定工作根 `SFT_ROOT = /home/work/Category_filesystem_V3/youwei.wang/sft/`，本仓库已 clone 到 `$SFT_ROOT/lora_sft`。**所有步骤都在 `$SFT_ROOT` 下执行**。

## 你只需要改这两行
```bash
export CSV_NAME="training_data_extract_attributes_and_item_tags.csv"
export MODEL_PATH="/home/work/model_repo/Qwen3-VL-8B-Instruct"
```

## 执行步骤

1. 拉 LLaMA-Factory（落到 `$SFT_ROOT/LLaMA-Factory`）：
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
```

2. 建 SFT venv（uv 会自动拉一份 standalone CPython 3.11）：
```bash
bash lora_sft/1_python_venv.sh
```

3. CSV -> sharegpt JSON + 并发下载图片（默认 16 路，可 `export IMG_CONCURRENCY=...` 调）：
```bash
./.sft_venv/bin/python lora_sft/2_convert_csv_to_json.py
```

4. 注册数据集 + 渲染并拷贝 yaml 到 LLaMA-Factory：
```bash
bash lora_sft/3_install_to_llamafactory.sh
```

5. LoRA SFT（单卡 A100）：
```bash
cd LLaMA-Factory && \
CUDA_VISIBLE_DEVICES=0 \
../.sft_venv/bin/llamafactory-cli train \
  examples/train_lora/qwen3vl_8b_lora_sft.yaml
```

6. 导出合并 ckpt：
```bash
cd LLaMA-Factory && \
../.sft_venv/bin/llamafactory-cli export \
  examples/merge_lora/qwen3vl_8b_lora_merge.yaml
```

## 产物落点
- venv：`$SFT_ROOT/.sft_venv/`
- 训练数据：`$SFT_ROOT/lora_sft_data/train.json` + `$SFT_ROOT/lora_sft_data/images/`
- LoRA：`$SFT_ROOT/saves/qwen3vl-8b/lora/sft/`
- 合并模型：`$SFT_ROOT/saves/qwen3vl-8b/lora/merged/`
