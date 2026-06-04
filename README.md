# Qwen3-VL-8B-Instruct LoRA SFT 最小链路

在 AIS A100 上用 LLaMA-Factory `main` 分支对本地 `Qwen3-VL-8B-Instruct` 跑一次能跑通的 LoRA SFT，并导出合并 ckpt。

**工作目录 = 本仓库目录**（下面记为 `WORK_DIR`）。`LLaMA-Factory/`、`.sft_venv/`、`train_data/`、`saves/` 都会落到 `WORK_DIR/` 下。

## 配置只改 0_config.sh 顶部三行
```bash
CSV_NAME="training_data_extract_attributes_and_item_tags.csv"
MODEL_PATH="/home/work/model_repo/Qwen3-VL-8B-Instruct"
IMG_CONCURRENCY=16
```

需要肉眼复核配置时：
```bash
cd WORK_DIR
bash 0_config.sh    # 只打印现场摘要，不会改任何文件
```

步骤脚本（1_/2_/3_）会自动 `source 0_config.sh`，所以无需手动 `source` 或 `export`。

## 执行步骤（一律 `cd WORK_DIR` 后执行）

0. 把 CSV 放到 `train_data/$CSV_NAME`（首次跑时 `mkdir -p train_data && mv <旧路径>/$CSV_NAME train_data/`）。

1. 拉 LLaMA-Factory：
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
```

2. 建 SFT venv（uv 会自动拉 standalone CPython 3.11）：
```bash
bash 1_python_venv.sh
```

3. CSV -> sharegpt JSON + 并发下载图片：
```bash
.sft_venv/bin/python 2_convert_csv_to_json.py
```

4. 注册数据集 + 渲染并拷贝 yaml 到 LLaMA-Factory：
```bash
bash 3_install_to_llamafactory.sh
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
- venv：`WORK_DIR/.sft_venv/`
- 训练数据：`WORK_DIR/train_data/{$CSV_NAME, train.json, images/}`
- LoRA：`WORK_DIR/saves/qwen3vl-8b/lora/sft/`
- 合并模型：`WORK_DIR/saves/qwen3vl-8b/lora/merged/`
