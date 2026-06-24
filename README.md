# Qwen3-VL LoRA SFT 链路

用 LLaMA-Factory `main` 分支对本地 Qwen3-VL 类模型跑 LoRA SFT，并导出合并 ckpt。

**工作目录 = 本仓库目录**（下面记为 `WORK_DIR`）。`LLaMA-Factory/`、`.sft_venv/`、`train_data/`、`saves/` 都会落到 `WORK_DIR/` 下。

## 配置只改 0_yaml_to_setting.py

每个任务都写在 `0_yaml_to_setting.py` 的 `TASK_SETTINGS` 里，切换 `TASK_NAME` 就能切换 model、CSV、dataset、yaml 模板、渲染后的 yaml 文件名和输出目录。

需要肉眼复核配置时：

```bash
cd WORK_DIR
bash 0_config.sh qwen3vl_8b_extract_attrs_h100_4gpu
```

步骤脚本会自动 `source 0_config.sh`。不传 task 时使用 `0_yaml_to_setting.py` 里的 `DEFAULT_TASK_NAME`。

## 执行步骤（一律 `cd WORK_DIR` 后执行）

0. 把 CSV 放到当前 task 配置的 `CSV_PATH`。先用下面命令查看具体路径：

```bash
bash 0_config.sh qwen3vl_8b_extract_attrs_h100_4gpu
```

1. 拉 LLaMA-Factory：

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
```

2. 建 SFT venv（uv 会自动拉 standalone CPython 3.11）：

```bash
bash 1_python_venv.sh qwen3vl_8b_extract_attrs_h100_4gpu
```

3. CSV -> sharegpt JSON + 并发下载图片：

```bash
.sft_venv/bin/python 2_convert_csv_to_json.py qwen3vl_8b_extract_attrs_h100_4gpu
```

4. 清洗 train JSON，校验图片路径和 LLaMA-Factory 等比截断后 `<image>` 是否保留：

```bash
.sft_venv/bin/python 2_check_train_json.py qwen3vl_8b_extract_attrs_h100_4gpu
```

5. 注册数据集 + 渲染并拷贝 yaml 到 LLaMA-Factory：

```bash
bash 3_install_to_llamafactory.sh qwen3vl_8b_extract_attrs_h100_4gpu
```

5.1 LoRA SFT（单卡 smoke test）：

```bash
source ./0_config.sh qwen3vl_8b_extract_attrs_h100_4gpu && \
cd "$LF_DIR" && \
CUDA_VISIBLE_DEVICES=0 \
../.sft_venv/bin/llamafactory-cli train "$TRAIN_YAML_OUT"
```

这一步用于先在单卡上调通数据、yaml、模型路径和 LLaMA-Factory 环境，不依赖 AIS 分布式变量。

5.2 LoRA SFT（AIS entry point / torchrun）：

```bash
cd /home/work/slamm/youwei.wang/lora_sft_demo && \
UV="$HOME/.local/bin/uv" && \
[ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | sh && \
RANK_VENV="/home/work/.sft_venv" && \
[ -x "$RANK_VENV/bin/python" ] || (rm -rf "$RANK_VENV" && "$UV" venv --python 3.11 --seed "$RANK_VENV" && "$UV" pip install --python "$RANK_VENV/bin/python" "torch>=2.4,<2.6" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu124 && "$UV" pip install --python "$RANK_VENV/bin/python" -e "./LLaMA-Factory[metrics]" pillow requests pandas tqdm) && \
source ./0_config.sh qwen3vl_8b_extract_attrs_h100_4gpu && \
cd "$LF_DIR" && \
"$RANK_VENV/bin/python" -m torch.distributed.run \
  --nproc_per_node 1 \
  --nnodes "$WORLD_SIZE" \
  --node_rank "$RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  -m llamafactory.cli train "$TRAIN_YAML_OUT"
```

这一步只依赖 task name、准备阶段导出的 `LF_DIR` 和 `TRAIN_YAML_OUT`，以及 AIS 注入的 `WORLD_SIZE`、`RANK`、`MASTER_ADDR`、`MASTER_PORT`。AIS 每个 pod 一张卡，所以 `--nproc_per_node` 固定写 1；这些分布式启动参数不写进 `0_yaml_to_setting.py`。

6. 导出合并 ckpt：

```bash
source ./0_config.sh qwen3vl_8b_extract_attrs_h100_4gpu && \
cd "$LF_DIR" && \
"$LF_CLI" export "$EXPORT_YAML_OUT"
```

## 产物落点

- venv：`WORK_DIR/.sft_venv/`
- 训练数据：当前 task 的 `CSV_PATH` / `TRAIN_JSON` / `IMG_DIR`
- LoRA：当前 task 的 `SAVE_DIR`
- 合并模型：当前 task 的 `MERGED_DIR`
