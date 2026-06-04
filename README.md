# 1. 下载 llama-factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git \
  /home/work/Category_filesystem_V3/youwei.wang/sft/LLaMA-Factory

# 5.0 建 venv
bash /home/work/Category_filesystem_V3/youwei.wang/sft/lora_sft/python_venv.sh

# 3 数据转换
/home/work/Category_filesystem_V3/youwei.wang/sft/.sft_venv/bin/python \
  /home/work/Category_filesystem_V3/youwei.wang/sft/lora_sft/convert_csv_to_json.py \
  /home/work/Category_filesystem_V3/youwei.wang/sft/training_data_extract_attributes_and_item_tags.csv

# 4 注册数据集 + 拷贝 yaml
bash /home/work/Category_filesystem_V3/youwei.wang/sft/lora_sft/install_to_llamafactory.sh

# 5 LoRA SFT
cd /home/work/Category_filesystem_V3/youwei.wang/sft/LLaMA-Factory && \
CUDA_VISIBLE_DEVICES=0 \
/home/work/Category_filesystem_V3/youwei.wang/sft/.sft_venv/bin/llamafactory-cli train \
  examples/train_lora/qwen3vl_8b_lora_sft.yaml

# 6 导出合并 ckpt
cd /home/work/Category_filesystem_V3/youwei.wang/sft/LLaMA-Factory && \
/home/work/Category_filesystem_V3/youwei.wang/sft/.sft_venv/bin/llamafactory-cli export \
  examples/merge_lora/qwen3vl_8b_lora_merge.yaml