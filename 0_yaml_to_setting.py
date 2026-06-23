"""Task settings for LoRA SFT runs.

Edit TASK_SETTINGS to switch model, data, yaml templates, output names, and
distributed launch defaults without rewriting the step scripts.

### AIS model prepare
pip3 install -U shopee-ais -i https://pypi.garenanow.com/simple/ && curl https://rclone.org/install.sh | sudo bash
ais login --email youwei.wang@shopee.com --token isqbmbgjgpafmrzlkfEplpoBscyDyxua --host https://ais.mlp.shopee.io
ais model download --model_id=39978 --version_id=67719 --output_path="<task_name.model_path>" --project=112
"""

DEFAULT_TASK_NAME = "qwen3vl_8b_extract_attrs_h100_4gpu"

TASK_SETTINGS = {
    "qwen3vl_8b_extract_attrs_h100_4gpu": {
        "_ready": 1,
        "description": "Qwen3-VL-8B extract attributes LoRA SFT on 4 H100 GPUs",
        "csv_name": "training_data_test_extract_attributes.csv",
        "train_json_name": "train_qwen3vl_8b_extract_attrs_h100_4gpu.json",
        "image_dir_name": "images_qwen3vl_8b_extract_attrs_h100_4gpu",
        "model_path": "/home/work/slamm/youwei.wang/lora_sft_demo/model_repo/Qwen3-VL-8B-Instruct",
        "dataset_name": "extract_attrs_sft_h100_4gpu",
        "train_yaml_template": "configs/lora_sft.yaml",
        "export_yaml_template": "configs/export.yaml",
        "train_yaml_name": "qwen3vl_8b_extract_attrs_h100_4gpu_lora_sft.yaml",
        "export_yaml_name": "qwen3vl_8b_extract_attrs_h100_4gpu_lora_merge.yaml",
        "save_dir": "saves/qwen3vl_8b_extract_attrs_h100_4gpu/lora/sft",
        "merged_dir": "saves/qwen3vl_8b_extract_attrs_h100_4gpu/lora/merged",
        "img_concurrency": 16,
        # These two fields render into the LLaMA-Factory yaml. Distributed launch
        # knobs such as WORLD_SIZE/RANK/NPROC_PER_NODE stay in the platform entrypoint.
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    },
}
