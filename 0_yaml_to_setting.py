"""Task settings for LoRA SFT runs.

Edit TASK_SETTINGS to switch model, data, yaml templates, output names, and
distributed launch defaults without rewriting the step scripts.
"""

DEFAULT_TASK_NAME = "qwen3vl_8b_extract_attrs_h100_4gpu"

TASK_SETTINGS = {
    "qwen3vl_8b_extract_attrs_h100_4gpu": {
        "_ready": 1,
        "description": "Qwen3-VL-8B extract attributes LoRA SFT on 4 H100 GPUs",
        "csv_name": "training_data_extract_attributes_and_item_tags.csv",
        "train_json_name": "train_qwen3vl_8b_extract_attrs_h100_4gpu.json",
        "image_dir_name": "images_qwen3vl_8b_extract_attrs_h100_4gpu",
        "model_path": "/home/work/model_repo/Qwen3-VL-8B-Instruct",
        "dataset_name": "extract_attrs_sft_h100_4gpu",
        "train_yaml_template": "configs/lora_sft.yaml",
        "export_yaml_template": "configs/export.yaml",
        "train_yaml_name": "qwen3vl_8b_extract_attrs_h100_4gpu_lora_sft.yaml",
        "export_yaml_name": "qwen3vl_8b_extract_attrs_h100_4gpu_lora_merge.yaml",
        "save_dir": "saves/qwen3vl_8b_extract_attrs_h100_4gpu/lora/sft",
        "merged_dir": "saves/qwen3vl_8b_extract_attrs_h100_4gpu/lora/merged",
        "img_concurrency": 16,
        "env": {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "NPROC_PER_NODE": 4,
            "MASTER_PORT": 29500,
        },
        # Keep the old effective update batch: 1 sample/GPU * 1 accum * 4 GPUs = 4.
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    },
}
