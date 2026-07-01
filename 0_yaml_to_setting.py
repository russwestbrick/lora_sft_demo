"""Task settings for LoRA SFT runs.

Edit TASK_SETTINGS to switch model, data, yaml templates, output names, and
distributed launch defaults without rewriting the step scripts.

### AIS model prepare
pip3 install -U shopee-ais -i https://pypi.garenanow.com/simple/ && curl https://rclone.org/install.sh | sudo bash
ais login --email youwei.wang@shopee.com --token isqbmbgjgpafmrzlkfEplpoBscyDyxua --host https://ais.mlp.shopee.io
ais model download --model_id=39978 --version_id=67719 --output_path="<task_name.model_path>" --project=112
"""

DEFAULT_TASK_NAME = "qwen35vl_9b_extract_attrs_tags_h100_8gpu"

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

    "qwen3vl_8b_extract_attrs_h100_8gpu": {
        "_ready": 1,
        "description": "Qwen3-VL-8B extract attributes LoRA SFT on 8 H100 GPUs",
        # "data_dir": "/Users/youwei.wang/Desktop/quality_score",
        "csv_name": "training_data_test_extract_attributes_cleaned.csv",
        "train_json_name": "train_qwen3vl_8b_extract_attrs_h100_8gpu_cleaned.json",
        "image_dir_name": "images_qwen3vl_8b_extract_attrs_h100_8gpu_cleaned",
        "model_path": "/home/work/slamm/youwei.wang/lora_sft_demo/model_repo/Qwen3-VL-8B-Instruct",
        "dataset_name": "extract_attrs_sft_h100_8gpu_cleaned",
        "train_yaml_template": "configs/lora_sft_attributes_8h100.yaml",
        "export_yaml_template": "configs/export.yaml",
        "train_yaml_name": "qwen3vl_8b_extract_attrs_h100_8gpu_lora_sft.yaml",
        "export_yaml_name": "qwen3vl_8b_extract_attrs_h100_8gpu_lora_merge.yaml",
        "save_dir": "saves/qwen3vl_8b_extract_attrs_h100_8gpu/lora/sft",
        "merged_dir": "saves/qwen3vl_8b_extract_attrs_h100_8gpu/lora/merged",
        "img_concurrency": 16,
        # These two fields render into the LLaMA-Factory yaml. Distributed launch
        # knobs such as WORLD_SIZE/RANK/NPROC_PER_NODE stay in the platform entrypoint.
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    },

    "qwen35vl_9b_extract_attrs_tags_h100_8gpu": {
        "_ready": 1,
        "description": "Qwen3.5-VL-9B extract attributes & item_tags LoRA SFT on 8 H100 GPUs",
        "csv_name": "sft_data_14543.csv",
        "train_json_name": "train_qwen35vl_9b_extract_attrs_tags_h100_8gpu.json",
        "image_dir_name": "images_qwen35vl_9b_extract_attrs_tags_h100_8gpu",
        "model_path": "/home/work/slamm/youwei.wang/lora_sft_demo/model_repo/Qwen35-9B",
        "dataset_name": "extract_attrs_tags_sft_h100_8gpu",
        "train_yaml_template": "configs/lora_sft_attrs_tags_8h100.yaml",
        "export_yaml_template": "configs/export.yaml",
        "train_yaml_name": "qwen35vl_9b_extract_attrs_tags_h100_8gpu_lora_sft.yaml",
        "export_yaml_name": "qwen35vl_9b_extract_attrs_tags_h100_8gpu_lora_merge.yaml",
        "save_dir": "saves/qwen35vl_9b_extract_attrs_tags_h100_8gpu/lora/sft",
        "merged_dir": "saves/qwen35vl_9b_extract_attrs_tags_h100_8gpu/lora/merged",
        "img_concurrency": 32,
        # These two fields render into the LLaMA-Factory yaml. Distributed launch
        # knobs such as WORLD_SIZE/RANK/NPROC_PER_NODE stay in the platform entrypoint.
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    },
}

"""
    bash 1_python_venv.sh qwen35vl_9b_extract_attrs_tags_h100_8gpu
    downloading uv 0.11.26 x86_64-unknown-linux-gnu
    installing to /home/work/.local/bin
    uv
    uvx
    everything's installed!
    Using CPython 3.11.15
    Creating virtual environment with seed packages at: .sft_venv
    warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
            If the cache and target directories are on different filesystems, hardlinking may not be supported.
            If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
    + packaging==26.2
    + pip==26.1.2
    + setuptools==82.0.1
    + wheel==0.47.0
    Activate with: source .sft_venv/bin/activate
    Using Python 3.11.15 environment at: .sft_venv
    Resolved 26 packages in 3.93s
    Prepared 26 packages in 26.77s
    ░░░░░░░░░░░░░░░░░░░░ [0/26] Installing wheels...                                                                                                 warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
            If the cache and target directories are on different filesystems, hardlinking may not be supported.
            If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
    Installed 26 packages in 4m 34s
    + filelock==3.29.0
    + fsspec==2026.4.0
    + jinja2==3.1.6
    + markupsafe==3.0.3
    + mpmath==1.3.0
    + networkx==3.6.1
    + numpy==2.4.4
    + nvidia-cublas-cu12==12.4.5.8
    + nvidia-cuda-cupti-cu12==12.4.127
    + nvidia-cuda-nvrtc-cu12==12.4.127
    + nvidia-cuda-runtime-cu12==12.4.127
    + nvidia-cudnn-cu12==9.1.0.70
    + nvidia-cufft-cu12==11.2.1.3
    + nvidia-curand-cu12==10.3.5.147
    + nvidia-cusolver-cu12==11.6.1.9
    + nvidia-cusparse-cu12==12.3.1.170
    + nvidia-nccl-cu12==2.21.5
    + nvidia-nvjitlink-cu12==12.4.127
    + nvidia-nvtx-cu12==12.4.127
    + pillow==12.2.0
    + sympy==1.13.1
    + torch==2.5.1+cu124
    + torchaudio==2.5.1+cu124
    + torchvision==0.20.1+cu124
    + triton==3.1.0
    + typing-extensions==4.15.0
    Using Python 3.11.15 environment at: .sft_venv
    Resolved 121 packages in 1.56s
        Built llamafactory @ file:///home/work/slamm/youwei.wang/lora_sft_demo/LLaMA-Factory
        Built antlr4-python3-runtime==4.9.3
    Prepared 95 packages in 3.51s
    Uninstalled 2 packages in 1.04s
    ░░░░░░░░░░░░░░░░░░░░ [0/95] Installing wheels...                                                                                                 warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
            If the cache and target directories are on different filesystems, hardlinking may not be supported.
            If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
    Installed 95 packages in 1m 58s
    + accelerate==1.11.0
    + aiofiles==24.1.0
    + aiohappyeyeballs==2.6.2
    + aiohttp==3.14.1
    + aiosignal==1.4.0
    + annotated-doc==0.0.4
    + annotated-types==0.7.0
    + antlr4-python3-runtime==4.9.3
    + anyio==4.14.1
    + attrs==26.1.0
    + av==16.0.0
    + brotli==1.2.0
    + certifi==2026.6.17
    + charset-normalizer==3.4.7
    + click==8.4.2
    + contourpy==1.3.3
    + cycler==0.12.1
    + datasets==4.0.0
    + dill==0.3.8
    + docstring-parser==0.18.0
    + einops==0.8.2
    + fastapi==0.138.2
    + ffmpy==1.0.0
    + fire==0.7.1
    + fonttools==4.63.0
    + frozenlist==1.8.0
    - fsspec==2026.4.0
    + fsspec==2025.3.0
    + gradio==5.50.0
    + gradio-client==1.14.0
    + groovy==0.1.2
    + h11==0.16.0
    + hf-transfer==0.1.9
    + hf-xet==1.5.1
    + httpcore==1.0.9
    + httpx==0.28.1
    + huggingface-hub==1.21.0
    + idna==3.18
    + kiwisolver==1.5.0
    + llamafactory==0.9.6.dev0 (from file:///home/work/slamm/youwei.wang/lora_sft_demo/LLaMA-Factory)
    + markdown-it-py==4.2.0
    + matplotlib==3.11.0
    + mdurl==0.1.2
    + modelscope==1.38.0
    + modelscope-hub==0.1.5
    + multidict==6.7.1
    + multiprocess==0.70.16
    + omegaconf==2.3.1
    + orjson==3.11.9
    + pandas==2.3.3
    + peft==0.18.1
    - pillow==12.2.0
    + pillow==11.3.0
    + propcache==0.5.2
    + protobuf==7.35.1
    + psutil==7.2.2
    + pyarrow==24.0.0
    + pydantic==2.12.3
    + pydantic-core==2.41.4
    + pydub==0.25.1
    + pygments==2.20.0
    + pyparsing==3.3.2
    + python-dateutil==2.9.0.post0
    + python-multipart==0.0.32
    + pytz==2026.2
    + pyyaml==6.0.3
    + regex==2026.6.28
    + requests==2.34.2
    + rich==15.0.0
    + ruff==0.15.20
    + safehttpx==0.1.7
    + safetensors==0.8.0
    + scipy==1.17.1
    + semantic-version==2.10.0
    + sentencepiece==0.2.1
    + shellingham==1.5.4
    + shtab==1.8.0
    + six==1.17.0
    + sse-starlette==3.4.5
    + starlette==0.52.1
    + termcolor==3.3.0
    + tiktoken==0.13.0
    + tokenizers==0.22.2
    + tomlkit==0.13.3
    + torchdata==0.11.0
    + tqdm==4.68.3
    + transformers==5.6.0
    + trl==0.24.0
    + typer==0.25.1
    + typing-inspection==0.4.2
    + tyro==0.8.14
    + tzdata==2026.2
    + urllib3==2.7.0
    + uvicorn==0.49.0
    + websockets==15.0.1
    + xxhash==3.8.0
    + yarl==1.24.2
    warning: The package `llamafactory @ file:///home/work/slamm/youwei.wang/lora_sft_demo/LLaMA-Factory` does not have an extra named `metrics`
    warning: The package `llamafactory @ file:///home/work/slamm/youwei.wang/lora_sft_demo/LLaMA-Factory` does not have an extra named `torch`
    Using Python 3.11.15 environment at: .sft_venv
    Checked 4 packages in 184ms
    venv ready: /home/work/slamm/youwei.wang/lora_sft_demo/.sft_venv
    /home/work/slamm/youwei.wang/lora_sft_demo/.sft_venv/lib/python3.11/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
    warnings.warn(
    torch 2.5.1+cu124 cuda True
    transformers 5.6.0
    peft 0.18.1 trl 0.24.0
    llamafactory 0.9.6.dev0
"""
