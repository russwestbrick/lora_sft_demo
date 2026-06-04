#!/usr/bin/env python3
"""CSV -> LLaMA-Factory sharegpt JSON (multimodal).

Usage: python convert_csv_to_json.py /path/to/input.csv
Output (fixed): /home/work/Category_filesystem_V3/youwei.wang/sft/lora_sft_data/train.json
Images dir   : /home/work/Category_filesystem_V3/youwei.wang/sft/lora_sft_data/images/
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

csv.field_size_limit(10**9)

OUT_ROOT = Path("/home/work/Category_filesystem_V3/youwei.wang/sft/lora_sft_data")
OUT_JSON = OUT_ROOT / "train.json"
IMG_DIR = OUT_ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# 匹配 user_prompt 中以 🖼️ 开头的整行图片链接
IMG_LINE_RE = re.compile(r"^\s*\U0001F5BC\uFE0F?\s*(\S+)\s*$", re.MULTILINE)
TIMEOUT = 20


def download_image(url: str):
    """Download a remote image into IMG_DIR; reuse existing file if present."""
    name = hashlib.md5(url.encode()).hexdigest()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = IMG_DIR / f"{name}{ext}"
        if p.exists():
            return p
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        out = IMG_DIR / f"{name}.jpg"
        img.save(out, format="JPEG", quality=90)
        return out
    except Exception as e:
        print(f"[warn] download failed: {url} ({e})", file=sys.stderr)
        return None


def split_user_prompt(text):
    """Replace each 🖼️ URL line with <image> token; return (cleaned_text, [paths])."""
    images = []

    def _sub(m):
        url = m.group(1)
        local = download_image(url)
        if local is None:
            return ""
        images.append(str(local))
        return "<image>"

    cleaned = IMG_LINE_RE.sub(_sub, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, images


def main():
    if len(sys.argv) != 2:
        print("usage: convert_csv_to_json.py <input_csv>", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    samples = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="rows"):
            sys_p = (row.get("system_prompt") or "").strip()
            user_p = (row.get("user_prompt") or "").strip()
            answer = (row.get("llm_output") or "").strip()
            if not user_p or not answer:
                continue
            user_text, images = split_user_prompt(user_p)
            if not images:
                continue
            sample = {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": answer},
                ],
                "images": images,
            }
            if sys_p:
                sample["system"] = sys_p
            samples.append(sample)

    OUT_JSON.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(samples)} samples -> {OUT_JSON}")
    print(f"images dir: {IMG_DIR}")


if __name__ == "__main__":
    main()
