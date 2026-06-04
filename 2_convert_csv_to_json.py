#!/usr/bin/env python3
"""CSV -> LLaMA-Factory sharegpt JSON (multimodal), with threaded image download.

Usage:
    python 2_convert_csv_to_json.py [/path/to/input.csv]

If the CSV path is omitted, falls back to:
    $SFT_ROOT / $CSV_NAME

Outputs (fixed, relative to $SFT_ROOT):
    lora_sft_data/train.json
    lora_sft_data/images/<md5>.<ext>
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

csv.field_size_limit(10**9)

SCRIPT_DIR = Path(__file__).resolve().parent
SFT_ROOT = SCRIPT_DIR.parent

OUT_ROOT = SFT_ROOT / "lora_sft_data"
OUT_JSON = OUT_ROOT / "train.json"
IMG_DIR = OUT_ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# 匹配 user_prompt 中以 🖼️ 开头的整行图片链接（U+1F5BC + 可选 VS16）
IMG_LINE_RE = re.compile(r"^\s*\U0001F5BC\uFE0F?\s*(\S+)\s*$", re.MULTILINE)
TIMEOUT = 20
CONCURRENCY = int(os.getenv("IMG_CONCURRENCY", "16"))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def cached_path(url: str):
    """Return existing local path for url if any cached file matches."""
    name = hashlib.md5(url.encode()).hexdigest()
    for ext in IMG_EXTS:
        p = IMG_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def download_image(url: str):
    """Download a remote image into IMG_DIR; reuse existing file if present."""
    cached = cached_path(url)
    if cached is not None:
        return cached
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        name = hashlib.md5(url.encode()).hexdigest()
        out = IMG_DIR / f"{name}.jpg"
        img.save(out, format="JPEG", quality=90)
        return out
    except Exception as e:
        print(f"[warn] download failed: {url} ({e})", file=sys.stderr)
        return None


def resolve_csv_path() -> Path:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).expanduser().resolve()
    csv_name = os.environ.get("CSV_NAME")
    if not csv_name:
        print(
            "usage: 2_convert_csv_to_json.py <input_csv>\n"
            "   or: export CSV_NAME=<file_in_sft_root>",
            file=sys.stderr,
        )
        sys.exit(2)
    return (SFT_ROOT / csv_name).resolve()


def iter_rows(csv_path: Path):
    """Yield (system, user_prompt, llm_output, [urls]) for each valid row."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sys_p = (row.get("system_prompt") or "").strip()
            user_p = (row.get("user_prompt") or "").strip()
            answer = (row.get("llm_output") or "").strip()
            if not user_p or not answer:
                continue
            urls = IMG_LINE_RE.findall(user_p)
            yield sys_p, user_p, answer, urls


def main():
    csv_path = resolve_csv_path()
    if not csv_path.is_file():
        print(f"[error] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    # Pass 1: collect rows + dedup all URLs (skip ones already cached)
    rows = []
    urls_to_fetch: set[str] = set()
    cache_hits: dict[str, Path] = {}
    print(f"[scan] {csv_path}")
    for sys_p, user_p, answer, urls in tqdm(iter_rows(csv_path), desc="scan", unit="row"):
        rows.append((sys_p, user_p, answer, urls))
        for u in urls:
            if u in cache_hits or u in urls_to_fetch:
                continue
            cached = cached_path(u)
            if cached is not None:
                cache_hits[u] = cached
            else:
                urls_to_fetch.add(u)

    print(
        f"[scan] rows={len(rows)} unique_urls={len(urls_to_fetch) + len(cache_hits)} "
        f"cached={len(cache_hits)} to_fetch={len(urls_to_fetch)}"
    )

    # Pass 2: threaded download
    url_to_path: dict[str, Path | None] = dict(cache_hits)
    if urls_to_fetch:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            fut_to_url = {ex.submit(download_image, u): u for u in urls_to_fetch}
            for fut in tqdm(
                as_completed(fut_to_url),
                total=len(fut_to_url),
                desc="download",
                unit="img",
            ):
                u = fut_to_url[fut]
                try:
                    url_to_path[u] = fut.result()
                except Exception as e:
                    print(f"[warn] worker error for {u}: {e}", file=sys.stderr)
                    url_to_path[u] = None

    # Pass 3: build samples using url_to_path
    samples = []
    for sys_p, user_p, answer, urls in rows:
        images: list[str] = []

        def _sub(m):
            url = m.group(1)
            local = url_to_path.get(url)
            if local is None:
                return ""
            images.append(str(local))
            return "<image>"

        cleaned = IMG_LINE_RE.sub(_sub, user_p)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        if not images:
            continue
        sample = {
            "messages": [
                {"role": "user", "content": cleaned},
                {"role": "assistant", "content": answer},
            ],
            "images": images,
        }
        if sys_p:
            sample["system"] = sys_p
        samples.append(sample)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(samples)} samples -> {OUT_JSON}")
    print(f"images dir: {IMG_DIR}")


if __name__ == "__main__":
    main()
