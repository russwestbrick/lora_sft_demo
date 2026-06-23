#!/usr/bin/env python3
"""CSV -> LLaMA-Factory sharegpt JSON (multimodal), threaded image download.

Configuration is sourced from sibling 0_config.sh. Pass an optional task name
or pre-source 0_config.sh to select a task.

Inputs  (from env, see 0_config.sh):
    CSV_PATH         absolute path to source CSV
    TRAIN_JSON       output JSON path
    IMG_DIR          directory to cache/download images into
    IMG_CONCURRENCY  thread pool size (default 16)
"""
import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

csv.field_size_limit(10**9)

IMG_LINE_RE = re.compile(r"^\s*\U0001F5BC\uFE0F?\s*(\S+)\s*$", re.MULTILINE)
IMAGE_TOKEN_RE = re.compile(r"<image>")
TIMEOUT = 20
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _load_config():
    """Source sibling 0_config.sh into os.environ if not already loaded."""
    task_name = sys.argv[1].strip() if len(sys.argv) > 1 else os.environ.get("TASK_NAME", "").strip()
    if not task_name and "WORK_DIR" in os.environ and "CSV_PATH" in os.environ:
        return
    cfg = Path(__file__).resolve().parent / "0_config.sh"
    if not cfg.is_file():
        print(f"[error] missing config file: {cfg}", file=sys.stderr)
        sys.exit(2)
    # set -a 让 source 出来的赋值自动 export；之后 env 列出所有变量
    source_cmd = f"set -a; source {shlex.quote(str(cfg))}"
    if task_name:
        source_cmd += f" {shlex.quote(task_name)}"
    source_cmd += "; env"
    out = subprocess.run(
        ["bash", "-c", source_cmd],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k] = v


_load_config()

CSV_PATH = Path(os.environ["CSV_PATH"])
TRAIN_JSON = Path(os.environ["TRAIN_JSON"])
IMG_DIR = Path(os.environ["IMG_DIR"])
CONCURRENCY = int(os.environ.get("IMG_CONCURRENCY", "16"))
IMG_DIR.mkdir(parents=True, exist_ok=True)


def cached_path(url: str):
    name = hashlib.md5(url.encode()).hexdigest()
    for ext in IMG_EXTS:
        p = IMG_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def download_image(url: str):
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


def iter_rows(csv_path: Path):
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
    if not CSV_PATH.is_file():
        print(f"[error] CSV not found: {CSV_PATH}", file=sys.stderr)
        print(f"        put your CSV at: {CSV_PATH.parent}/<CSV_NAME>", file=sys.stderr)
        sys.exit(2)

    # Pass 1: collect rows + dedup URLs (skip ones already cached)
    MAX_CASES = int(os.environ.get("MAX_CASES", "1000"))

    rows = []
    urls_to_fetch: set[str] = set()
    cache_hits: dict[str, Path] = {}
    print(f"[scan] {CSV_PATH} max_cases={MAX_CASES}")

    for sys_p, user_p, answer, urls in tqdm(iter_rows(CSV_PATH), desc="scan", unit="row"):
        if len(rows) >= MAX_CASES:
            break

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

    # Pass 3: build samples
    samples = []
    for sys_p, user_p, answer, urls in rows:
        images: list[str] = []
        user_p = IMAGE_TOKEN_RE.sub("", user_p)

        def _sub(m):
            url = m.group(1)
            local = url_to_path.get(url)
            if local is None or images:
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

    TRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_JSON.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(samples)} samples -> {TRAIN_JSON}")
    print(f"images dir: {IMG_DIR}")


if __name__ == "__main__":
    main()
