#!/usr/bin/env python3
"""Check & clean train.json against LLaMA-Factory's proportional truncation.

- 读 cutoff_len、image_max_pixels、model_name_or_path 都从 configs/lora_sft.yaml
- 真实模拟 src/llamafactory/data/processor/processor_utils.py 里的 infer_seqlen
- 等比截断后看 <image> 占位是否被保留
- 删掉会被砍掉 image 占位的样本，剩余 shuffle 后原地覆盖 train.json
- 旧文件备份为 train.json.bak
- 执行：.sft_venv/bin/python /home/work/slamm/youwei.wang/lora_sft_demo/_check_train_json.py
"""
import json, math, os, random, re, shutil, subprocess, sys
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent
CFG = WORK_DIR / "0_config.sh"

# ---- 拉 0_config.sh 的环境 -------------------------------------------------
def load_env():
    out = subprocess.check_output(
        ["bash", "-c", f'set -a; source "{CFG}"; env'], text=True
    )
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

load_env()
TRAIN_JSON = Path(os.environ["TRAIN_JSON"])
MODEL_PATH = os.environ["MODEL_PATH"]

# ---- 从 yaml 读 cutoff_len / image_max_pixels ------------------------------
YAML = WORK_DIR / "configs" / "lora_sft.yaml"
def yaml_get(key, default):
    for line in YAML.read_text().splitlines():
        m = re.match(rf"\s*{key}\s*:\s*(\S+)", line)
        if m:
            return m.group(1)
    return default

CUTOFF_LEN = int(yaml_get("cutoff_len", 12000))
IMAGE_MAX_PIXELS = int(yaml_get("image_max_pixels", 262144))

# Qwen3-VL: patch_size=14, merge_size=2 -> 一个 vision token 覆盖 28x28 像素
# 所以单张图展开的 image_pad token 数 ≈ image_max_pixels / (28*28)
VISION_TOKENS_PER_IMAGE = max(1, IMAGE_MAX_PIXELS // (28 * 28))

print(f"TRAIN_JSON       : {TRAIN_JSON}")
print(f"MODEL_PATH       : {MODEL_PATH}")
print(f"CUTOFF_LEN       : {CUTOFF_LEN}")
print(f"IMAGE_MAX_PIXELS : {IMAGE_MAX_PIXELS}")
print(f"VISION_TOKENS/img: {VISION_TOKENS_PER_IMAGE}")
print()

# ---- tokenizer -------------------------------------------------------------
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 视觉占位串：<|vision_start|> + N * <|image_pad|> + <|vision_end|>
VSTART = "<|vision_start|>"
VPAD   = "<|image_pad|>"
VEND   = "<|vision_end|>"
# 确认这三个 token 在词表里都是单 token
for t in (VSTART, VPAD, VEND):
    assert tk.convert_tokens_to_ids(t) != tk.unk_token_id, f"{t} not in vocab"

def expand_image_placeholder(text: str) -> str:
    """把 user 文本里每个 <image> 展开成 qwen3-vl 的真实视觉 token 串。"""
    repl = VSTART + (VPAD * VISION_TOKENS_PER_IMAGE) + VEND
    return text.replace("<image>", repl)

# ---- 完全复刻 LLaMA-Factory 的 infer_seqlen --------------------------------
def infer_seqlen(source_len, target_len, cutoff_len):
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    max_source_len = cutoff_len - new_target_len
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len

# ---- 主流程 ----------------------------------------------------------------
data = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
print(f"loaded {len(data)} samples")

VPAD_ID = tk.convert_tokens_to_ids(VPAD)

stats = {
    "total": len(data),
    "over_cutoff": 0,
    "image_pad_dropped": 0,   # 等比截断后 source 里 image_pad 数量 < 期望
    "kept": 0,
    "dropped": 0,
}
src_lens, tgt_lens, total_lens = [], [], []
kept = []


### if you skip the /lora_sft_demo/2_convert_csv_to_json.py, need to fix the image local paths here;
# for s in data: 
#     s["images"] = [
#         img.replace(
#             "/home/work/Category_filesystem_V3/youwei.wang/sft", 
#             "/home/work/slamm/youwei.wang/lora_sft_demo"
#         )
#         for img in s.get("images", [])
#     ]

for s in data:
    sys_p = s.get("system", "")
    user  = s["messages"][0]["content"]
    asst  = s["messages"][1]["content"]
    n_img = len(s.get("images") or [])

    user_expanded = expand_image_placeholder(user)
    src_text = (sys_p + "\n" + user_expanded) if sys_p else user_expanded

    src_ids = tk.encode(src_text, add_special_tokens=False)
    tgt_ids = tk.encode(asst,     add_special_tokens=False)
    src_lens.append(len(src_ids))
    tgt_lens.append(len(tgt_ids))
    total_lens.append(len(src_ids) + len(tgt_ids))

    ns, nt = infer_seqlen(len(src_ids), len(tgt_ids), CUTOFF_LEN)
    if ns < len(src_ids) or nt < len(tgt_ids):
        stats["over_cutoff"] += 1

    kept_src = src_ids[:ns]
    kept_pads = kept_src.count(VPAD_ID)
    expect_pads = n_img * VISION_TOKENS_PER_IMAGE

    if n_img > 0 and kept_pads < expect_pads:
        # 任何一张图的 image_pad 段被砍掉一截，都会让 features/tokens 数不上
        stats["image_pad_dropped"] += 1
        stats["dropped"] += 1
        continue

    kept.append(s)
    stats["kept"] += 1

def pct(xs, p):
    xs = sorted(xs); i = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]

print()
print("== length stats (with vision expansion) ==")
for name, xs in [("source", src_lens), ("target", tgt_lens), ("total", total_lens)]:
    print(f"  {name:6s} p50={pct(xs,50):>6d}  p90={pct(xs,90):>6d}  "
          f"p99={pct(xs,99):>6d}  max={max(xs):>6d}")

print()
print("== truncation result ==")
for k, v in stats.items():
    print(f"  {k:20s}: {v}")

# ---- 备份 + shuffle + 覆盖写回 ---------------------------------------------
backup = TRAIN_JSON.with_suffix(".json.bak")
shutil.copy2(TRAIN_JSON, backup)
random.seed(42)
random.shuffle(kept)
TRAIN_JSON.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"backup  -> {backup}")
print(f"wrote   -> {TRAIN_JSON}  ({len(kept)} samples, shuffled with seed=42)")