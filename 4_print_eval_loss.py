#!/usr/bin/env python3
"""Print eval_loss rows from a trainer log JSONL file.

Usage:
    python 4_print_eval_loss.py /path/to/trainer_log.jsonl
"""

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python 4_print_eval_loss.py /path/to/trainer_log.jsonl", file=sys.stderr)
        return 1

    log_path = Path(sys.argv[1]).expanduser()
    if not log_path.is_file():
        print(f"[error] file not found: {log_path}", file=sys.stderr)
        return 1

    eval_rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(f"[warn] skip invalid json at line {line_no}: {exc}", file=sys.stderr)
                continue

            if "eval_loss" not in row:
                continue

            eval_rows.append((line_no, row))

    if not eval_rows:
        print("[info] no eval_loss rows found")
        return 0

    best_line_no, best_row = min(eval_rows, key=lambda item: item[1]["eval_loss"])

    for line_no, row in eval_rows:
        marker = "<<< BEST" if line_no == best_line_no else "        "
        current_steps = row.get("current_steps", "-")
        epoch = row.get("epoch", "-")
        percentage = row.get("percentage", "-")
        elapsed_time = row.get("elapsed_time", "-")
        remaining_time = row.get("remaining_time", "-")
        print(
            f"{marker} line={line_no} "
            f"step={current_steps} "
            f"eval_loss={row['eval_loss']:.12f} "
            f"epoch={epoch} "
            f"percentage={percentage} "
            f"elapsed={elapsed_time} "
            f"remaining={remaining_time}"
        )

    print()
    print(
        f"best eval_loss: {best_row['eval_loss']:.12f} "
        f"(line={best_line_no}, step={best_row.get('current_steps', '-')}, epoch={best_row.get('epoch', '-')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
