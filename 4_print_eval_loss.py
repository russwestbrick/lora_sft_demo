#!/usr/bin/env python3
"""Print eval_loss rows and save an eval_loss curve image.

Usage:
    python 4_print_eval_loss.py /path/to/trainer_log.jsonl
    python 4_print_eval_loss.py /path/to/trainer_log.jsonl --save-dir /tmp/my_plots
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print eval_loss rows and draw an eval_loss curve.")
    parser.add_argument("log_path", help="Path to trainer_log.jsonl")
    parser.add_argument(
        "--save-dir",
        default="",
        help="Optional output directory for the generated image. Defaults to the trainer_log.jsonl directory.",
    )
    return parser.parse_args()


def read_eval_rows(log_path: Path) -> list[tuple[int, dict]]:
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
            if "eval_loss" in row:
                eval_rows.append((line_no, row))
    return eval_rows


def print_eval_rows(eval_rows: list[tuple[int, dict]]) -> tuple[int, dict]:
    best_line_no, best_row = min(eval_rows, key=lambda item: item[1]["eval_loss"])
    for line_no, row in eval_rows:
        marker = "<<< BEST" if line_no == best_line_no else "        "
        print(
            f"{marker} line={line_no} "
            f"step={row.get('current_steps', '-')} "
            f"eval_loss={row['eval_loss']:.12f} "
            f"epoch={row.get('epoch', '-')} "
            f"percentage={row.get('percentage', '-')} "
            f"elapsed={row.get('elapsed_time', '-')} "
            f"remaining={row.get('remaining_time', '-')}"
        )

    print()
    print(
        f"best eval_loss: {best_row['eval_loss']:.12f} "
        f"(line={best_line_no}, step={best_row.get('current_steps', '-')}, epoch={best_row.get('epoch', '-')})"
    )
    return best_line_no, best_row


def build_output_path(save_dir: Path, suffix: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval_loss_curve_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    candidate = save_dir / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = save_dir / f"{stem}_{index:02d}{suffix}"
        index += 1
    return candidate


def extract_series(eval_rows: list[tuple[int, dict]]) -> tuple[list[float], list[float], int, dict]:
    best_idx = min(range(len(eval_rows)), key=lambda idx: eval_rows[idx][1]["eval_loss"])
    x_values = []
    y_values = []
    for idx, (_, row) in enumerate(eval_rows):
        step = row.get("current_steps")
        x_values.append(float(step) if isinstance(step, (int, float)) else float(idx + 1))
        y_values.append(float(row["eval_loss"]))
    return x_values, y_values, best_idx, eval_rows[best_idx][1]


def save_matplotlib_plot(
    output_path: Path,
    x_values: list[float],
    y_values: list[float],
    best_idx: int,
    title_label: str,
    generated_at: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(x_values, y_values, color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax.scatter(
        [x_values[best_idx]],
        [y_values[best_idx]],
        color="#d62728",
        s=90,
        marker="*",
        zorder=3,
        label="best eval_loss",
    )
    ax.annotate(
        f"best={y_values[best_idx]:.6f}\nstep={int(x_values[best_idx]) if x_values[best_idx].is_integer() else x_values[best_idx]}",
        xy=(x_values[best_idx], y_values[best_idx]),
        xytext=(10, -28),
        textcoords="offset points",
        fontsize=9,
        color="#d62728",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#fff5f5", "ec": "#d62728"},
    )
    ax.set_title(f"Eval Loss Curve - {title_label}")
    ax.set_xlabel("current_steps")
    ax.set_ylabel("eval_loss")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend()
    fig.text(0.99, 0.01, f"generated_at: {generated_at}", ha="right", va="bottom", fontsize=8, color="#6b7280")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_svg_plot(
    output_path: Path,
    x_values: list[float],
    y_values: list[float],
    best_idx: int,
    title_label: str,
    generated_at: str,
) -> Path:
    width = 1200
    height = 700
    left = 90
    right = 50
    top = 70
    bottom = 90
    chart_width = width - left - right
    chart_height = height - top - bottom

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 0.01
        max_y += 0.01

    y_pad = (max_y - min_y) * 0.12
    min_y -= y_pad
    max_y += y_pad

    def px(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * chart_width

    def py(y: float) -> float:
        return top + chart_height - (y - min_y) / (max_y - min_y) * chart_height

    points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(x_values, y_values))
    best_x = px(x_values[best_idx])
    best_y = py(y_values[best_idx])

    y_ticks = []
    for i in range(6):
        value = min_y + (max_y - min_y) * i / 5
        y_pos = py(value)
        y_ticks.append(
            f'<line x1="{left}" y1="{y_pos:.2f}" x2="{width-right}" y2="{y_pos:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
            f'<text x="{left-12}" y="{y_pos+4:.2f}" font-size="14" text-anchor="end" fill="#374151">{value:.4f}</text>'
        )

    x_ticks = []
    for idx, x in enumerate(x_values):
        x_pos = px(x)
        label = int(x) if float(x).is_integer() else f"{x:.2f}"
        x_ticks.append(
            f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{height-bottom}" stroke="#f3f4f6" stroke-width="1"/>'
            f'<text x="{x_pos:.2f}" y="{height-bottom+28}" font-size="13" text-anchor="middle" fill="#374151">{label}</text>'
        )
        if idx >= 14 and len(x_values) > 15:
            break

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="36" font-size="28" font-weight="700" fill="#111827">Eval Loss Curve - {title_label}</text>
  <text x="{left}" y="60" font-size="15" fill="#4b5563">Generated from trainer_log.jsonl</text>
  <text x="{width-right}" y="60" font-size="14" text-anchor="end" fill="#6b7280">generated_at: {generated_at}</text>
  <rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#fcfcfd" stroke="#d1d5db" stroke-width="1.5"/>
  {''.join(y_ticks)}
  {''.join(x_ticks)}
  <polyline fill="none" stroke="#2563eb" stroke-width="3" points="{points}"/>
  <circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="6" fill="#dc2626"/>
  <text x="{best_x+12:.2f}" y="{best_y-12:.2f}" font-size="15" fill="#b91c1c">best={y_values[best_idx]:.6f}</text>
  <text x="{best_x+12:.2f}" y="{best_y+10:.2f}" font-size="14" fill="#b91c1c">step={int(x_values[best_idx]) if x_values[best_idx].is_integer() else x_values[best_idx]}</text>
  <text x="{width/2:.2f}" y="{height-20}" font-size="16" text-anchor="middle" fill="#111827">current_steps</text>
  <text x="26" y="{height/2:.2f}" font-size="16" text-anchor="middle" fill="#111827" transform="rotate(-90 26 {height/2:.2f})">eval_loss</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def save_plot(eval_rows: list[tuple[int, dict]], save_dir: Path, title_label: str, generated_at: str) -> Path:
    x_values, y_values, best_idx, _ = extract_series(eval_rows)
    try:
        output_path = build_output_path(save_dir, ".png")
        return save_matplotlib_plot(output_path, x_values, y_values, best_idx, title_label, generated_at)
    except Exception as exc:
        print(f"[warn] matplotlib plot failed, fallback to svg: {exc}", file=sys.stderr)
        output_path = build_output_path(save_dir, ".svg")
        return save_svg_plot(output_path, x_values, y_values, best_idx, title_label, generated_at)


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_path).expanduser()
    if not log_path.is_file():
        print(f"[error] file not found: {log_path}", file=sys.stderr)
        return 1

    save_dir = Path(args.save_dir).expanduser() if args.save_dir else log_path.parent
    title_label = log_path.name
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    eval_rows = read_eval_rows(log_path)
    if not eval_rows:
        print("[info] no eval_loss rows found")
        return 0

    print_eval_rows(eval_rows)
    image_path = save_plot(eval_rows, save_dir, title_label, generated_at)
    print(f"saved eval_loss image: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
