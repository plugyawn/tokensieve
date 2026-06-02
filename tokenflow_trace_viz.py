"""Trace-only visualizations for TokenFlow VOI selection.

This avoids rerunning the 14B VLM. It loads saved selector traces plus the
reference images and redraws the evidence ladder: base semantic evidence,
coarse parent regions, and incremental fine/detail cells at each budget.
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from tokenflow_data import load_records, normalize_answer

STAGE_COLORS = {
    "K32": "#e63946",
    "K128": "#f4a261",
    "K512": "#2a9d8f",
}


def build_blocks(n_tokens: int, block_size: int) -> list[list[int]]:
    grid = int(n_tokens**0.5)
    if grid * grid != n_tokens:
        raise ValueError(f"n_tokens={n_tokens} is not a square grid")
    blocks = []
    for y in range(0, grid, block_size):
        for x in range(0, grid, block_size):
            block = []
            for yy in range(y, min(y + block_size, grid)):
                for xx in range(x, min(x + block_size, grid)):
                    block.append(yy * grid + xx)
            blocks.append(block)
    return blocks


def selected_blocks(blocks: list[list[int]], selected_tokens: list[int]) -> set[int]:
    selected = set(int(tok) for tok in selected_tokens)
    return {idx for idx, block in enumerate(blocks) if set(block).issubset(selected)}


def block_bounds(block: list[int], grid: int, width: int, height: int) -> tuple[float, float, float, float]:
    rows = [tok // grid for tok in block]
    cols = [tok % grid for tok in block]
    return (
        min(cols) / grid * width,
        min(rows) / grid * height,
        (max(cols) + 1) / grid * width,
        (max(rows) + 1) / grid * height,
    )


def parent_key(block: list[int], grid: int, coarse_block_size: int) -> tuple[int, int]:
    rows = [tok // grid for tok in block]
    cols = [tok % grid for tok in block]
    cy = sum(rows) / len(rows)
    cx = sum(cols) / len(cols)
    return int(cx // coarse_block_size) * coarse_block_size, int(cy // coarse_block_size) * coarse_block_size


def stage_blocks(row: dict[str, Any], blocks: list[list[int]]) -> dict[str, set[int]]:
    k32 = selected_blocks(blocks, row["selected_tokens"].get("K32", []))
    k128_all = selected_blocks(blocks, row["selected_tokens"].get("K128", []))
    k512_all = selected_blocks(blocks, row["selected_tokens"].get("K512", []))
    return {
        "K32": k32,
        "K128": k128_all - k32,
        "K512": k512_all - k128_all,
    }


def draw_base(ax, image):
    ax.imshow(image)
    ax.axis("off")
    width, height = image.size
    ax.add_patch(patches.Rectangle((0, 0), width, height, linewidth=1.3, edgecolor="#4361ee", facecolor="#4361ee", alpha=0.15, linestyle="--"))
    ax.text(0.5, 0.05, "initial evidence:\nsemantic/base everywhere\ndetail channels hidden", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="white", bbox={"facecolor": "#111827", "alpha": 0.78, "edgecolor": "none", "pad": 3})


def draw_parents(ax, image, blocks, stages, grid, coarse_block_size):
    ax.imshow(image)
    ax.axis("off")
    width, height = image.size
    parent_to_counts: dict[tuple[int, int], dict[str, int]] = {}
    for stage, block_ids in stages.items():
        for idx in block_ids:
            key = parent_key(blocks[idx], grid, coarse_block_size)
            parent_to_counts.setdefault(key, {"K32": 0, "K128": 0, "K512": 0})[stage] += 1
    for key, counts in parent_to_counts.items():
        first_stage = next(stage for stage in ["K32", "K128", "K512"] if counts[stage])
        color = STAGE_COLORS[first_stage]
        px, py = key
        x0 = px / grid * width
        y0 = py / grid * height
        x1 = min(px + coarse_block_size, grid) / grid * width
        y1 = min(py + coarse_block_size, grid) / grid * height
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.2, edgecolor=color, facecolor=color, alpha=0.22, linestyle="--"))
        label = f"{sum(counts.values())} cells"
        ax.text(x0 + 3, y0 + 3, label, ha="left", va="top", fontsize=8, color="white", bbox={"facecolor": "#111827", "alpha": 0.76, "edgecolor": "none", "pad": 2})
    ax.text(0.5, 0.05, "coarse parent regions\ncontaining selected detail cells", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="white", bbox={"facecolor": "#111827", "alpha": 0.78, "edgecolor": "none", "pad": 3})


def draw_stage(ax, image, blocks, block_ids, stage, grid):
    ax.imshow(image)
    ax.axis("off")
    width, height = image.size
    color = STAGE_COLORS[stage]
    for idx in block_ids:
        x0, y0, x1, y1 = block_bounds(blocks[idx], grid, width, height)
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.37))
    ax.text(0.5, 0.05, f"new detail cells at {stage}\nnot a finer scale label", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="white", bbox={"facecolor": "#111827", "alpha": 0.78, "edgecolor": "none", "pad": 3})


def save_row(out: Path, row: dict[str, Any], image, coarse_block_size: int) -> Path:
    n_tokens = int(row["n_tokens"])
    grid = int(row["token_grid"])
    block_size = int(row["block_size"])
    blocks = build_blocks(n_tokens, block_size)
    stages = stage_blocks(row, blocks)
    fig, axes = plt.subplots(1, 6, figsize=(27, 5.8))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("reference image", fontsize=10)
    draw_base(axes[1], image)
    axes[1].set_title(f"K0/base\npred: {row['predictions'].get('K0', '')[:34]}", fontsize=9)
    draw_parents(axes[2], image, blocks, stages, grid, coarse_block_size)
    axes[2].set_title("where it asks broadly", fontsize=9)
    for ax, stage in zip(axes[3:], ["K32", "K128", "K512"]):
        draw_stage(ax, image, blocks, stages[stage], stage, grid)
        actual = row["actual_tokens"].get(stage, 0)
        pred = row["predictions"].get(stage, "")[:30]
        ax.set_title(f"{stage}: +{len(stages[stage])} blocks, cumulative {actual} tokens\npred: {pred}", fontsize=9)
    q = "\n".join(textwrap.wrap(row["question"], 105))
    fig.suptitle(f"heldout record {row['record_idx']} | Question: {q}\nGold: {row['gold']}", fontsize=12, y=1.04)
    fig.text(0.5, 0.02, "Read this left-to-right: base semantic evidence first, then coarse regions, then incremental new detail cells. Green means late-added cells, not finest scale.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    path = out / f"record_{int(row['record_idx']):03d}_trace_ladder.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_contact_sheet(paths: list[Path], out: Path) -> Path:
    from PIL import Image, ImageDraw
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1320, 540))
        thumbs.append((path.name, image.copy()))
    width = 1360
    row_h = 585
    canvas = Image.new("RGB", (width, row_h * len(thumbs)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for name, image in thumbs:
        draw.text((16, y + 8), name, fill=(20, 20, 20))
        canvas.paste(image, (20, y + 34))
        y += row_h
    path = out / "trace_ladder_contact_sheet.png"
    canvas.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default="sionic-ai/textvqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--skip-usable", type=int, default=256)
    parser.add_argument("--coarse-block-size", type=int, default=9)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    max_idx = max(int(row["record_idx"]) for row in rows)
    records = load_records(args.dataset, args.split, max_idx + 1, streaming=True, min_answer_len=1, skip_usable=args.skip_usable)
    paths = []
    summary = []
    for row in rows:
        idx = int(row["record_idx"])
        record = records[idx]
        image = record.image.convert("RGB")
        row["gold"] = row.get("gold") or normalize_answer(record.answer)
        path = save_row(args.out, row, image, args.coarse_block_size)
        paths.append(path)
        summary.append({"record_idx": idx, "question": row["question"], "gold": row["gold"], "figure": str(path), "actual_tokens": row["actual_tokens"], "stage_block_counts": row["stage_block_counts"]})
    contact = make_contact_sheet(paths, args.out)
    with (args.out / "summary.json").open("w") as f:
        json.dump({"trace": str(args.trace), "contact_sheet": str(contact), "examples": summary}, f, indent=2)
    print(contact)


if __name__ == "__main__":
    main()
