"""Qualitative overlays for the TokenFlow VOI selector.

This script loads a trained question-conditioned selector, runs it on a small
heldout slice, and saves diagrams showing which visual blocks are revealed as
the budget increases. It is intentionally evaluation-only: no gold-answer
delta labels are used to choose blocks.
"""

import argparse
import json
import math
import random
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from tokenflow_data import load_records, normalize_answer
from tokenflow_vqa_eval import (
    TOKENFLOW_MODEL,
    TOKENFLOW_REPO,
    TOKENFLOW_TOKENIZER_FILE,
    TOKENFLOW_TOKENIZER_REPO,
    build_blocks,
    build_prompt,
    download_tokenizer_checkpoint,
    ensure_tokenflow_source,
    generate_answer,
    install_visual_mask,
    load_tokenflow_model,
    process_image,
    raw_visual_features,
    selected_from_blocks,
    square_grid,
)
from tokenflow_voi_policy import (
    QuestionBlockPolicy,
    block_feature_tensor,
    block_pos_features,
    projected_visual_features,
    question_feature,
    selector_blocks,
)


STAGE_COLORS = {
    "K32": "#e63946",
    "K128": "#f4a261",
    "K512": "#2a9d8f",
}


def parse_indices(value: str) -> list[int] | None:
    value = value.strip()
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_policy(path: Path, device: torch.device) -> QuestionBlockPolicy:
    ckpt = torch.load(path, map_location="cpu")
    policy = QuestionBlockPolicy(
        visual_dim=int(ckpt["visual_dim"]),
        question_dim=int(ckpt["question_dim"]),
        pos_dim=int(ckpt["pos_dim"]),
        hidden=int(ckpt["hidden"]),
        dropout=float(ckpt["dropout"]),
    )
    policy.load_state_dict(ckpt["model_state"])
    policy.sequential_inference = True
    policy.hierarchical_inference = False
    policy.coarse_block_size = 9
    policy.eval()
    return policy.to(device)


def block_ids_until_budget(
    ordered: list[list[int]],
    blocks: list[list[int]],
    budget: int,
    n_tokens: int,
) -> set[int]:
    selected = selected_from_blocks(ordered, budget, n_tokens)
    if selected is None:
        return set(range(len(blocks)))
    selected_set = set(int(x) for x in selected.tolist())
    out: set[int] = set()
    for idx, block in enumerate(blocks):
        if set(block).issubset(selected_set):
            out.add(idx)
    return out


def stage_by_block(
    ordered: list[list[int]],
    blocks: list[list[int]],
    n_tokens: int,
) -> dict[int, str]:
    k32 = block_ids_until_budget(ordered, blocks, 32, n_tokens)
    k128 = block_ids_until_budget(ordered, blocks, 128, n_tokens)
    k512 = block_ids_until_budget(ordered, blocks, 512, n_tokens)
    stage: dict[int, str] = {}
    for idx in k32:
        stage[idx] = "K32"
    for idx in k128 - k32:
        stage[idx] = "K128"
    for idx in k512 - k128:
        stage[idx] = "K512"
    return stage


def block_order_ranks(ordered: list[list[int]], blocks: list[list[int]]) -> dict[int, int]:
    block_lookup = {tuple(sorted(block)): idx for idx, block in enumerate(blocks)}
    ranks: dict[int, int] = {}
    for rank, block in enumerate(ordered):
        idx = block_lookup.get(tuple(sorted(int(tok) for tok in block)))
        if idx is not None and idx not in ranks:
            ranks[idx] = rank
    return ranks


def block_bounds(block: list[int], token_grid: int, width: int, height: int) -> tuple[float, float, float, float]:
    rows = [tok // token_grid for tok in block]
    cols = [tok % token_grid for tok in block]
    x0 = min(cols) / token_grid * width
    y0 = min(rows) / token_grid * height
    x1 = (max(cols) + 1) / token_grid * width
    y1 = (max(rows) + 1) / token_grid * height
    return x0, y0, x1, y1


def parent_key_for_block(block: list[int], token_grid: int, coarse_block_size: int) -> tuple[int, int]:
    rows = [tok // token_grid for tok in block]
    cols = [tok % token_grid for tok in block]
    cy = sum(rows) / max(1, len(rows))
    cx = sum(cols) / max(1, len(cols))
    py = int(cy // coarse_block_size) * coarse_block_size
    px = int(cx // coarse_block_size) * coarse_block_size
    return px, py


def stage_is_active(block_stage: str | None, upto: str) -> bool:
    if block_stage is None:
        return False
    upto_order = {"K32": 0, "K128": 1, "K512": 2}
    return upto_order[block_stage] <= upto_order[upto]


def draw_overlay(
    ax,
    image,
    blocks: list[list[int]],
    n_tokens: int,
    stage: dict[int, str],
    ranks: dict[int, int],
    upto: str | None,
    *,
    mask_mode: str = "token",
    learned_inference: str = "sequential",
    coarse_block_size: int = 9,
    show_ranks: bool = False,
    incremental_only: bool = False,
) -> None:
    ax.imshow(image)
    ax.axis("off")
    if upto is None:
        return
    token_grid = square_grid(n_tokens)
    if token_grid is None:
        return
    width, height = image.size
    if upto == "K0" and mask_mode == "semantic_dense_pixel_sparse":
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                width,
                height,
                linewidth=1.2,
                edgecolor="#4361ee",
                facecolor="#4361ee",
                alpha=0.16,
                linestyle="--",
            )
        )
        ax.text(
            0.5,
            0.04,
            "semantic channels visible everywhere\npixel/detail hidden",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="white",
            bbox={"facecolor": "#1f2937", "alpha": 0.72, "edgecolor": "none", "pad": 3},
        )
        return
    if upto == "K0":
        ax.add_patch(
            patches.Rectangle((0, 0), width, height, linewidth=1.2, edgecolor="#6b7280", facecolor="#111827", alpha=0.18)
        )
        ax.text(
            0.5,
            0.04,
            "no visual detail tokens",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="white",
            bbox={"facecolor": "#1f2937", "alpha": 0.72, "edgecolor": "none", "pad": 3},
        )
        return
    active_block_indices = []
    for idx, block in enumerate(blocks):
        block_stage = stage.get(idx)
        if incremental_only:
            if block_stage != upto:
                continue
        elif not stage_is_active(block_stage, upto):
            continue
        active_block_indices.append(idx)
        x0, y0, x1, y1 = block_bounds(block, token_grid, width, height)
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.1,
            edgecolor=STAGE_COLORS[block_stage],
            facecolor=STAGE_COLORS[block_stage],
            alpha=0.36,
        )
        ax.add_patch(rect)
        rank = ranks.get(idx)
        if show_ranks and rank is not None and rank < 16:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                str(rank + 1),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
                bbox={"facecolor": "#111827", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
            )
    if learned_inference == "hierarchical" and coarse_block_size > 0:
        parent_boxes = {}
        for idx in active_block_indices:
            block = blocks[idx]
            key = parent_key_for_block(block, token_grid, coarse_block_size)
            parent_boxes[key] = min(parent_boxes.get(key, "K512"), stage.get(idx, "K512"), key=lambda name: {"K32": 0, "K128": 1, "K512": 2}[name])
        for (px, py), parent_stage in parent_boxes.items():
            x0 = px / token_grid * width
            y0 = py / token_grid * height
            x1 = min(px + coarse_block_size, token_grid) / token_grid * width
            y1 = min(py + coarse_block_size, token_grid) / token_grid * height
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    linewidth=2.2,
                    edgecolor=STAGE_COLORS[parent_stage],
                    facecolor="none",
                    alpha=0.95,
                    linestyle="--",
                )
            )


def draw_parent_ladder(
    ax,
    image,
    blocks: list[list[int]],
    n_tokens: int,
    stage: dict[int, str],
    ranks: dict[int, int],
    *,
    coarse_block_size: int,
    upto: str = "K512",
) -> None:
    ax.imshow(image)
    ax.axis("off")
    token_grid = square_grid(n_tokens)
    if token_grid is None or coarse_block_size <= 0:
        return
    width, height = image.size
    parent_groups: dict[tuple[int, int], dict[str, Any]] = {}
    for idx, block in enumerate(blocks):
        block_stage = stage.get(idx)
        if not stage_is_active(block_stage, upto):
            continue
        key = parent_key_for_block(block, token_grid, coarse_block_size)
        rank = ranks.get(idx, 10**9)
        group = parent_groups.setdefault(
            key,
            {"first_rank": rank, "first_stage": block_stage, "count": 0, "children": []},
        )
        group["first_rank"] = min(group["first_rank"], rank)
        group["first_stage"] = min(
            group["first_stage"],
            block_stage,
            key=lambda name: {"K32": 0, "K128": 1, "K512": 2}[name],
        )
        group["count"] += 1
        group["children"].append(idx)
    sorted_parents = sorted(parent_groups.items(), key=lambda item: item[1]["first_rank"])
    for parent_order, ((px, py), group) in enumerate(sorted_parents, start=1):
        parent_stage = group["first_stage"]
        color = STAGE_COLORS[parent_stage]
        x0 = px / token_grid * width
        y0 = py / token_grid * height
        x1 = min(px + coarse_block_size, token_grid) / token_grid * width
        y1 = min(py + coarse_block_size, token_grid) / token_grid * height
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=2.6,
                edgecolor=color,
                facecolor=color,
                alpha=0.22,
                linestyle="--",
            )
        )
        ax.text(
            x0 + 0.05 * (x1 - x0),
            y0 + 0.12 * (y1 - y0),
            f"P{parent_order}\n{group['count']} detail cells",
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            weight="bold",
            bbox={"facecolor": "#111827", "alpha": 0.78, "edgecolor": "none", "pad": 2},
        )
    ax.text(
        0.5,
        0.04,
        "coarse question-conditioned regions\nopened before fine detail cells",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="white",
        bbox={"facecolor": "#1f2937", "alpha": 0.72, "edgecolor": "none", "pad": 3},
    )


def summarize_order(stage: dict[int, str]) -> dict[str, int]:
    return {name: sum(1 for value in stage.values() if value == name) for name in ["K32", "K128", "K512"]}


def save_example_figure(
    out: Path,
    record_idx: int,
    image,
    question: str,
    gold: str,
    predictions: dict[str, str],
    blocks: list[list[int]],
    n_tokens: int,
    stage: dict[int, str],
    ranks: dict[int, int],
    actual_tokens: dict[str, int],
    *,
    mask_mode: str,
    learned_inference: str,
    coarse_block_size: int,
) -> Path:
    fig, axes = plt.subplots(1, 6, figsize=(27, 5.8))
    draw_overlay(axes[0], image, blocks, n_tokens, stage, ranks, None)
    axes[0].set_title("reference image", fontsize=11)
    draw_overlay(
        axes[1],
        image,
        blocks,
        n_tokens,
        stage,
        ranks,
        "K0",
        mask_mode=mask_mode,
        learned_inference=learned_inference,
        coarse_block_size=coarse_block_size,
    )
    axes[1].set_title(f"K0 semantic/base\npred: {predictions.get('K0', '')[:42]}", fontsize=10)
    draw_parent_ladder(
        axes[2],
        image,
        blocks,
        n_tokens,
        stage,
        ranks,
        coarse_block_size=coarse_block_size,
        upto="K512",
    )
    axes[2].set_title("coarse ask order\nlarge parent regions", fontsize=10)
    for ax, budget in zip(axes[3:], ["K32", "K128", "K512"]):
        draw_overlay(
            ax,
            image,
            blocks,
            n_tokens,
            stage,
            ranks,
            budget,
            mask_mode=mask_mode,
            learned_inference=learned_inference,
            coarse_block_size=coarse_block_size,
            show_ranks=budget == "K32",
            incremental_only=True,
        )
        pred = predictions.get(budget, "")
        added = summarize_order(stage).get(budget, 0)
        ax.set_title(f"{budget} cumulative actual={actual_tokens[budget]}\n+{added} new detail cells; pred: {pred[:34]}", fontsize=9)
    q = "\n".join(textwrap.wrap(question, 105))
    fig.suptitle(f"heldout record {record_idx}\nQuestion: {q}\nGold: {gold}", fontsize=12, y=1.04)
    legend = (
        "blue = dense semantic/base evidence; parent panel = coarse region asks; "
        "red/orange/green panels show newly added pixel/detail cells, not scale levels; "
        "small numbers = first fine-detail block ranks"
    )
    fig.text(0.5, 0.02, legend, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    path = out / f"record_{record_idx:03d}_selection_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_contact_sheet(paths: list[Path], out: Path) -> Path:
    from PIL import Image, ImageDraw

    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1200, 520))
        thumbs.append((path.name, image.copy()))
    if not thumbs:
        raise ValueError("no example figures to combine")
    width = 1240
    row_h = 560
    canvas = Image.new("RGB", (width, row_h * len(thumbs)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for name, image in thumbs:
        draw.text((16, y + 8), name, fill=(20, 20, 20))
        canvas.paste(image, (20, y + 34))
        y += row_h
    path = out / "contact_sheet.png"
    canvas.save(path)
    return path


@torch.no_grad()
def run(args) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_tokenflow_source(args.tokenflow_repo_dir, args.tokenflow_repo_url)
    tokenizer_ckpt = download_tokenizer_checkpoint(args)
    tokenizer, model = load_tokenflow_model(args, tokenizer_ckpt)
    controller = install_visual_mask(model, mode=args.mask_mode, semantic_dim=args.semantic_dim)
    image_processor = model.get_vision_tower().image_processor
    device = torch.device(args.policy_device if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.policy, device)
    policy.sequential_inference = args.learned_inference == "sequential"
    policy.hierarchical_inference = args.learned_inference == "hierarchical"
    policy.coarse_block_size = args.coarse_block_size

    requested = parse_indices(args.record_indices)
    scan_count = args.count if requested is None else max(requested) + 1
    records = load_records(
        args.dataset,
        args.split,
        scan_count,
        streaming=True,
        min_answer_len=1,
        skip_usable=args.skip_usable,
    )
    indices = requested if requested is not None else list(range(min(args.count, len(records))))

    trace_path = args.out / "selection_traces.jsonl"
    figure_paths: list[Path] = []
    trace_rows: list[dict[str, Any]] = []
    with trace_path.open("w") as trace_file:
        for local_idx in indices:
            record = records[local_idx]
            print(f"viz_record {local_idx}", flush=True)
            image = record.image.convert("RGB")
            image_tensor = process_image(record.image, image_processor, model)
            prompt = build_prompt(record.question, args.conv_template)
            raw_features = raw_visual_features(model, image_tensor)
            projected_features = projected_visual_features(model, raw_features)
            controller.set_features(raw_features)
            n_tokens = int(raw_features.shape[1])
            blocks = build_blocks(n_tokens, args.block_size)
            block_features = block_feature_tensor(projected_features, blocks)
            qfeat = question_feature(model, tokenizer, record.question, args.question_feature)
            pos_features = block_pos_features(n_tokens, blocks)
            ordered = selector_blocks(
                "learned",
                policy,
                raw_features,
                block_features,
                qfeat,
                pos_features,
                blocks,
                args.block_size,
                args.seed + local_idx * 9973,
                device,
            )
            stage = stage_by_block(ordered, blocks, n_tokens)
            ranks = block_order_ranks(ordered, blocks)
            predictions: dict[str, str] = {}
            actual_tokens: dict[str, int] = {}
            selected_by_budget: dict[str, list[int]] = {}
            for budget_name, budget in [("K0", 0), ("K32", 32), ("K128", 128), ("K512", 512), ("full", None)]:
                selected = None if budget is None else selected_from_blocks(ordered, budget, n_tokens)
                pred = generate_answer(model, tokenizer, controller, image_tensor, prompt, selected, args.max_new_tokens)
                predictions[budget_name] = normalize_answer(pred)
                actual_tokens[budget_name] = n_tokens if selected is None else int(selected.numel())
                selected_by_budget[budget_name] = [] if selected is None else [int(x) for x in selected.tolist()]
            figure_path = save_example_figure(
                args.out,
                local_idx,
                image,
                record.question,
                normalize_answer(record.answer),
                predictions,
                blocks,
                n_tokens,
                stage,
                ranks,
                actual_tokens,
                mask_mode=args.mask_mode,
                learned_inference=args.learned_inference,
                coarse_block_size=args.coarse_block_size,
            )
            figure_paths.append(figure_path)
            row = {
                "record_idx": local_idx,
                "question": record.question,
                "gold": normalize_answer(record.answer),
                "predictions": predictions,
                "actual_tokens": actual_tokens,
                "n_tokens": n_tokens,
                "block_size": args.block_size,
                "token_grid": square_grid(n_tokens),
                "stage_block_counts": summarize_order(stage),
                "first_detail_block_ranks": [
                    {"block_idx": idx, "rank": rank + 1, "stage": stage.get(idx)}
                    for idx, rank in sorted(ranks.items(), key=lambda item: item[1])[:16]
                ],
                "selected_tokens": selected_by_budget,
                "figure": str(figure_path),
            }
            trace_rows.append(row)
            trace_file.write(json.dumps(row) + "\n")
            trace_file.flush()

    contact_sheet = make_contact_sheet(figure_paths, args.out)
    summary = {
        "policy": str(args.policy),
        "dataset": args.dataset,
        "split": args.split,
        "skip_usable": args.skip_usable,
        "record_indices": indices,
        "question_feature": args.question_feature,
        "learned_inference": args.learned_inference,
        "coarse_block_size": args.coarse_block_size,
        "mask_mode": args.mask_mode,
        "semantic_dim": args.semantic_dim,
        "block_size": args.block_size,
        "figures": [str(path) for path in figure_paths],
        "contact_sheet": str(contact_sheet),
        "trace_path": str(trace_path),
        "examples": trace_rows,
    }
    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/tokenflow_voi_viz"))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--dataset", default="sionic-ai/textvqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--skip-usable", type=int, default=256)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--record-indices", default="0,3,1,2,4,5")
    parser.add_argument("--model", default=TOKENFLOW_MODEL)
    parser.add_argument("--tokenflow-repo-dir", type=Path, default=Path("external/TokenFlow"))
    parser.add_argument("--tokenflow-repo-url", default=TOKENFLOW_REPO)
    parser.add_argument("--tokenflow-tokenizer-repo", default=TOKENFLOW_TOKENIZER_REPO)
    parser.add_argument("--tokenflow-tokenizer-file", default=TOKENFLOW_TOKENIZER_FILE)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--learned-inference", default="sequential", choices=["static", "sequential", "hierarchical"])
    parser.add_argument("--coarse-block-size", type=int, default=9)
    parser.add_argument("--mask-mode", default="token", choices=["token", "semantic_dense_pixel_sparse"])
    parser.add_argument("--semantic-dim", type=int, default=32)
    parser.add_argument("--question-feature", default="embedding_mean", choices=["lm_hidden", "embedding_mean"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--conv-template", default="qwen_2_5")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    summary = run(args)
    print(json.dumps(summary, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
