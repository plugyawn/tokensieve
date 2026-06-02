"""TokenFlow fixed-consumer VQA budget evaluator.

This script keeps the released TokenFlow VQA checkpoint fixed and only changes
which visual tokens are exposed to the LLM. It is intentionally separate from
newly trained visual bridge code so tokenizer/model quality is not confounded
with a newly trained projection layer.
"""

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from tokenflow_data import RealRecord, load_records, normalize_answer


TOKENFLOW_REPO = "https://github.com/ByteFlow-AI/TokenFlow.git"
TOKENFLOW_MODEL = "ByteVisionLab/Tokenflow-llava-qwen2.5-14B-finetuning"
TOKENFLOW_TOKENIZER_REPO = "ByteFlow-AI/TokenFlow"
TOKENFLOW_TOKENIZER_FILE = "tokenflow_siglip_32k.pt"
IGNORE_INDEX = -100


@dataclass
class EvalRow:
    selector: str
    budget: str
    actual_tokens: int
    accuracy: float
    contains_accuracy: float
    mean_nll: float
    mean_answer_tokens: float
    n: int


class VisualMaskController:
    def __init__(self, mode: str = "token", semantic_dim: int = 32) -> None:
        self.selected: torch.Tensor | None = None
        self.features: torch.Tensor | None = None
        self.mode = mode
        self.semantic_dim = semantic_dim

    def set_selected(self, selected: torch.Tensor | None) -> None:
        self.selected = selected

    def set_features(self, features: torch.Tensor | None) -> None:
        self.features = features

    def apply(self, features: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self.selected is None:
            return features, int(features.shape[1])
        selected = self.selected.to(features.device)
        if self.mode == "semantic_dense_pixel_sparse":
            semantic_dim = min(max(0, int(self.semantic_dim)), int(features.shape[-1]))
            semantic = features[..., :semantic_dim]
            pixel = features[..., semantic_dim:]
            if pixel.shape[-1] == 0:
                return features, int(features.shape[1])
            if selected.numel() == 0:
                pixel = torch.zeros_like(pixel)
                return torch.cat([semantic, pixel], dim=-1), 0
            mask = torch.zeros(features.shape[1], dtype=features.dtype, device=features.device)
            mask[selected.clamp(min=0, max=features.shape[1] - 1)] = 1
            pixel = pixel * mask.view(1, -1, 1)
            return torch.cat([semantic, pixel], dim=-1), int(mask.sum().item())
        if self.mode != "token":
            raise ValueError(f"unknown visual mask mode {self.mode}")
        if selected.numel() == 0:
            return torch.zeros_like(features), 0
        mask = torch.zeros(features.shape[1], dtype=features.dtype, device=features.device)
        mask[selected.clamp(min=0, max=features.shape[1] - 1)] = 1
        return features * mask.view(1, -1, 1), int(mask.sum().item())


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_budgets(value: str) -> list[int | None]:
    out: list[int | None] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in {"full", "all"}:
            out.append(None)
        else:
            budget = int(part)
            if budget < 0:
                raise ValueError(f"budget must be non-negative or full, got {part}")
            out.append(budget)
    if not out:
        raise ValueError("at least one budget is required")
    return out


def budget_name(budget: int | None) -> str:
    return "full" if budget is None else str(budget)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name}")


def ensure_tokenflow_source(repo_dir: Path, repo_url: str) -> None:
    if (repo_dir / "i2t" / "llava").exists() and (repo_dir / "tokenflow").exists():
        return
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)


def download_tokenizer_checkpoint(args) -> Path:
    from huggingface_hub import hf_hub_download

    cache_path = Path(
        hf_hub_download(
            repo_id=args.tokenflow_tokenizer_repo,
            filename=args.tokenflow_tokenizer_file,
        )
    )
    args.out.mkdir(parents=True, exist_ok=True)
    local_path = args.out / args.tokenflow_tokenizer_file
    if not local_path.exists() or local_path.stat().st_size != cache_path.stat().st_size:
        shutil.copy2(cache_path, local_path)
    return local_path


def load_tokenflow_model(args, tokenizer_ckpt: Path):
    repo_dir = args.tokenflow_repo_dir.resolve()
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "i2t"))

    from transformers import AutoTokenizer
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = LlavaQwenConfig.from_pretrained(args.model)
    config.mm_vision_tower = str(tokenizer_ckpt)
    config.mm_vision_vq_type = "TOKENFLOW"
    config.tokenizer_model_max_length = args.context_len
    config.use_cache = False

    model = LlavaQwenForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    )
    model.eval()
    model.requires_grad_(False)
    return tokenizer, model


def install_visual_mask(model, mode: str = "token", semantic_dim: int = 32) -> VisualMaskController:
    controller = VisualMaskController(mode=mode, semantic_dim=semantic_dim)

    def encode_images_with_mask(self, images):
        image_features = controller.features
        if image_features is None:
            image_features = self.get_model().get_vision_tower()(images)
        image_features, _ = controller.apply(image_features)
        return self.get_model().mm_projector(image_features)

    model.encode_images = types.MethodType(encode_images_with_mask, model)
    return controller


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def build_prompt(question: str, conv_template: str) -> str:
    from llava.conversation import conv_templates

    conv = conv_templates[conv_template].copy()
    question = question.strip()
    if "<image>" not in question:
        question = "<image>\n" + question
    question = question + "\nAnswer the question using a single short phrase."
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def tokenize_image_prompt(prompt: str, tokenizer) -> torch.Tensor:
    from llava.mm_utils import tokenizer_image_token

    return tokenizer_image_token(prompt, tokenizer, return_tensors="pt")


def process_image(image, image_processor, model) -> torch.Tensor:
    from llava.mm_utils import process_images

    tensor = process_images([image], image_processor, model.config)
    return tensor.to(device=model_device(model), dtype=next(model.parameters()).dtype)


@torch.no_grad()
def raw_visual_features(model, image_tensor: torch.Tensor) -> torch.Tensor:
    return model.get_model().get_vision_tower()(image_tensor)


def square_grid(n_tokens: int) -> int | None:
    grid = int(math.sqrt(n_tokens))
    return grid if grid * grid == n_tokens else None


def build_blocks(n_tokens: int, block_size: int) -> list[list[int]]:
    grid = square_grid(n_tokens)
    if grid is None:
        return [list(range(start, min(start + block_size, n_tokens))) for start in range(0, n_tokens, block_size)]
    blocks: list[list[int]] = []
    for y in range(0, grid, block_size):
        for x in range(0, grid, block_size):
            block = []
            for yy in range(y, min(y + block_size, grid)):
                for xx in range(x, min(x + block_size, grid)):
                    block.append(yy * grid + xx)
            blocks.append(block)
    return blocks


def order_blocks(selector: str, features: torch.Tensor, block_size: int, seed: int) -> list[list[int]]:
    n_tokens = int(features.shape[1])
    blocks = build_blocks(n_tokens, block_size)
    if selector == "random":
        rng = random.Random(seed)
        rng.shuffle(blocks)
        return blocks
    if selector == "center":
        grid = square_grid(n_tokens)
        if grid is None:
            return blocks
        center = (grid - 1) / 2.0

        def distance(block: list[int]) -> float:
            ys = [idx // grid for idx in block]
            xs = [idx % grid for idx in block]
            return (sum((y - center) ** 2 for y in ys) + sum((x - center) ** 2 for x in xs)) / max(1, len(block))

        return sorted(blocks, key=distance)
    if selector == "magnitude":
        scores = features.float().pow(2).mean(dim=-1).squeeze(0)
        return sorted(blocks, key=lambda block: float(scores[block].mean().item()), reverse=True)
    if selector in {"none", "zero", "full"}:
        return blocks
    raise ValueError(f"unknown selector {selector}")


def selected_from_blocks(blocks: list[list[int]], budget: int | None, n_tokens: int) -> torch.Tensor | None:
    if budget is None:
        return None
    if budget <= 0:
        return torch.empty(0, dtype=torch.long)
    selected: list[int] = []
    for block in blocks:
        selected.extend(block)
        if len(selected) >= budget:
            break
    selected = sorted(set(idx for idx in selected if 0 <= idx < n_tokens))
    return torch.tensor(selected, dtype=torch.long)


@torch.no_grad()
def answer_nll(model, tokenizer, controller: VisualMaskController, image_tensor: torch.Tensor, prompt: str, answer: str, selected: torch.Tensor | None) -> tuple[float, int]:
    controller.set_selected(selected)
    prompt_ids = tokenize_image_prompt(prompt, tokenizer)
    full_prompt = prompt + " " + answer.strip() + "<|im_end|>\n"
    input_ids = tokenize_image_prompt(full_prompt, tokenizer).unsqueeze(0).to(model_device(model))
    labels = input_ids.clone()
    labels[:, : prompt_ids.numel()] = IGNORE_INDEX
    answer_tokens = int((labels != IGNORE_INDEX).sum().item())
    output = model(
        input_ids=input_ids,
        labels=labels,
        images=image_tensor,
        image_sizes=[image_tensor.shape[-2:]],
        use_cache=False,
    )
    return float(output.loss.detach().float().item()), answer_tokens


@torch.no_grad()
def generate_answer(model, tokenizer, controller: VisualMaskController, image_tensor: torch.Tensor, prompt: str, selected: torch.Tensor | None, max_new_tokens: int) -> str:
    controller.set_selected(selected)
    input_ids = tokenize_image_prompt(prompt, tokenizer).unsqueeze(0).to(model_device(model))
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_tensor.shape[-2:]],
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    continuation = output_ids[0, input_ids.shape[1] :]
    if continuation.numel() == 0:
        continuation = output_ids[0]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def greedy_oracle_blocks(
    args,
    model,
    tokenizer,
    controller: VisualMaskController,
    image_tensor: torch.Tensor,
    prompt: str,
    answer: str,
    features: torch.Tensor,
    max_budget: int,
) -> list[list[int]]:
    remaining = build_blocks(int(features.shape[1]), args.block_size)
    selected_blocks: list[list[int]] = []
    selected_tokens: set[int] = set()
    while remaining and len(selected_tokens) < max_budget:
        best_i = 0
        best_loss = float("inf")
        for i, block in enumerate(remaining):
            trial = torch.tensor(sorted(selected_tokens.union(block)), dtype=torch.long)
            loss, _ = answer_nll(model, tokenizer, controller, image_tensor, prompt, answer, trial)
            if loss < best_loss:
                best_loss = loss
                best_i = i
        block = remaining.pop(best_i)
        selected_blocks.append(block)
        selected_tokens.update(block)
    return selected_blocks


def load_eval_records(args) -> list[RealRecord]:
    return load_records(
        args.dataset,
        args.split,
        args.count,
        streaming=True,
        min_answer_len=1,
        skip_usable=args.skip_usable,
    )


def plot_curves(rows: list[EvalRow], out: Path) -> None:
    finite = [row for row in rows if row.budget != "full"]
    if not finite:
        return
    selectors = sorted(set(row.selector for row in finite))
    plt.figure(figsize=(8, 5))
    for selector in selectors:
        xs = [int(row.budget) for row in finite if row.selector == selector]
        ys = [row.accuracy for row in finite if row.selector == selector]
        if xs:
            plt.plot(xs, ys, marker="o", label=selector)
    plt.xlabel("revealed TokenFlow visual tokens")
    plt.ylabel("generated exact-match accuracy")
    plt.title("TokenFlow fixed-consumer VQA budget curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "budget_curve.png", dpi=180)
    plt.close()


def run_eval(args) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_tokenflow_source(args.tokenflow_repo_dir, args.tokenflow_repo_url)
    tokenizer_ckpt = download_tokenizer_checkpoint(args)
    tokenizer, model = load_tokenflow_model(args, tokenizer_ckpt)
    controller = install_visual_mask(model)
    image_processor = model.get_vision_tower().image_processor
    records = load_eval_records(args)
    budgets = parse_budgets(args.budgets)
    selectors = [x.strip() for x in args.selectors.split(",") if x.strip()]

    metric_accum: dict[tuple[str, str], dict[str, Any]] = {}
    example_rows: list[dict[str, Any]] = []
    max_oracle_budget = max((b for b in budgets if b is not None), default=0)

    for record_idx, record in enumerate(records):
        if args.log_every > 0 and record_idx % args.log_every == 0:
            print(f"tokenflow_eval_record {record_idx}/{len(records)}", flush=True)
        image_tensor = process_image(record.image, image_processor, model)
        prompt = build_prompt(record.question, args.conv_template)
        features = raw_visual_features(model, image_tensor)
        controller.set_features(features)
        n_tokens = int(features.shape[1])
        oracle_blocks = None
        if "oracle" in selectors and record_idx < args.oracle_max_examples and max_oracle_budget > 0:
            oracle_blocks = greedy_oracle_blocks(args, model, tokenizer, controller, image_tensor, prompt, record.answer, features, max_oracle_budget)

        for selector in selectors:
            if selector == "oracle" and oracle_blocks is None:
                continue
            if selector == "oracle":
                blocks = oracle_blocks or []
            else:
                blocks = order_blocks(selector, features, args.block_size, args.seed + record_idx * 9973)
            if selector == "full":
                active_budgets = [None]
            elif selector in {"none", "zero"}:
                active_budgets = [0]
            else:
                active_budgets = budgets
            for budget in active_budgets:
                if selector == "full":
                    selected = None
                elif selector in {"none", "zero"}:
                    selected = torch.empty(0, dtype=torch.long)
                else:
                    selected = selected_from_blocks(blocks, budget, n_tokens)
                actual_tokens = n_tokens if selected is None else int(selected.numel())
                loss, answer_tokens = answer_nll(model, tokenizer, controller, image_tensor, prompt, record.answer, selected)
                pred = generate_answer(model, tokenizer, controller, image_tensor, prompt, selected, args.max_new_tokens)
                pred_norm = normalize_answer(pred)
                answer_norm = normalize_answer(record.answer)
                exact = float(pred_norm == answer_norm)
                contains = float(answer_norm in pred_norm.split() or answer_norm == pred_norm or (answer_norm and answer_norm in pred_norm))
                key = (selector, budget_name(budget))
                metric = metric_accum.setdefault(
                    key,
                    {"loss_sum": 0.0, "answer_tokens": 0, "exact": 0.0, "contains": 0.0, "n": 0, "actual_tokens": []},
                )
                metric["loss_sum"] += loss
                metric["answer_tokens"] += answer_tokens
                metric["exact"] += exact
                metric["contains"] += contains
                metric["n"] += 1
                metric["actual_tokens"].append(actual_tokens)
                if len(example_rows) < args.max_examples:
                    example_rows.append(
                        {
                            "record_idx": record_idx,
                            "selector": selector,
                            "budget": budget_name(budget),
                            "actual_tokens": actual_tokens,
                            "question": record.question,
                            "gold": answer_norm,
                            "prediction": pred,
                            "prediction_norm": pred_norm,
                            "exact": exact,
                            "nll": loss,
                        }
                    )
        controller.set_features(None)

    rows: list[EvalRow] = []
    for (selector, budget), metric in sorted(metric_accum.items()):
        n = metric["n"]
        rows.append(
            EvalRow(
                selector=selector,
                budget=budget,
                actual_tokens=int(round(sum(metric["actual_tokens"]) / max(1, n))),
                accuracy=metric["exact"] / max(1, n),
                contains_accuracy=metric["contains"] / max(1, n),
                mean_nll=metric["loss_sum"] / max(1, n),
                mean_answer_tokens=metric["answer_tokens"] / max(1, n),
                n=n,
            )
        )

    metric_dicts = [row.__dict__ for row in rows]
    write_csv(metric_dicts, args.out / "generated_metrics.csv")
    write_csv(metric_dicts, args.out / "loss_metrics.csv")
    with (args.out / "examples.jsonl").open("w") as f:
        for row in example_rows:
            f.write(json.dumps(row) + "\n")
    plot_curves(rows, args.out)
    summary = {
        "model": args.model,
        "tokenflow_tokenizer": str(tokenizer_ckpt),
        "dataset": args.dataset,
        "split": args.split,
        "count": len(records),
        "skip_usable": args.skip_usable,
        "selectors": selectors,
        "budgets": [budget_name(x) for x in budgets],
        "block_size": args.block_size,
        "metrics": metric_dicts,
    }
    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/tokenflow_vqa_eval"))
    parser.add_argument("--dataset", default="sionic-ai/textvqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--skip-usable", type=int, default=0)
    parser.add_argument("--model", default=TOKENFLOW_MODEL)
    parser.add_argument("--tokenflow-repo-dir", type=Path, default=Path("external/TokenFlow"))
    parser.add_argument("--tokenflow-repo-url", default=TOKENFLOW_REPO)
    parser.add_argument("--tokenflow-tokenizer-repo", default=TOKENFLOW_TOKENIZER_REPO)
    parser.add_argument("--tokenflow-tokenizer-file", default=TOKENFLOW_TOKENIZER_FILE)
    parser.add_argument("--selectors", default="full,none,center,random,magnitude")
    parser.add_argument("--budgets", default="0,32,128,512,full")
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--oracle-max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--conv-template", default="qwen_2_5")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run_eval(args)
    print(json.dumps(summary, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
