"""Question-conditioned VOI selector for the fixed TokenFlow VQA consumer.

This keeps the released TokenFlow VQA checkpoint frozen.  The only trained
component is a small block-ranking policy that sees projected visual block
features, a frozen-LM question embedding, and spatial features.  Supervision is
generated from real VQA examples by measuring the answer-loss reduction from
revealing each block under the fixed consumer.
"""

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenflow_data import RealRecord, load_records, normalize_answer
from tokenflow_vqa_eval import (
    TOKENFLOW_MODEL,
    TOKENFLOW_REPO,
    TOKENFLOW_TOKENIZER_FILE,
    TOKENFLOW_TOKENIZER_REPO,
    VisualMaskController,
    answer_nll,
    build_blocks,
    build_prompt,
    budget_name,
    download_tokenizer_checkpoint,
    dtype_from_name,
    ensure_tokenflow_source,
    generate_answer,
    install_visual_mask,
    load_tokenflow_model,
    model_device,
    order_blocks,
    parse_budgets,
    process_image,
    raw_visual_features,
    selected_from_blocks,
    square_grid,
)


@dataclass
class LabelRecord:
    block_features: torch.Tensor
    question_feature: torch.Tensor
    pos_features: torch.Tensor
    deltas: torch.Tensor
    blocks: list[list[int]]
    question: str
    answer: str
    base_loss: float


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


class QuestionBlockPolicy(nn.Module):
    def __init__(self, visual_dim: int, question_dim: int, pos_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.question_norm = nn.LayerNorm(question_dim)
        self.visual = nn.Linear(visual_dim, hidden)
        self.question = nn.Linear(question_dim, hidden)
        self.pos = nn.Linear(pos_dim, hidden)
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, block_features: torch.Tensor, question_feature: torch.Tensor, pos_features: torch.Tensor) -> torch.Tensor:
        if question_feature.dim() == 1:
            question_feature = question_feature.unsqueeze(0).expand(block_features.shape[0], -1)
        h = self.visual(self.visual_norm(block_features))
        h = h + self.question(self.question_norm(question_feature))
        h = h + self.pos(pos_features)
        return self.out(h).squeeze(-1)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_real_records(args, count: int, skip_usable: int) -> list[RealRecord]:
    return load_records(
        args.dataset,
        args.split,
        count,
        streaming=True,
        min_answer_len=1,
        skip_usable=skip_usable,
    )


@torch.no_grad()
def projected_visual_features(model, raw_features: torch.Tensor) -> torch.Tensor:
    return model.get_model().mm_projector(raw_features).detach().float()


@torch.no_grad()
def question_feature(model, tokenizer, question: str, method: str) -> torch.Tensor:
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    text = question.strip()
    if method == "lm_hidden":
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=128)
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device) if hasattr(encoded, "attention_mask") else None
        if input_ids.numel() == 0:
            return torch.zeros(embedding.weight.shape[1], dtype=torch.float32)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        return output.hidden_states[-1][0, -1].detach().float().cpu()
    if method != "embedding_mean":
        raise ValueError(f"unknown question feature method {method}")
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(device)
    if input_ids.numel() == 0:
        return torch.zeros(embedding.weight.shape[1], dtype=torch.float32)
    return embedding(input_ids).float().mean(dim=1).squeeze(0).detach().cpu()


def block_feature_tensor(features: torch.Tensor, blocks: list[list[int]]) -> torch.Tensor:
    token_features = features.squeeze(0).detach().float().cpu()
    return torch.stack([token_features[block].mean(dim=0) for block in blocks], dim=0)


def block_pos_features(n_tokens: int, blocks: list[list[int]]) -> torch.Tensor:
    grid = square_grid(n_tokens)
    rows = []
    if grid is None:
        denom = max(1, n_tokens - 1)
        for block in blocks:
            center = sum(block) / max(1, len(block))
            x = 2.0 * center / denom - 1.0
            width = len(block) / max(1, n_tokens)
            rows.append([x, 0.0, abs(x), width, 1.0])
        return torch.tensor(rows, dtype=torch.float32)

    denom = max(1, grid - 1)
    for block in blocks:
        ys = torch.tensor([idx // grid for idx in block], dtype=torch.float32)
        xs = torch.tensor([idx % grid for idx in block], dtype=torch.float32)
        cx = 2.0 * float(xs.mean().item()) / denom - 1.0
        cy = 2.0 * float(ys.mean().item()) / denom - 1.0
        dist = math.sqrt(cx * cx + cy * cy)
        width = (float(xs.max().item() - xs.min().item()) + 1.0) / grid
        height = (float(ys.max().item() - ys.min().item()) + 1.0) / grid
        rows.append([cx, cy, dist, width, height])
    return torch.tensor(rows, dtype=torch.float32)


def policy_pos_features(base_pos: torch.Tensor, selected_count: int, n_tokens: int) -> torch.Tensor:
    prefix = float(selected_count) / max(1, n_tokens)
    prefix_col = torch.full((base_pos.shape[0], 1), prefix, dtype=base_pos.dtype)
    return torch.cat([base_pos, prefix_col], dim=-1)


def order_blocks_uniform(features: torch.Tensor, block_size: int) -> list[list[int]]:
    n_tokens = int(features.shape[1])
    blocks = build_blocks(n_tokens, block_size)
    grid = square_grid(n_tokens)
    block_grid = square_grid(len(blocks))
    if grid is None or block_grid is None:
        return blocks
    center = (block_grid - 1) / 2.0
    keyed = []
    for i, block in enumerate(blocks):
        y = i // block_grid
        x = i % block_grid
        ring = max(abs(y - center), abs(x - center))
        parity = (x + y) % 2
        keyed.append((ring, parity, y, x, block))
    return [item[-1] for item in sorted(keyed)]


def order_blocks_qsim(block_features: torch.Tensor, question: torch.Tensor, blocks: list[list[int]]) -> list[list[int]]:
    q = F.normalize(question.float(), dim=0)
    v = F.normalize(block_features.float(), dim=-1)
    scores = torch.mv(v, q)
    order = torch.argsort(scores, descending=True).tolist()
    return [blocks[i] for i in order]


def order_blocks_learned(
    policy: QuestionBlockPolicy,
    block_features: torch.Tensor,
    question: torch.Tensor,
    pos_features: torch.Tensor,
    blocks: list[list[int]],
    device: torch.device,
) -> list[list[int]]:
    policy.eval()
    with torch.no_grad():
        scores = policy(
            block_features.to(device),
            question.to(device),
            pos_features.to(device),
        ).detach().float().cpu()
    order = torch.argsort(scores, descending=True).tolist()
    return [blocks[i] for i in order]


def order_blocks_learned_sequential(
    policy: QuestionBlockPolicy,
    block_features: torch.Tensor,
    question: torch.Tensor,
    base_pos_features: torch.Tensor,
    blocks: list[list[int]],
    n_tokens: int,
    device: torch.device,
) -> list[list[int]]:
    remaining = list(range(len(blocks)))
    ordered: list[list[int]] = []
    selected_count = 0
    policy.eval()
    while remaining:
        pos = policy_pos_features(base_pos_features[remaining], selected_count, n_tokens)
        with torch.no_grad():
            scores = policy(
                block_features[remaining].to(device),
                question.to(device),
                pos.to(device),
            ).detach().float().cpu()
        pick_local = int(torch.argmax(scores).item())
        pick = remaining.pop(pick_local)
        ordered.append(blocks[pick])
        selected_count = min(n_tokens, selected_count + len(blocks[pick]))
    return ordered


def order_blocks_learned_hierarchical(
    policy: QuestionBlockPolicy,
    block_features: torch.Tensor,
    question: torch.Tensor,
    base_pos_features: torch.Tensor,
    blocks: list[list[int]],
    n_tokens: int,
    coarse_block_size: int,
    device: torch.device,
) -> list[list[int]]:
    """Progressive parent-region then child-block ordering.

    The action space remains final-grid child blocks, but inference is nested:
    score coarse parent regions, open the best parent, then reveal child blocks
    inside that parent before moving to another parent. This is a minimal
    systems-friendly approximation of "look coarsely, then sharpen locally"
    without changing the fixed TokenFlow consumer.
    """
    token_grid = square_grid(n_tokens)
    if token_grid is None or coarse_block_size <= 0 or coarse_block_size <= 3:
        return order_blocks_learned_sequential(policy, block_features, question, base_pos_features, blocks, n_tokens, device)
    by_block = {tuple(block): i for i, block in enumerate(blocks)}
    parent_blocks = build_blocks(n_tokens, coarse_block_size)
    child_sets: list[list[int]] = []
    for parent in parent_blocks:
        parent_set = set(parent)
        children = [by_block[tuple(block)] for block in blocks if set(block).issubset(parent_set)]
        if children:
            child_sets.append(children)
    if not child_sets:
        return order_blocks_learned_sequential(policy, block_features, question, base_pos_features, blocks, n_tokens, device)

    remaining_parents = list(range(len(child_sets)))
    selected_children: set[int] = set()
    ordered: list[list[int]] = []
    selected_count = 0
    policy.eval()
    while remaining_parents:
        parent_scores: list[tuple[float, int]] = []
        for parent_idx in remaining_parents:
            children = [idx for idx in child_sets[parent_idx] if idx not in selected_children]
            if not children:
                parent_scores.append((float("-inf"), parent_idx))
                continue
            pos = policy_pos_features(base_pos_features[children], selected_count, n_tokens)
            with torch.no_grad():
                scores = policy(
                    block_features[children].to(device),
                    question.to(device),
                    pos.to(device),
                ).detach().float().cpu()
            parent_scores.append((float(scores.max().item()), parent_idx))
        _, picked_parent = max(parent_scores, key=lambda item: item[0])
        remaining_parents.remove(picked_parent)
        local_remaining = [idx for idx in child_sets[picked_parent] if idx not in selected_children]
        while local_remaining:
            pos = policy_pos_features(base_pos_features[local_remaining], selected_count, n_tokens)
            with torch.no_grad():
                scores = policy(
                    block_features[local_remaining].to(device),
                    question.to(device),
                    pos.to(device),
                ).detach().float().cpu()
            pick_local = int(torch.argmax(scores).item())
            child_idx = local_remaining.pop(pick_local)
            selected_children.add(child_idx)
            ordered.append(blocks[child_idx])
            selected_count = min(n_tokens, selected_count + len(blocks[child_idx]))
    if len(ordered) < len(blocks):
        for idx, block in enumerate(blocks):
            if idx not in selected_children:
                ordered.append(block)
    return ordered


def selector_blocks(
    selector: str,
    policy: QuestionBlockPolicy | None,
    raw_features: torch.Tensor,
    block_features: torch.Tensor,
    question: torch.Tensor,
    pos_features: torch.Tensor,
    blocks: list[list[int]],
    block_size: int,
    seed: int,
    device: torch.device,
) -> list[list[int]]:
    if selector == "learned":
        if policy is None:
            raise ValueError("learned selector requested without a policy")
        if pos_features.shape[-1] == 5:
            pos_features = policy_pos_features(pos_features, 0, int(raw_features.shape[1]))
        if getattr(policy, "hierarchical_inference", False):
            base_pos = pos_features[:, :5]
            return order_blocks_learned_hierarchical(
                policy,
                block_features,
                question,
                base_pos,
                blocks,
                int(raw_features.shape[1]),
                getattr(policy, "coarse_block_size", 9),
                device,
            )
        if getattr(policy, "sequential_inference", False):
            base_pos = pos_features[:, :5]
            return order_blocks_learned_sequential(
                policy,
                block_features,
                question,
                base_pos,
                blocks,
                int(raw_features.shape[1]),
                device,
            )
        return order_blocks_learned(policy, block_features, question, pos_features, blocks, device)
    if selector == "qsim":
        return order_blocks_qsim(block_features, question, blocks)
    if selector == "uniform":
        return order_blocks_uniform(raw_features, block_size)
    return order_blocks(selector, raw_features, block_size, seed)


def label_candidate_indices(
    args,
    raw_features: torch.Tensor,
    block_features: torch.Tensor,
    question: torch.Tensor,
    pos_features: torch.Tensor,
    blocks: list[list[int]],
    seed: int,
) -> list[int]:
    if args.label_max_blocks <= 0 or args.label_max_blocks >= len(blocks):
        return list(range(len(blocks)))
    by_key = {tuple(block): i for i, block in enumerate(blocks)}
    orders: list[list[int]] = []
    for selector in [x.strip() for x in args.label_candidate_selectors.split(",") if x.strip()]:
        if selector in {"learned", "full", "none", "zero"}:
            continue
        ordered_blocks = selector_blocks(
            selector,
            None,
            raw_features,
            block_features,
            question,
            pos_features,
            blocks,
            args.block_size,
            seed,
            torch.device("cpu"),
        )
        orders.append([by_key[tuple(block)] for block in ordered_blocks if tuple(block) in by_key])
    if not orders:
        return list(range(min(args.label_max_blocks, len(blocks))))

    picked: list[int] = []
    seen: set[int] = set()
    max_len = max(len(order) for order in orders)
    for rank in range(max_len):
        for order in orders:
            if rank >= len(order):
                continue
            idx = order[rank]
            if idx in seen:
                continue
            seen.add(idx)
            picked.append(idx)
            if len(picked) >= args.label_max_blocks:
                return picked
    return picked


def listnet_loss(logits: torch.Tensor, deltas: torch.Tensor, temperature: float) -> torch.Tensor:
    centered = deltas - deltas.mean()
    scale = centered.std(unbiased=False).clamp_min(1e-4)
    target = torch.softmax((centered / scale) / temperature, dim=0)
    return -(target * F.log_softmax(logits, dim=0)).sum()


def make_label_records(args, model, tokenizer, controller: VisualMaskController, image_processor) -> list[LabelRecord]:
    records = load_real_records(args, args.train_count, args.train_skip_usable)
    out: list[LabelRecord] = []
    for record_idx, record in enumerate(records):
        if args.log_every > 0 and record_idx % args.log_every == 0:
            print(f"voi_label_record {record_idx}/{len(records)}", flush=True)
        image_tensor = process_image(record.image, image_processor, model)
        prompt = build_prompt(record.question, args.conv_template)
        raw_features = raw_visual_features(model, image_tensor)
        projected_features = projected_visual_features(model, raw_features)
        controller.set_features(raw_features)
        n_tokens = int(raw_features.shape[1])
        blocks = build_blocks(n_tokens, args.block_size)
        all_block_features = block_feature_tensor(projected_features, blocks)
        qfeat = question_feature(model, tokenizer, record.question, args.question_feature)
        all_base_pos = block_pos_features(n_tokens, blocks)
        candidate_indices = label_candidate_indices(
            args,
            raw_features,
            all_block_features,
            qfeat,
            all_base_pos,
            blocks,
            args.seed + record_idx * 9973,
        )
        blocks = [blocks[i] for i in candidate_indices]
        block_features = all_block_features[candidate_indices]
        base_pos = all_base_pos[candidate_indices]
        label_budgets = parse_budgets(args.sequential_label_budgets)
        label_budgets = [0 if b is None else int(b) for b in label_budgets]
        if args.label_mode == "single":
            label_budgets = [0]
        if not label_budgets or label_budgets[0] != 0:
            label_budgets = [0] + label_budgets
        selected_tokens: set[int] = set()
        for stage_idx, stage_budget in enumerate(label_budgets):
            if len(selected_tokens) < stage_budget:
                print(
                    f"warning: selected prefix {len(selected_tokens)} below requested stage {stage_budget}; "
                    "continuing with current prefix",
                    flush=True,
                )
            remaining_indices = [i for i, block in enumerate(blocks) if not set(block).issubset(selected_tokens)]
            if not remaining_indices:
                break
            current_selected = torch.tensor(sorted(selected_tokens), dtype=torch.long)
            current_loss, _ = answer_nll(model, tokenizer, controller, image_tensor, prompt, record.answer, current_selected)
            deltas = []
            for local_idx, block_idx in enumerate(remaining_indices):
                if args.log_label_blocks and local_idx % args.log_label_blocks == 0:
                    print(
                        f"voi_label_record {record_idx} stage {stage_budget} block {local_idx}/{len(remaining_indices)}",
                        flush=True,
                    )
                block = blocks[block_idx]
                trial = torch.tensor(sorted(selected_tokens.union(block)), dtype=torch.long)
                loss, _ = answer_nll(model, tokenizer, controller, image_tensor, prompt, record.answer, trial)
                deltas.append(current_loss - loss)
            stage_deltas = torch.tensor(deltas, dtype=torch.float32)
            out.append(
                LabelRecord(
                    block_features=block_features[remaining_indices],
                    question_feature=qfeat,
                    pos_features=policy_pos_features(base_pos[remaining_indices], len(selected_tokens), n_tokens),
                    deltas=stage_deltas,
                    blocks=[blocks[i] for i in remaining_indices],
                    question=record.question,
                    answer=record.answer,
                    base_loss=float(current_loss),
                )
            )
            if args.label_mode != "sequential" or stage_idx + 1 >= len(label_budgets):
                continue
            next_budget = min(n_tokens, label_budgets[stage_idx + 1])
            ranked_remaining = [remaining_indices[i] for i in torch.argsort(stage_deltas, descending=True).tolist()]
            for block_idx in ranked_remaining:
                selected_tokens.update(blocks[block_idx])
                if len(selected_tokens) >= next_budget:
                    break
        controller.set_features(None)
    return out


def train_policy(args, labels: list[LabelRecord], device: torch.device) -> QuestionBlockPolicy:
    if not labels:
        raise RuntimeError("no label records available")
    policy = QuestionBlockPolicy(
        visual_dim=int(labels[0].block_features.shape[-1]),
        question_dim=int(labels[0].question_feature.shape[-1]),
        pos_dim=int(labels[0].pos_features.shape[-1]),
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    loss_rows = []
    for epoch in range(args.epochs):
        order = list(range(len(labels)))
        rng.shuffle(order)
        total = 0.0
        opt.zero_grad(set_to_none=True)
        for step, idx in enumerate(order, start=1):
            item = labels[idx]
            logits = policy(
                item.block_features.to(device),
                item.question_feature.to(device),
                item.pos_features.to(device),
            )
            loss = listnet_loss(logits, item.deltas.to(device), args.target_temperature)
            (loss / args.grad_accum).backward()
            total += float(loss.detach().cpu().item())
            if step % args.grad_accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
        mean_loss = total / max(1, len(order))
        loss_rows.append({"epoch": epoch + 1, "listnet_loss": mean_loss})
        print(f"voi_policy_epoch {epoch + 1}/{args.epochs} listnet_loss={mean_loss:.6f}", flush=True)
    write_csv(loss_rows, args.out / "policy_loss.csv")
    return policy


def eval_selectors(args, model, tokenizer, controller: VisualMaskController, image_processor, policy: QuestionBlockPolicy | None) -> list[EvalRow]:
    skip = args.eval_skip_usable if args.eval_skip_usable >= 0 else args.train_skip_usable + args.train_count
    records = load_real_records(args, args.eval_count, skip)
    budgets = parse_budgets(args.budgets)
    selectors = [x.strip() for x in args.selectors.split(",") if x.strip()]
    metric_accum: dict[tuple[str, str], dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    device = torch.device(args.policy_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if policy is not None:
        policy.to(device)
        policy.sequential_inference = args.learned_inference == "sequential"
        policy.hierarchical_inference = args.learned_inference == "hierarchical"
        policy.coarse_block_size = args.coarse_block_size

    for record_idx, record in enumerate(records):
        if args.log_every > 0 and record_idx % args.log_every == 0:
            print(f"voi_eval_record {record_idx}/{len(records)}", flush=True)
        image_tensor = process_image(record.image, image_processor, model)
        prompt = build_prompt(record.question, args.conv_template)
        raw_features = raw_visual_features(model, image_tensor)
        projected_features = projected_visual_features(model, raw_features)
        controller.set_features(raw_features)
        n_tokens = int(raw_features.shape[1])
        base_blocks = build_blocks(n_tokens, args.block_size)
        block_features = block_feature_tensor(projected_features, base_blocks)
        qfeat = question_feature(model, tokenizer, record.question, args.question_feature)
        pos_features = block_pos_features(n_tokens, base_blocks)
        learned_pos_features = policy_pos_features(pos_features, 0, n_tokens)
        for selector in selectors:
            if selector == "full":
                active_budgets = [None]
                ordered = base_blocks
            elif selector in {"none", "zero"}:
                active_budgets = [0]
                ordered = base_blocks
            else:
                active_budgets = budgets
                ordered = selector_blocks(
                    selector,
                    policy,
                    raw_features,
                    block_features,
                    qfeat,
                    learned_pos_features if selector == "learned" else pos_features,
                    base_blocks,
                    args.block_size,
                    args.seed + record_idx * 9973,
                    device,
                )
            for budget in active_budgets:
                if selector == "full":
                    selected = None
                elif selector in {"none", "zero"}:
                    selected = torch.empty(0, dtype=torch.long)
                else:
                    selected = selected_from_blocks(ordered, budget, n_tokens)
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
                if len(examples) < args.max_examples:
                    examples.append(
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

    rows = []
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
    with (args.out / "selector_examples.jsonl").open("w") as f:
        for row in examples:
            f.write(json.dumps(row) + "\n")
    return rows


def plot_curves(rows: list[EvalRow], out: Path) -> None:
    finite = [row for row in rows if row.budget != "full"]
    if not finite:
        return
    plt.figure(figsize=(8, 5))
    for selector in sorted(set(row.selector for row in finite)):
        points = sorted((int(row.budget), row.accuracy) for row in finite if row.selector == selector)
        if points:
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o", label=selector)
    plt.xlabel("revealed TokenFlow visual tokens")
    plt.ylabel("generated exact-match accuracy")
    plt.title("Question-conditioned VOI selector on fixed TokenFlow VQA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "selector_budget_curve.png", dpi=180)
    plt.close()


def label_summary(labels: list[LabelRecord]) -> dict[str, float]:
    if not labels:
        return {}
    deltas = torch.cat([item.deltas for item in labels])
    positive = (deltas > 0).float().mean().item()
    return {
        "label_records": float(len(labels)),
        "mean_delta": float(deltas.mean().item()),
        "max_delta": float(deltas.max().item()),
        "min_delta": float(deltas.min().item()),
        "positive_delta_fraction": float(positive),
    }


def run(args) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_tokenflow_source(args.tokenflow_repo_dir, args.tokenflow_repo_url)
    tokenizer_ckpt = download_tokenizer_checkpoint(args)
    tokenizer, model = load_tokenflow_model(args, tokenizer_ckpt)
    controller = install_visual_mask(model, mode=args.mask_mode, semantic_dim=args.semantic_dim)
    image_processor = model.get_vision_tower().image_processor
    labels = make_label_records(args, model, tokenizer, controller, image_processor)
    policy_device = torch.device(args.policy_device)
    if policy_device.type == "cuda" and not torch.cuda.is_available():
        policy_device = torch.device("cpu")
    policy = train_policy(args, labels, policy_device)
    torch.save(
        {
            "model_state": policy.state_dict(),
            "visual_dim": int(labels[0].block_features.shape[-1]),
            "question_dim": int(labels[0].question_feature.shape[-1]),
            "pos_dim": int(labels[0].pos_features.shape[-1]),
            "hidden": args.hidden,
            "dropout": args.dropout,
            "block_size": args.block_size,
        },
        args.out / "tokenflow_voi_policy.pt",
    )
    rows = eval_selectors(args, model, tokenizer, controller, image_processor, policy)
    metric_dicts = [row.__dict__ for row in rows]
    write_csv(metric_dicts, args.out / "selector_metrics.csv")
    plot_curves(rows, args.out)
    summary = {
        "model": args.model,
        "tokenflow_tokenizer": str(tokenizer_ckpt),
        "dataset": args.dataset,
        "split": args.split,
        "train_count": args.train_count,
        "train_skip_usable": args.train_skip_usable,
        "eval_count": args.eval_count,
        "eval_skip_usable": args.eval_skip_usable if args.eval_skip_usable >= 0 else args.train_skip_usable + args.train_count,
        "selectors": [x.strip() for x in args.selectors.split(",") if x.strip()],
        "budgets": [budget_name(x) for x in parse_budgets(args.budgets)],
        "block_size": args.block_size,
        "label_summary": label_summary(labels),
        "label_mode": args.label_mode,
        "sequential_label_budgets": args.sequential_label_budgets,
        "learned_inference": args.learned_inference,
        "coarse_block_size": args.coarse_block_size,
        "question_feature": args.question_feature,
        "mask_mode": args.mask_mode,
        "semantic_dim": args.semantic_dim,
        "label_max_blocks": args.label_max_blocks,
        "label_candidate_selectors": args.label_candidate_selectors,
        "metrics": metric_dicts,
    }
    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/tokenflow_voi_policy"))
    parser.add_argument("--dataset", default="sionic-ai/textvqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--train-skip-usable", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--eval-skip-usable", type=int, default=-1)
    parser.add_argument("--model", default=TOKENFLOW_MODEL)
    parser.add_argument("--tokenflow-repo-dir", type=Path, default=Path("external/TokenFlow"))
    parser.add_argument("--tokenflow-repo-url", default=TOKENFLOW_REPO)
    parser.add_argument("--tokenflow-tokenizer-repo", default=TOKENFLOW_TOKENIZER_REPO)
    parser.add_argument("--tokenflow-tokenizer-file", default=TOKENFLOW_TOKENIZER_FILE)
    parser.add_argument("--selectors", default="learned,qsim,center,magnitude,random,full,none")
    parser.add_argument("--budgets", default="0,32,128,512,full")
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--label-max-blocks", type=int, default=32)
    parser.add_argument("--label-candidate-selectors", default="qsim,magnitude,center,uniform")
    parser.add_argument("--label-mode", default="single", choices=["single", "sequential"])
    parser.add_argument("--sequential-label-budgets", default="0,32,128,512")
    parser.add_argument("--learned-inference", default="sequential", choices=["static", "sequential", "hierarchical"])
    parser.add_argument("--coarse-block-size", type=int, default=9)
    parser.add_argument("--question-feature", default="lm_hidden", choices=["lm_hidden", "embedding_mean"])
    parser.add_argument("--mask-mode", default="token", choices=["token", "semantic_dense_pixel_sparse"])
    parser.add_argument("--semantic-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-temperature", type=float, default=0.7)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=96)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--log-label-blocks", type=int, default=0)
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
