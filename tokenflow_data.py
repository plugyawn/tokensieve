"""Real image-QA loading utilities for TokenFlow selection experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def normalize_answer(text: Any) -> str:
    if isinstance(text, list) and text:
        text = text[0]
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(value)


def _extract_qa(example: dict[str, Any]) -> tuple[str, str] | None:
    if "question" in example:
        answer = example.get("answer", example.get("answers", example.get("label", "")))
        if isinstance(answer, dict):
            answer = answer.get("answer", answer.get("text", ""))
        if isinstance(answer, list) and answer and isinstance(answer[0], dict):
            answer = answer[0].get("answer", answer[0].get("text", ""))
        return str(example["question"]), normalize_answer(answer)

    for key in ("messages", "conversations"):
        messages = example.get(key)
        if not isinstance(messages, list):
            continue
        question = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", message.get("from", ""))).lower()
            content = _message_text(message.get("content", message.get("value", "")))
            if role in {"user", "human"} and question is None:
                question = content.replace("<image>", "").strip()
            elif role in {"assistant", "gpt"} and question:
                return question, normalize_answer(content)
    return None


def _extract_image(example: dict[str, Any]) -> Image.Image | None:
    for key in ("image", "img", "picture"):
        value = example.get(key)
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
    for key in ("image_path", "path", "file_name", "filename"):
        value = example.get(key)
        if isinstance(value, str) and Path(value).exists():
            return Image.open(value).convert("RGB")
    return None


@dataclass
class RealRecord:
    image: Image.Image
    question: str
    answer: str


def load_records(
    dataset_name: str,
    split: str,
    size: int,
    *,
    streaming: bool,
    answer_vocab: dict[str, int] | None = None,
    min_answer_len: int = 1,
    skip_usable: int = 0,
) -> list[RealRecord]:
    from datasets import load_dataset

    records: list[RealRecord] = []
    usable_seen = 0
    for example in load_dataset(dataset_name, split=split, streaming=streaming):
        qa = _extract_qa(example)
        image = _extract_image(example)
        if qa is None or image is None:
            continue
        question, answer = qa
        if len(answer) < min_answer_len:
            continue
        if answer_vocab is not None and answer not in answer_vocab:
            continue
        if usable_seen < skip_usable:
            usable_seen += 1
            continue
        usable_seen += 1
        records.append(RealRecord(image=image, question=question, answer=answer))
        if len(records) >= size:
            break
    return records
