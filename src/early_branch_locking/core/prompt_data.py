#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared helpers for Exp V activation steering scripts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set

import torch

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import METRICS_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402

for path in (RAW_DIR, METRICS_DIR):
    path.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_shared import (  # noqa: E402
    build_prompt_text,
    enumerate_solution_set,
    extract_ground_truth,
    get_prompt_content,
)


@dataclass(frozen=True)
class PromptExample:
    pid: int
    sample_id: int
    prompt_text: str
    prompt_ids: List[int]
    numbers: List[int]
    target: int
    feasible_label: str
    solution_set: Set[str]


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"


def single_gpu_id(gpu_arg: str) -> str:
    gpu_ids = [part.strip() for part in gpu_arg.split(",") if part.strip()]
    if len(gpu_ids) != 1:
        raise ValueError("Pass exactly one GPU id per Exp V process.")
    return gpu_ids[0]


def parse_layer_list(text: str) -> List[int]:
    layers = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not layers:
        raise ValueError("steer_layers must contain at least one layer index.")
    return layers


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def build_prompt_data(records: Sequence[dict], tokenizer) -> List[PromptExample]:
    prompt_data: List[PromptExample] = []
    for pid, record in enumerate(records):
        numbers, target, feasible_label = extract_ground_truth(record)
        prompt_text = build_prompt_text(get_prompt_content(record), tokenizer)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        solution_set = enumerate_solution_set(numbers, target) if feasible_label == "yes" else set()
        prompt_data.append(
            PromptExample(
                pid=pid,
                sample_id=int(record.get("sample_id", pid)),
                prompt_text=prompt_text,
                prompt_ids=prompt_ids,
                numbers=numbers,
                target=target,
                feasible_label=feasible_label,
                solution_set=solution_set,
            )
        )
    return prompt_data
