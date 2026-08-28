
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""operator_intervention - Residual patching at the first operator.
Hypothesis: restoring early-checkpoint residual state at the first operator can recover a less collapsed operator distribution.
Inputs: teacher prefixes; dataset/test.parquet; base and student model paths.
Outputs: data/analysis_results/rlvr_passk/metrics/operator_intervention_op1_patch_op1_patch_200.csv
Status: paper-main
"""
from __future__ import annotations
"""
operator_intervention_op1_patch_countdown.py

Residual patching at op1 decision:
- Build prefix_before_op1 from teacher success trajectories.
- Clean prompt + prefix -> next-token op distribution.
- Patch student layer output with base hidden state at the last position.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import METRICS_DIR, TEST_PARQUET  # noqa: E402

METRICS_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.op1_utils import (  # noqa: E402
    build_prefix_records,
    build_format_free_inputs,
    load_problem_ids_from_sets,
    get_device,
    get_op_token_ids,
    entropy_from_probs,
    kl_divergence,
    summarise_op_probs,
    get_layers_container,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_raw_path", type=str, required=True)
    p.add_argument("--base_model_path", type=str, default="model/qwen253B")
    p.add_argument("--student_model_path", type=str, required=True)
    p.add_argument("--num_problems", type=int, default=100)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--layers", type=str, required=True, help="Comma-separated layer indices")
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--sets_path", type=str, default="")
    p.add_argument("--set_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def parse_layers(text: str) -> List[int]:
    layers = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    if not layers:
        raise ValueError("No layers specified.")
    return layers


def make_patch_hook(base_last: torch.Tensor):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        hidden = hidden.clone()
        hidden[:, -1, :] = base_last
        if rest is None:
            return hidden
        return (hidden,) + rest

    return hook


def compute_op_probs(logits: torch.Tensor, op_ids: List[int]) -> np.ndarray:
    p = torch.softmax(logits[:, op_ids], dim=-1)
    return p.float().detach().cpu().numpy()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    teacher_raw_path = Path(args.teacher_raw_path)
    prefix_by_pid, _ = build_prefix_records(teacher_raw_path, args.num_problems)
    pids, prompts, _ = build_format_free_inputs(TEST_PARQUET, prefix_by_pid, args.num_problems)
    prefixes = [prefix_by_pid[pid] for pid in pids]

    if args.sets_path and args.set_name:
        set_ids = load_problem_ids_from_sets(Path(args.sets_path), args.set_name)
        if set_ids:
            keep = set(set_ids)
            filtered = [(pid, p, pref) for pid, p, pref in zip(pids, prompts, prefixes) if pid in keep]
            pids, prompts, prefixes = zip(*filtered) if filtered else ([], [], [])
            pids, prompts, prefixes = list(pids), list(prompts), list(prefixes)

    if args.max_examples and args.max_examples > 0 and len(pids) > args.max_examples:
        idx = list(range(len(pids)))
        random.shuffle(idx)
        idx = idx[: args.max_examples]
        pids = [pids[i] for i in idx]
        prompts = [prompts[i] for i in idx]
        prefixes = [prefixes[i] for i in idx]

    if not pids:
        raise ValueError("No valid prefix records found for patching.")

    input_texts = [p + pref for p, pref in zip(prompts, prefixes)]
    layers = parse_layers(args.layers)

    device = get_device(args.gpu_id)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    op_ids = get_op_token_ids(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    base_model.eval()

    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    student_model.eval()
    student_layers = get_layers_container(student_model)

    # aggregate
    base_probs = []
    student_probs = []
    patch_probs_by_layer = {l: [] for l in layers}

    for start in range(0, len(input_texts), args.batch_size):
        batch_texts = input_texts[start : start + args.batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            base_out = base_model(**enc, output_hidden_states=True, use_cache=False)
        base_logits = base_out.logits[:, -1, :]
        base_probs.append(compute_op_probs(base_logits, op_ids))
        base_hidden = base_out.hidden_states

        with torch.no_grad():
            student_out = student_model(**enc, output_hidden_states=False, use_cache=False)
        student_logits = student_out.logits[:, -1, :]
        student_probs.append(compute_op_probs(student_logits, op_ids))

        # patch per layer
        for layer_idx in layers:
            if layer_idx + 1 >= len(base_hidden):
                raise ValueError(f"Layer {layer_idx} out of range for base hidden states.")
            base_last = base_hidden[layer_idx + 1][:, -1, :].detach()
            hook = student_layers[layer_idx].register_forward_hook(
                make_patch_hook(base_last)
            )
            with torch.no_grad():
                patched_out = student_model(**enc, output_hidden_states=False, use_cache=False)
            hook.remove()
            patched_logits = patched_out.logits[:, -1, :]
            patch_probs_by_layer[layer_idx].append(compute_op_probs(patched_logits, op_ids))

    base_probs = np.concatenate(base_probs, axis=0)
    student_probs = np.concatenate(student_probs, axis=0)
    for layer_idx in layers:
        patch_probs_by_layer[layer_idx] = np.concatenate(patch_probs_by_layer[layer_idx], axis=0)

    rows = []
    # baseline row
    base_row = {"checkpoint": "base", "layer": -1}
    base_row.update(summarise_op_probs(base_probs))
    rows.append(base_row)

    student_row = {"checkpoint": "student", "layer": -1}
    student_row.update(summarise_op_probs(student_probs))
    student_row["op1_kl_base_mean"] = float(kl_divergence(base_probs, student_probs).mean())
    rows.append(student_row)

    for layer_idx in layers:
        patched = patch_probs_by_layer[layer_idx]
        row = {"checkpoint": "patched", "layer": layer_idx}
        row.update(summarise_op_probs(patched))
        row["op1_kl_base_mean"] = float(kl_divergence(base_probs, patched).mean())
        rows.append(row)

    df = pd.DataFrame(rows)
    tag = args.tag or Path(args.student_model_path).name
    out_path = METRICS_DIR / f"operator_intervention_op1_patch_{tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved patch metrics -> {out_path}")


if __name__ == "__main__":
    main()
