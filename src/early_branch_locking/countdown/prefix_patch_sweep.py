
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""prefix_patch - Temporal residual patch sweep.
Hypothesis: the causal effect of checkpoint state depends jointly on layer block and reasoning position.
Inputs: teacher raw trajectories; source/student model paths; dataset/test.parquet.
Outputs: data/analysis_results/rlvr_passk/metrics/prefix_patch_patch_sweep_summary_sweep_200_v1.csv; data/analysis_results/rlvr_passk/metrics/prefix_patch_patch_sweep_per_problem_sweep_200_v1.parquet
Status: paper-appendix
"""
from __future__ import annotations
"""Temporal residual patch sweep for Countdown."""

import argparse
import gc
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import COUNTDOWN_DATA_ROOT as ANALYSIS_ROOT, METRICS_DIR, TEST_PARQUET  # noqa: E402

METRICS_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.op1_utils import get_layers_container  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_raw_path", type=str, required=True)
    parser.add_argument("--source_model_path", type=str, required=True)
    parser.add_argument("--student_model_path", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--positions", type=str, default="think_end,answer_start,op1_before,after_op1,after_op2")
    parser.add_argument("--layer_blocks", type=str, default="24-27,28-31,32-35")
    parser.add_argument("--sets_path", type=str, default=str(ANALYSIS_ROOT / "metrics" / "branch_set_collection_sets_global_step_50_to_global_step_275_n320.json"))
    parser.add_argument("--set_name", type=str, default="s_loss")
    parser.add_argument("--m_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--tag", type=str, default="temporal_patch_v1")
    parser.add_argument("--save_per_problem", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    positions = [item.strip() for item in args.positions.split(",") if item.strip()]
    layer_blocks = parse_layer_blocks(args.layer_blocks)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    examples, solution_sets = build_prefix_examples(
        Path(args.teacher_raw_path), TEST_PARQUET, args.num_problems, positions, args.sets_path, args.set_name
    )
    if args.max_examples and args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("No prefix examples found for patch sweep.")
    source_model = AutoModelForCausalLM.from_pretrained(args.source_model_path, torch_dtype=dtype, trust_remote_code=True).cuda()
    student_model = AutoModelForCausalLM.from_pretrained(args.student_model_path, torch_dtype=dtype, trust_remote_code=True).cuda()
    source_model.eval()
    student_model.eval()
    source_layers = get_layers_container(source_model)
    student_layers = get_layers_container(student_model)
    summary_rows = []
    per_problem_rows = []
    baseline_examples = dedupe_baseline_examples(examples)
    baseline_rows = run_student_baseline(student_model, tokenizer, baseline_examples, solution_sets, args)
    summary_rows.append(summarize_rows("student_baseline", baseline_rows, {"position": "none", "layer_block": "none"}))
    if args.save_per_problem:
        for pid, row in baseline_rows.items():
            per_problem_rows.append(dict(variant="student_baseline", position="none", layer_block="none", problem_index=pid, **row))

    for position in positions:
        pos_examples = [item for item in examples if item["position"] == position]
        if not pos_examples:
            continue
        baseline = run_variant(student_model, source_model, source_layers, student_layers, tokenizer, pos_examples, solution_sets, None, args)
        summary_rows.append(summarize_rows(f"student_prefix_baseline_{position}", baseline, {"position": position, "layer_block": "none"}))
        if args.save_per_problem:
            for pid, row in baseline.items():
                per_problem_rows.append(dict(variant=f"student_prefix_baseline_{position}", position=position, layer_block="none", problem_index=pid, **row))
        for block in layer_blocks:
            patched = run_variant(student_model, source_model, source_layers, student_layers, tokenizer, pos_examples, solution_sets, block, args)
            label = f"patch_resid_B{block[0]}_{block[1]}_{position}"
            summary_rows.append(summarize_rows(label, patched, {"position": position, "layer_block": f"{block[0]}-{block[1]}"}))
            if args.save_per_problem:
                for pid, row in patched.items():
                    per_problem_rows.append(dict(variant=label, position=position, layer_block=f"{block[0]}-{block[1]}", problem_index=pid, **row))

    summary = pd.DataFrame(summary_rows)
    summary_path = METRICS_DIR / f"prefix_patch_patch_sweep_summary_{args.tag}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[prefix_patch] Saved summary → {summary_path}")
    if args.save_per_problem and per_problem_rows:
        per_path = METRICS_DIR / f"prefix_patch_patch_sweep_per_problem_{args.tag}.parquet"
        pd.DataFrame(per_problem_rows).to_parquet(per_path, index=False)
        print(f"[prefix_patch] Saved per-problem → {per_path}")
    del source_model, student_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dedupe_baseline_examples(examples):
    seen = {}
    for item in examples:
        seen.setdefault(item["problem_index"], item)
    return list(seen.values())


def run_student_baseline(student_model, tokenizer, examples, solution_sets, args):
    completions_by_pid = defaultdict(list)
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        for item in batch:
            prompt_text = (
                tokenizer.apply_chat_template(item["prompt_content"], tokenize=False, add_generation_prompt=True)
                if isinstance(item["prompt_content"], list) else str(item["prompt_content"])
            )
            base_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            generated = sample_repeated_prefix(student_model, tokenizer, base_ids, args.m_samples, args)
            completions_by_pid[item["problem_index"]].extend(generated)
    baseline_examples = [{**item, "prefix_text": ""} for item in examples]
    return evaluate_completions(completions_by_pid, baseline_examples, solution_sets)


def run_variant(student_model, source_model, source_layers, student_layers, tokenizer, examples, solution_sets, block, args):
    completions_by_pid = defaultdict(list)
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        prompt_texts = [tokenizer.apply_chat_template(item["prompt_content"], tokenize=False, add_generation_prompt=True) + item["prefix_text"] if isinstance(item["prompt_content"], list) else str(item["prompt_content"]) + item["prefix_text"] for item in batch]
        enc = tokenizer(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(student_model.device)
        source_outputs = capture_source_layer_outputs(source_layers, block, source_model, enc) if block else {}
        hooks = make_resid_patch_hooks(student_layers, source_outputs, block) if block else []
        with torch.no_grad():
            student_logits = student_model(**enc, output_hidden_states=False, use_cache=False).logits[:, -1, :]
        for hook in hooks:
            hook.remove()
        next_tokens = sample_next_tokens(student_logits, args.m_samples, args.temperature, args.top_p)
        for local_idx, item in enumerate(batch):
            base_ids = enc["input_ids"][local_idx, enc["attention_mask"][local_idx].bool()].tolist()
            batch_input_ids = []
            prompt_lengths = []
            for token_id in next_tokens[local_idx]:
                batch_input_ids.append(base_ids + [int(token_id)])
                prompt_lengths.append(len(base_ids))
            generated = generate_suffixes(student_model, tokenizer, batch_input_ids, prompt_lengths, args)
            completions_by_pid[item["problem_index"]].extend(generated)
    return evaluate_completions(completions_by_pid, examples, solution_sets)


def generate_suffixes(student_model, tokenizer, input_id_rows, prompt_lengths, args):
    generated = []
    for start in range(0, len(input_id_rows), args.sample_batch_size):
        chunk = input_id_rows[start:start + args.sample_batch_size]
        chunk_prompt_lengths = prompt_lengths[start:start + args.sample_batch_size]
        max_len = max(len(row) for row in chunk)
        input_ids = []
        attention_mask = []
        for row in chunk:
            pad = [tokenizer.pad_token_id] * (max_len - len(row))
            input_ids.append(pad + row)
            attention_mask.append([0] * len(pad) + [1] * len(row))
        input_ids = torch.tensor(input_ids, device=student_model.device)
        attention_mask = torch.tensor(attention_mask, device=student_model.device)
        generated.extend(generate_from_inputs(student_model, tokenizer, input_ids, attention_mask, chunk_prompt_lengths, args))
    return generated


def sample_repeated_prefix(student_model, tokenizer, base_ids, n_samples, args):
    prompt_lengths = [len(base_ids)] * n_samples
    return generate_suffixes(student_model, tokenizer, [base_ids] * n_samples, prompt_lengths, args)


def generate_from_inputs(student_model, tokenizer, input_ids, attention_mask, prompt_lengths, args):
    with torch.no_grad():
        outputs = student_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    rows = []
    for row_idx, row in enumerate(outputs):
        prompt_len = int(prompt_lengths[row_idx])
        generated = row[prompt_len:]
        rows.append(tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return rows


# ---- merged patch_sweep_helpers ----
"""Utilities for temporal residual patch sweeps on Countdown."""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from early_branch_locking.core.countdown_utils import evaluate_countdown_expression, parse_countdown_completion
from early_branch_locking.core.countdown_shared import (
    canonicalize_expression,
    entropy_from_counts,
    evaluate_countdown_completion,
    extract_ground_truth,
    get_prompt_content,
    load_parquet_sorted,
    enumerate_solution_set,
)
from early_branch_locking.core.prefix_utils import extract_prefix_text
from early_branch_locking.core.op1_utils import load_problem_ids_from_sets, load_raw_indexed, pick_shortest_success_completion, get_layers_container
from early_branch_locking.core.structure_utils import first_operator


def parse_layer_blocks(raw: str) -> List[Tuple[int, int]]:
    blocks = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        start, end = item.split("-", 1)
        blocks.append((int(start), int(end)))
    return blocks


def build_prefix_examples(
    teacher_raw_path: Path,
    test_parquet: Path,
    num_problems: int,
    positions: List[str],
    sets_path: str,
    set_name: str,
) -> Tuple[List[dict], Dict[int, set[str]]]:
    raw = load_raw_indexed(teacher_raw_path)
    keep_ids = set(load_problem_ids_from_sets(Path(sets_path), set_name) or [])
    records = load_parquet_sorted(test_parquet, n=num_problems, sort_key="sample_id")
    examples = []
    solution_sets = {}
    for pid, rec in enumerate(records):
        if keep_ids and pid not in keep_ids:
            continue
        completion = pick_shortest_success_completion(raw, pid)
        if not completion:
            continue
        numbers, target, feasible_label = extract_ground_truth(rec)
        if feasible_label != "yes":
            continue
        solution_sets[pid] = enumerate_solution_set(numbers, target)
        prompt_content = get_prompt_content(rec)
        for position in positions:
            prefix_text = extract_prefix_text(completion, position)
            if prefix_text:
                examples.append(dict(
                    problem_index=pid,
                    position=position,
                    prompt_content=prompt_content,
                    numbers=numbers,
                    target=target,
                    feasible_label=feasible_label,
                    prefix_text=prefix_text,
                ))
    return examples, solution_sets


def capture_source_layer_outputs(source_layers, block: Tuple[int, int], model, enc) -> Dict[int, torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook

    for layer_idx in range(block[0], block[1] + 1):
        hooks.append(source_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    with torch.no_grad():
        model(**enc, output_hidden_states=False, use_cache=False)
    for hook in hooks:
        hook.remove()
    return captured


def make_resid_patch_hooks(student_layers, source_layer_outputs: Dict[int, torch.Tensor], block: Tuple[int, int]):
    hooks = []
    for layer_idx in range(block[0], block[1] + 1):
        if layer_idx not in source_layer_outputs:
            raise ValueError(f"Missing captured source output for layer {layer_idx}")
        replacement = source_layer_outputs[layer_idx][:, -1, :].detach()

        def hook(module, inputs, output, replacement=replacement):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, -1, :] = replacement
                return (hidden,) + output[1:]
            hidden = output.clone()
            hidden[:, -1, :] = replacement
            return hidden

        hooks.append(student_layers[layer_idx].register_forward_hook(hook))
    return hooks


def sample_next_tokens(logits: torch.Tensor, n_samples: int, temperature: float, top_p: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=n_samples, replacement=True)
    return torch.gather(sorted_idx, -1, sampled).cpu().numpy()


def evaluate_completions(completions_by_pid: Dict[int, List[str]], examples: List[dict], solution_sets: Dict[int, set[str]]) -> Dict[int, dict]:
    example_by_pid = {item["problem_index"]: item for item in examples}
    rows = {}
    for pid, completions in completions_by_pid.items():
        example = example_by_pid[pid]
        solution_counter: Counter[str] = Counter()
        first_ops: Counter[str] = Counter()
        correct = 0
        for completion in completions:
            full_completion = example["prefix_text"] + completion
            result = evaluate_countdown_completion(
                full_completion,
                example["numbers"],
                example["target"],
                example["feasible_label"],
                parse_countdown_completion=parse_countdown_completion,
                evaluate_countdown_expression=evaluate_countdown_expression,
            )
            if result.overall_ok:
                correct += 1
            if result.canonical_expr and result.canonical_expr in solution_sets.get(pid, set()):
                solution_counter[result.canonical_expr] += 1
                first_ops[first_operator(result.canonical_expr)] += 1
        n = len(completions)
        unique_solution = len(solution_counter)
        solution_count = len(solution_sets.get(pid, set()))
        rows[pid] = dict(
            correct_mass=(correct / n) if n else 0.0,
            coverage=(unique_solution / solution_count) if solution_count else 0.0,
            unique_solution=unique_solution,
            top1_sol_mass=(max(solution_counter.values()) / n) if solution_counter else 0.0,
            solution_entropy=entropy_from_counts(solution_counter) if solution_counter else 0.0,
            op1_entropy=entropy_from_counts(first_ops) if first_ops else 0.0,
        )
    return rows


def summarize_rows(variant: str, per_problem: Dict[int, dict], extra: dict) -> dict:
    keys = ("correct_mass", "coverage", "unique_solution", "top1_sol_mass", "solution_entropy", "op1_entropy")
    row = {"variant": variant, "n_problems": len(per_problem), **extra}
    for key in keys:
        row[f"{key}_mean"] = float(np.mean([item[key] for item in per_problem.values()])) if per_problem else 0.0
    return row

if __name__ == "__main__":
    main()
