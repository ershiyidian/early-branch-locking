
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""candidate_to_final - Countdown coarse-to-fine inference and ablation.
Hypothesis: a draft-derived prefix can improve a refine model's pass@k at lower token cost than independent sampling.
Inputs: Countdown dataset and raw trajectories; draft/refine checkpoint paths; prefix and budget settings.
Outputs: data/analysis_results/rlvr_passk/metrics/candidate_to_final_systematic_ablation_summary_candidate_to_final_systematic_v1.csv; data/analysis_results/rlvr_passk/metrics/candidate_to_final_systematic_ablation_budget_candidate_to_final_systematic_v1.csv; data/analysis_results/rlvr_passk/metrics/candidate_to_final_systematic_ablation_manifest_candidate_to_final_systematic_v1.json; data/analysis_results/rlvr_passk/metrics/candidate_to_final_c2f_per_problem_candidate_to_final_systematic_v1_partB_d100_r275_after_op2.parquet
Status: paper-main
"""
from __future__ import annotations
"""
candidate_to_final_c2f_inference_countdown.py

Systematic Coarse-to-Fine inference on Countdown.
- draft-only baseline
- refine-only baseline
- c2f continuation from draft-derived prefixes

Outputs summary / per-problem metrics and optional raw jsonl with token-cost stats.
"""

import argparse
import gc
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import COUNTDOWN_ACTOR_DIR as ACTOR_DIR, METRICS_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402

RAW_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_utils import evaluate_countdown_expression, parse_countdown_completion  # noqa: E402
from early_branch_locking.core.countdown_shared import build_prompt_text, get_prompt_content, load_parquet_sorted  # noqa: E402
# The evaluation/runtime helpers are defined below in this canonical file.

DEFAULT_KS = (1, 4, 16, 64, 128, 256)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft_model_path", type=str, required=True)
    parser.add_argument("--refine_model_path", type=str, required=True)
    parser.add_argument("--draft_raw_path", type=str, default="")
    parser.add_argument("--refine_raw_path", type=str, default="")
    parser.add_argument("--draft_ckpt", type=str, default="")
    parser.add_argument("--refine_ckpt", type=str, default="")
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--draft_temperature", type=float, default=0.7)
    parser.add_argument("--draft_top_p", type=float, default=0.9)
    parser.add_argument("--draft_max_new_tokens", type=int, default=192)
    parser.add_argument("--draft_stop_strings", type=str, default="<feasible>,<answer>")
    parser.add_argument("--refine_temperature", type=float, default=0.7)
    parser.add_argument("--refine_top_p", type=float, default=0.9)
    parser.add_argument("--refine_max_new_tokens", type=int, default=256)
    parser.add_argument("--prefix_mode", type=str, default="think_end", choices=PREFIX_MODE_CHOICES)
    parser.add_argument("--prefix_tokens", type=int, default=64)
    parser.add_argument("--prefix_max_tokens", type=int, default=0)
    parser.add_argument("--use_token_ids", action="store_true", default=True)
    parser.add_argument("--force_answer_tags", action="store_true", default=False)
    parser.add_argument("--answer_tag_prefix", type=str, default="<feasible>yes</feasible>\\n<answer>")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--save_raw", action="store_true", default=True)
    parser.add_argument("--save_baseline_raw", action="store_true", default=False)
    parser.add_argument("--save_per_problem", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def build_summary_rows(method_rows: Dict[str, MethodEvaluation], draft_ckpt: str, refine_ckpt: str, args) -> pd.DataFrame:
    ks = [k for k in DEFAULT_KS if k <= max(args.n_samples, 1)]
    rows = []
    for variant, evaluation in method_rows.items():
        rows.append({
            "variant": variant,
            "draft_ckpt": draft_ckpt,
            "refine_ckpt": refine_ckpt,
            "prefix_mode": args.prefix_mode,
            "prefix_tokens": args.prefix_tokens,
            "n_samples": args.n_samples,
            "num_problems": args.num_problems,
            **evaluation.summary_metrics,
            **passk_from_problem_counts(evaluation.correct_counts, evaluation.attempt_counts, ks),
        })
    return pd.DataFrame(rows)


def build_per_problem_rows(method_rows: Dict[str, MethodEvaluation], draft_ckpt: str, refine_ckpt: str, args) -> pd.DataFrame:
    rows = []
    for variant, evaluation in method_rows.items():
        for row in evaluation.per_problem_rows:
            rows.append({
                "variant": variant,
                "draft_ckpt": draft_ckpt,
                "refine_ckpt": refine_ckpt,
                "prefix_mode": args.prefix_mode,
                "prefix_tokens": args.prefix_tokens,
                **row,
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    from vllm import LLM

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    draft_ckpt = ckpt_name(args.draft_model_path, args.draft_ckpt)
    refine_ckpt = ckpt_name(args.refine_model_path, args.refine_ckpt)
    tag = args.tag or f"{draft_ckpt}_to_{refine_ckpt}_n{args.n_samples}_{args.prefix_mode}"
    tokenizer = AutoTokenizer.from_pretrained(args.refine_model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompt_ids = [build_prompt_token_ids(get_prompt_content(rec), tokenizer) for rec in records]
    prompts = [{"prompt_token_ids": item} for item in prompt_ids] if args.use_token_ids else [build_prompt_text(get_prompt_content(rec), tokenizer) for rec in records]
    params = build_sampling_params(args, tokenizer)
    solution_sets = build_solution_sets(records)

    draft_outputs = load_raw_completions(Path(args.draft_raw_path), args.num_problems, args.n_samples) if args.draft_raw_path else None
    if draft_outputs is None:
        draft_llm = LLM(model=args.draft_model_path, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=args.enforce_eager, seed=args.seed)
        draft_outputs = {pid: outs for pid, outs in enumerate(generate_with_llm(draft_llm, prompts, params["draft"]))}
        if hasattr(draft_llm, "shutdown"):
            draft_llm.shutdown()
        del draft_llm
        gc.collect()
        time.sleep(2)
        cleanup_vllm_engines(args.gpu_id)
        time.sleep(2)
    prefix_ids_by_pid = build_prefix_ids_by_pid(draft_outputs, args, tokenizer)

    refine_outputs = load_raw_completions(Path(args.refine_raw_path), args.num_problems, args.n_samples) if args.refine_raw_path else None
    refine_llm = None
    if refine_outputs is None or prefix_ids_by_pid:
        refine_llm = LLM(model=args.refine_model_path, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=args.enforce_eager, seed=args.seed)
    if refine_outputs is None:
        refine_outputs = {pid: outs for pid, outs in enumerate(generate_with_llm(refine_llm, prompts, params["baseline"]))}

    c2f_prompts = []
    c2f_pid_index = []
    c2f_source_indices_by_pid: Dict[int, List[int]] = defaultdict(list)
    for pid, prefixes in prefix_ids_by_pid.items():
        for source_index, prefix_ids in enumerate(prefixes):
            c2f_prompts.append({"prompt_token_ids": prompt_ids[pid] + prefix_ids})
            c2f_pid_index.append(pid)
            c2f_source_indices_by_pid[pid].append(source_index)
    c2f_outputs = []
    for start in range(0, len(c2f_prompts), args.batch_size):
        c2f_outputs.extend(generate_with_llm(refine_llm, c2f_prompts[start:start + args.batch_size], params["refine"]))
    c2f_by_pid: Dict[int, List[str]] = defaultdict(list)
    for index, outputs in enumerate(c2f_outputs):
        if outputs:
            c2f_by_pid[c2f_pid_index[index]].append(outputs[0])
    if hasattr(refine_llm, "shutdown"):
        refine_llm.shutdown()

    method_rows = {
        "draft": evaluate_method_outputs(records, draft_outputs, "draft", tokenizer, parse_countdown_completion, evaluate_countdown_expression, solution_sets, raw_output_path("draft", draft_ckpt, refine_ckpt, args.n_samples, tag, RAW_DIR) if args.save_baseline_raw else None),
        "refine": evaluate_method_outputs(records, refine_outputs, "refine", tokenizer, parse_countdown_completion, evaluate_countdown_expression, solution_sets, raw_output_path("refine", draft_ckpt, refine_ckpt, args.n_samples, tag, RAW_DIR) if args.save_baseline_raw else None),
        "c2f": evaluate_method_outputs(records, c2f_by_pid, "c2f", tokenizer, parse_countdown_completion, evaluate_countdown_expression, solution_sets, raw_output_path("c2f", draft_ckpt, refine_ckpt, args.n_samples, tag, RAW_DIR) if args.save_raw else None, c2f_source_indices_by_pid),
    }
    summary_df = build_summary_rows(method_rows, draft_ckpt, refine_ckpt, args)
    summary_path = METRICS_DIR / f"candidate_to_final_c2f_summary_{tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")
    if args.save_per_problem:
        per_problem_df = build_per_problem_rows(method_rows, draft_ckpt, refine_ckpt, args)
        per_problem_path = METRICS_DIR / f"candidate_to_final_c2f_per_problem_{tag}.parquet"
        per_problem_df.to_parquet(per_problem_path, index=False)
        print(f"Saved per-problem -> {per_problem_path}")


# ---- merged pair_ablation mode ----
"""Run all ExpI prefix configs for one draft/refine pair with one refine-model load."""

import argparse
import gc
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

METRICS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_utils import evaluate_countdown_expression, parse_countdown_completion  # noqa: E402
from early_branch_locking.core.countdown_shared import build_prompt_text, get_prompt_content, load_parquet_sorted  # noqa: E402


def parse_args_pair_ablation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft_model_path", type=str, required=True)
    parser.add_argument("--refine_model_path", type=str, required=True)
    parser.add_argument("--draft_raw_path", type=str, required=True)
    parser.add_argument("--refine_raw_path", type=str, required=True)
    parser.add_argument("--draft_ckpt", type=str, required=True)
    parser.add_argument("--refine_ckpt", type=str, required=True)
    parser.add_argument("--prefix_configs", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draft_temperature", type=float, default=0.7)
    parser.add_argument("--draft_top_p", type=float, default=0.9)
    parser.add_argument("--draft_max_new_tokens", type=int, default=192)
    parser.add_argument("--draft_stop_strings", type=str, default="<feasible>,<answer>")
    parser.add_argument("--refine_temperature", type=float, default=0.7)
    parser.add_argument("--refine_top_p", type=float, default=0.9)
    parser.add_argument("--refine_max_new_tokens", type=int, default=256)
    parser.add_argument("--prefix_max_tokens", type=int, default=0)
    parser.add_argument("--use_token_ids", action="store_true", default=True)
    parser.add_argument("--force_answer_tags", action="store_true", default=False)
    parser.add_argument("--answer_tag_prefix", type=str, default="<feasible>yes</feasible>\\n<answer>")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--save_raw", action="store_true", default=True)
    parser.add_argument("--save_per_problem", action="store_true", default=True)
    parser.add_argument("--tag_prefix", type=str, required=True)
    return parser.parse_args()


def parse_prefix_configs(raw: str) -> List[Tuple[str, int, str]]:
    configs = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if item.startswith("tokens"):
            token_count = int(item.replace("tokens", ""))
            configs.append(("tokens", token_count, f"tokens{token_count}"))
        else:
            configs.append((item, 0, item))
    return configs


def config_args(base_args, prefix_mode: str, prefix_tokens: int):
    clone = argparse.Namespace(**vars(base_args))
    clone.prefix_mode = prefix_mode
    clone.prefix_tokens = prefix_tokens
    return clone


def run_pair_ablation():
    from vllm import LLM

    args = parse_args_pair_ablation()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    tokenizer = AutoTokenizer.from_pretrained(args.refine_model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    prompt_ids = [build_prompt_token_ids(get_prompt_content(rec), tokenizer) for rec in records]
    prompts = [{"prompt_token_ids": item} for item in prompt_ids] if args.use_token_ids else [build_prompt_text(get_prompt_content(rec), tokenizer) for rec in records]
    params = build_sampling_params(args, tokenizer)
    solution_sets = build_solution_sets(records)
    draft_outputs = load_raw_completions(Path(args.draft_raw_path), args.num_problems, args.n_samples)
    refine_outputs = load_raw_completions(Path(args.refine_raw_path), args.num_problems, args.n_samples)
    draft_eval = evaluate_method_outputs(records, draft_outputs, "draft", tokenizer, parse_countdown_completion, evaluate_countdown_expression, solution_sets, None)
    refine_eval = evaluate_method_outputs(records, refine_outputs, "refine", tokenizer, parse_countdown_completion, evaluate_countdown_expression, solution_sets, None)
    refine_llm = LLM(model=args.refine_model_path, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=args.enforce_eager, seed=args.seed)

    for prefix_mode, prefix_tokens, prefix_label in parse_prefix_configs(args.prefix_configs):
        cfg = config_args(args, prefix_mode, prefix_tokens)
        prefix_ids_by_pid = build_prefix_ids_by_pid(draft_outputs, cfg, tokenizer)
        c2f_prompts = []
        c2f_pid_index = []
        c2f_source_indices_by_pid = defaultdict(list)
        for pid, prefixes in prefix_ids_by_pid.items():
            for source_index, prefix_ids in enumerate(prefixes):
                c2f_prompts.append({"prompt_token_ids": prompt_ids[pid] + prefix_ids})
                c2f_pid_index.append(pid)
                c2f_source_indices_by_pid[pid].append(source_index)
        c2f_outputs = []
        for start in range(0, len(c2f_prompts), args.batch_size):
            c2f_outputs.extend(generate_with_llm(refine_llm, c2f_prompts[start:start + args.batch_size], params["refine"]))
        c2f_by_pid = defaultdict(list)
        for index, outputs in enumerate(c2f_outputs):
            if outputs:
                c2f_by_pid[c2f_pid_index[index]].append(outputs[0])
        tag = f"{args.tag_prefix}_{prefix_label}"
        method_rows = {
            "draft": draft_eval,
            "refine": refine_eval,
            "c2f": evaluate_method_outputs(records, c2f_by_pid, "c2f", tokenizer, parse_countdown_completion, evaluate_countdown_expression, solution_sets, raw_output_path("c2f", args.draft_ckpt, args.refine_ckpt, args.n_samples, tag, RAW_DIR) if args.save_raw else None, c2f_source_indices_by_pid),
        }
        summary_df = build_summary_rows(method_rows, args.draft_ckpt, args.refine_ckpt, cfg)
        summary_path = METRICS_DIR / f"candidate_to_final_c2f_summary_{tag}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary -> {summary_path}")
        if args.save_per_problem:
            per_problem_df = build_per_problem_rows(method_rows, args.draft_ckpt, args.refine_ckpt, cfg)
            per_problem_path = METRICS_DIR / f"candidate_to_final_c2f_per_problem_{tag}.parquet"
            per_problem_df.to_parquet(per_problem_path, index=False)
            print(f"Saved per-problem -> {per_problem_path}")
    if hasattr(refine_llm, "shutdown"):
        refine_llm.shutdown()
    del refine_llm
    gc.collect()
    time.sleep(2)
    cleanup_vllm_engines(args.gpu_id)


# ---- merged evaluation mode ----
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from early_branch_locking.core.countdown_shared import (
    bootstrap_ci_mean,
    canonicalize_expression,
    entropy_from_counts,
    evaluate_countdown_completion,
    extract_ground_truth,
    pass_at_k,
    enumerate_solution_set as _enumerate_solution_set_shared,
)


@dataclass(frozen=True)
class MethodEvaluation:
    correct_counts: Dict[int, int]
    attempt_counts: Dict[int, int]
    per_problem_rows: List[dict]
    summary_metrics: Dict[str, float]
    raw_rows: List[dict]


def build_solution_sets(records: Sequence[dict]) -> Dict[int, set[str]]:
    solution_sets: Dict[int, set[str]] = {}
    for pid, rec in enumerate(records):
        numbers, target, feasible_label = extract_ground_truth(rec)
        solution_sets[pid] = _enumerate_solution_set_shared(numbers, target) if feasible_label == "yes" else set()
    return solution_sets


def enumerate_solution_set(numbers: List[int], target: int) -> set[str]:
    """Backward-compatible wrapper."""
    return _enumerate_solution_set_shared(numbers, target)


def evaluate_method_outputs(
    records: Sequence[dict],
    completions_by_pid: Dict[int, List[str]],
    label: str,
    tokenizer,
    parse_countdown_completion: Callable,
    evaluate_countdown_expression: Callable,
    solution_sets: Dict[int, set[str]],
    raw_path: Optional[Path],
    sample_indices_by_pid: Optional[Dict[int, List[int]]] = None,
) -> MethodEvaluation:
    rows: List[dict] = []
    problem_rows: List[dict] = []
    correct_counts: Dict[int, int] = {}
    attempt_counts: Dict[int, int] = {}
    for pid, completions in completions_by_pid.items():
        row, raw_rows = evaluate_problem_outputs(
            pid,
            records[pid],
            completions,
            label,
            tokenizer,
            parse_countdown_completion,
            evaluate_countdown_expression,
            solution_sets.get(pid, set()),
            sample_indices_by_pid.get(pid) if sample_indices_by_pid else None,
        )
        problem_rows.append(row)
        rows.extend(raw_rows)
        correct_counts[pid] = int(row["correct_count"])
        attempt_counts[pid] = int(row["attempt_count"])
    if raw_path is not None:
        write_jsonl(raw_path, rows)
    return MethodEvaluation(
        correct_counts=correct_counts,
        attempt_counts=attempt_counts,
        per_problem_rows=problem_rows,
        summary_metrics=summarize_problem_rows(problem_rows),
        raw_rows=rows,
    )


def evaluate_problem_outputs(
    pid: int,
    record: dict,
    completions: Sequence[str],
    label: str,
    tokenizer,
    parse_countdown_completion: Callable,
    evaluate_countdown_expression: Callable,
    solution_set: set[str],
    sample_indices: Optional[List[int]],
) -> Tuple[dict, List[dict]]:
    numbers, target, feasible_label = extract_ground_truth(record)
    correct_count = 0
    sol_counter: Counter[str] = Counter()
    total_tokens = 0
    raw_rows: List[dict] = []
    for local_index, completion in enumerate(completions):
        sample_index = sample_indices[local_index] if sample_indices else local_index
        eval_res = evaluate_countdown_completion(
            completion,
            numbers,
            target,
            feasible_label,
            parse_countdown_completion=parse_countdown_completion,
            evaluate_countdown_expression=evaluate_countdown_expression,
        )
        completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        total_tokens += completion_tokens
        if eval_res.overall_ok:
            correct_count += 1
        if eval_res.canonical_expr and eval_res.canonical_expr in solution_set:
            sol_counter[eval_res.canonical_expr] += 1
        raw_rows.append(build_raw_row(label, pid, sample_index, completion, completion_tokens, eval_res))
    row = build_problem_row(pid, correct_count, len(completions), sol_counter, solution_set, total_tokens)
    return row, raw_rows


def build_raw_row(label: str, pid: int, sample_index: int, completion: str, completion_tokens: int, eval_res) -> dict:
    return {
        "source": label,
        "problem_index": pid,
        "sample_index": sample_index,
        "completion": completion,
        "completion_tokens": completion_tokens,
        "feasible_pred": eval_res.feasible_pred,
        "feasible_ok": eval_res.feasible_ok,
        "expr_ok": eval_res.expr_ok,
        "overall_ok": eval_res.overall_ok,
        "answer_label": eval_res.answer_label,
        "trace_label": eval_res.trace_label,
        "expr_status": eval_res.expr_status,
        "canonical_expr": eval_res.canonical_expr,
        "opseq_label": eval_res.opseq_label,
        "parse_status": eval_res.parse_status,
        "has_feasible_tag": eval_res.has_feasible_tag,
        "has_answer_tag": eval_res.has_answer_tag,
        "tag_order_ok": eval_res.tag_order_ok,
    }


def build_problem_row(
    pid: int,
    correct_count: int,
    attempt_count: int,
    sol_counter: Counter[str],
    solution_set: set[str],
    total_tokens: int,
) -> dict:
    solution_count = len(solution_set)
    unique_solution = len(sol_counter)
    return {
        "problem_index": pid,
        "attempt_count": attempt_count,
        "correct_count": correct_count,
        "correct_mass": correct_count / attempt_count if attempt_count else 0.0,
        "unique_solution": unique_solution,
        "coverage": unique_solution / solution_count if solution_count else 0.0,
        "solution_entropy": entropy_from_counts(sol_counter) if sol_counter else 0.0,
        "solution_count": solution_count,
        "total_output_tokens": total_tokens,
        "avg_output_tokens": total_tokens / attempt_count if attempt_count else 0.0,
    }


def summarize_problem_rows(problem_rows: Sequence[dict]) -> Dict[str, float]:
    keys = ("correct_mass", "coverage", "unique_solution", "solution_entropy", "avg_output_tokens")
    if not problem_rows:
        out = {f"{key}_mean": 0.0 for key in keys}
        out["total_output_tokens"] = 0.0
        out["mean_attempts"] = 0.0
        out["mean_output_tokens_per_completion"] = 0.0
        return out
    out = {f"{key}_mean": float(np.mean([row[key] for row in problem_rows])) for key in keys}
    total_tokens = sum(int(row["total_output_tokens"]) for row in problem_rows)
    total_attempts = sum(int(row["attempt_count"]) for row in problem_rows)
    out["total_output_tokens"] = float(total_tokens)
    out["mean_attempts"] = float(np.mean([row["attempt_count"] for row in problem_rows]))
    out["mean_output_tokens_per_completion"] = total_tokens / total_attempts if total_attempts else 0.0
    return out


def passk_from_problem_counts(
    correct_counts: Dict[int, int],
    attempt_counts: Dict[int, int],
    ks: Iterable[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        estimates = []
        for pid, correct in correct_counts.items():
            attempts = attempt_counts.get(pid, 0)
            if attempts > 0:
                estimates.append(pass_at_k(attempts, correct, min(k, attempts)))
        metrics[f"pass@{k}"] = float(np.mean(estimates)) if estimates else 0.0
        if estimates:
            _, ci_lo, ci_hi = bootstrap_ci_mean(estimates)
        else:
            ci_lo = 0.0
            ci_hi = 0.0
        metrics[f"pass@{k}_ci_low"] = ci_lo
        metrics[f"pass@{k}_ci_high"] = ci_hi
    return metrics


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---- merged runtime mode ----
import os
import signal
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from early_branch_locking.core.countdown_shared import build_prompt_text
from early_branch_locking.core.prefix_utils import extract_prefix_text as extract_structural_prefix

PREFIX_MODE_CHOICES = (
    "think_end",
    "tokens",
    "answer_start",
    "op1_before",
    "after_op1",
    "after_op2",
)


def ckpt_name(path: str, fallback: str) -> str:
    if fallback:
        return fallback
    model_name = Path(path).name
    return model_name if model_name.startswith("global_step_") else model_name


def build_prompt_token_ids(prompt_content, tokenizer) -> List[int]:
    prompt_text = build_prompt_text(prompt_content, tokenizer)
    return tokenizer.encode(prompt_text, add_special_tokens=False)


def load_raw_completions(raw_path: Path, num_problems: int, n_samples: int) -> Dict[int, List[str]]:
    by_pid: Dict[int, List[str]] = defaultdict(list)
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = __import__("json").loads(line)
            pid = int(record.get("problem_index", -1))
            if 0 <= pid < num_problems:
                by_pid[pid].append(record.get("completion", ""))
    return {pid: by_pid.get(pid, [])[:n_samples] for pid in range(num_problems)}


def raw_output_path(kind: str, draft_ckpt: str, refine_ckpt: str, n_samples: int, tag: str, raw_dir: Path) -> Path:
    suffix = f"_{tag}" if tag else ""
    if kind == "draft":
        return raw_dir / f"countdown_raw_c2f_draft_{draft_ckpt}_n{n_samples}{suffix}.jsonl"
    if kind == "refine":
        return raw_dir / f"countdown_raw_c2f_refine_{refine_ckpt}_n{n_samples}{suffix}.jsonl"
    return raw_dir / f"countdown_raw_c2f_{draft_ckpt}_to_{refine_ckpt}_n{n_samples}{suffix}.jsonl"


def build_sampling_params(args, tokenizer) -> Dict[str, SamplingParams]:
    from vllm import SamplingParams

    eos_id = tokenizer.eos_token_id
    draft_stop = [item.strip() for item in args.draft_stop_strings.split(",") if item.strip()]
    return {
        "draft": SamplingParams(
            n=args.n_samples,
            temperature=args.draft_temperature,
            top_p=args.draft_top_p,
            max_tokens=args.draft_max_new_tokens,
            stop=draft_stop or None,
            stop_token_ids=[eos_id] if eos_id is not None else None,
            seed=args.seed,
        ),
        "refine": SamplingParams(
            n=1,
            temperature=args.refine_temperature,
            top_p=args.refine_top_p,
            max_tokens=args.refine_max_new_tokens,
            stop_token_ids=[eos_id] if eos_id is not None else None,
            seed=args.seed,
        ),
        "baseline": SamplingParams(
            n=args.n_samples,
            temperature=args.refine_temperature,
            top_p=args.refine_top_p,
            max_tokens=args.refine_max_new_tokens,
            stop_token_ids=[eos_id] if eos_id is not None else None,
            seed=args.seed,
        ),
    }


def generate_with_llm(llm, prompts, sampling_params: SamplingParams) -> List[List[str]]:
    outputs = llm.generate(prompts, sampling_params)
    return [[seq.text for seq in output.outputs] for output in outputs]


def build_prefix_text_from_completion(completion: str, args, tokenizer) -> str | None:
    if args.prefix_mode == "tokens":
        if args.prefix_tokens <= 0:
            return ""
        ids = tokenizer.encode(completion, add_special_tokens=False)
        prefix_ids = ids[: args.prefix_tokens]
        return tokenizer.decode(prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return extract_structural_prefix(completion, args.prefix_mode)


def build_prefix_ids_by_pid(completions_by_pid: Dict[int, List[str]], args, tokenizer) -> Dict[int, List[List[int]]]:
    prefix_ids_by_pid: Dict[int, List[List[int]]] = defaultdict(list)
    tag_ids = tokenizer.encode(args.answer_tag_prefix, add_special_tokens=False)
    for pid, completions in completions_by_pid.items():
        for completion in completions:
            prefix_text = build_prefix_text_from_completion(completion, args, tokenizer)
            if prefix_text is None:
                continue
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            if args.prefix_max_tokens and args.prefix_max_tokens > 0:
                prefix_ids = prefix_ids[: args.prefix_max_tokens]
            if args.force_answer_tags:
                prefix_ids = prefix_ids + tag_ids
            prefix_ids_by_pid[pid].append(prefix_ids)
    return prefix_ids_by_pid


def cleanup_vllm_engines(gpu_id: str) -> None:
    try:
        gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        gpu_uuid = {row.split(",")[0].strip(): row.split(",")[1].strip() for row in gpu_rows.strip().splitlines()}.get(str(gpu_id))
        if not gpu_uuid:
            return
        proc_rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
            text=True,
        )
    except Exception:
        return
    for row in proc_rows.strip().splitlines():
        gpu, pid, name = [part.strip() for part in row.split(",")[:3]]
        if gpu == gpu_uuid and "VLLM::EngineCore" in name:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                continue


# ---- merged budget mode ----
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class RawSample:
    problem_index: int
    sample_index: int
    overall_ok: bool
    completion_tokens: int


@dataclass(frozen=True)
class BudgetSummary:
    variant: str
    budget_tokens_total: int
    budget_tokens_mean: float
    metrics: Dict[str, float]


def load_raw_samples(raw_path: Path, tokenizer, n_samples: int) -> Dict[int, List[RawSample]]:
    grouped: Dict[int, List[RawSample]] = defaultdict(list)
    sample_index_by_pid: Dict[int, int] = defaultdict(int)
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = int(record["problem_index"])
            sample_index = int(record.get("sample_index", sample_index_by_pid[pid]))
            sample_index_by_pid[pid] = sample_index + 1
            if sample_index >= n_samples:
                continue
            completion = record.get("completion", "")
            grouped[pid].append(
                RawSample(
                    problem_index=pid,
                    sample_index=sample_index,
                    overall_ok=bool(record.get("overall_ok", False)),
                    completion_tokens=int(record.get("completion_tokens", len(tokenizer.encode(completion, add_special_tokens=False)))),
                )
            )
    return {pid: sorted(rows, key=lambda item: item.sample_index)[:n_samples] for pid, rows in grouped.items()}


def combine_c2f_tokens(
    draft_samples: Dict[int, List[RawSample]],
    c2f_samples: Dict[int, List[RawSample]],
) -> Dict[int, List[RawSample]]:
    combined: Dict[int, List[RawSample]] = {}
    for pid, rows in c2f_samples.items():
        draft_by_index = {row.sample_index: row for row in draft_samples.get(pid, [])}
        merged = []
        for row in rows:
            draft_row = draft_by_index.get(row.sample_index)
            if draft_row is not None:
                merged.append(
                    RawSample(
                        problem_index=pid,
                        sample_index=row.sample_index,
                        overall_ok=row.overall_ok,
                        completion_tokens=row.completion_tokens + draft_row.completion_tokens,
                    )
                )
        combined[pid] = merged
    return combined


def per_problem_budgets(method_rows: Sequence[Dict[int, List[RawSample]]]) -> Dict[int, int]:
    budgets: Dict[int, int] = {}
    all_pids = set().union(*(rows.keys() for rows in method_rows))
    for pid in all_pids:
        totals = [sum(sample.completion_tokens for sample in rows.get(pid, [])) for rows in method_rows]
        positives = [total for total in totals if total > 0]
        budgets[pid] = min(positives) if positives else 0
    return budgets


def matched_budget_counts(
    samples_by_pid: Dict[int, List[RawSample]],
    budget_by_pid: Dict[int, int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    correct_counts: Dict[int, int] = {}
    attempt_counts: Dict[int, int] = {}
    for pid, samples in samples_by_pid.items():
        correct, attempts = consume_budget(samples, budget_by_pid.get(pid, 0))
        correct_counts[pid] = correct
        attempt_counts[pid] = attempts
    return correct_counts, attempt_counts


def consume_budget(samples: Sequence[RawSample], budget: int) -> Tuple[int, int]:
    correct = 0
    attempts = 0
    used = 0
    for sample in samples:
        next_used = used + sample.completion_tokens
        if attempts > 0 and next_used > budget:
            break
        used = next_used
        attempts += 1
        if sample.overall_ok:
            correct += 1
    return correct, attempts


def compute_budget_summary(
    variant: str,
    per_problem_budget: Dict[int, int],
    correct_counts: Dict[int, int],
    attempt_counts: Dict[int, int],
    metric_fn,
    ks: Iterable[int],
) -> BudgetSummary:
    metrics = metric_fn(correct_counts, attempt_counts, ks)
    total_budget = sum(per_problem_budget.values())
    mean_budget = total_budget / len(per_problem_budget) if per_problem_budget else 0.0
    return BudgetSummary(
        variant=variant,
        budget_tokens_total=total_budget,
        budget_tokens_mean=mean_budget,
        metrics=metrics,
    )


# ---- merged systematic mode ----
"""
candidate_to_final_systematic_ablation_driver.py

Run ExpI systematic ablations and compute token-budget-matched summaries.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from transformers import AutoTokenizer

DEFAULT_PREFIX_CONFIGS = ("think_end", "tokens32", "tokens64", "tokens96", "tokens128", "after_op1", "after_op2")
PASS_KS = (1, 4, 16, 64, 128, 256)

def parse_args_systematic():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_dir", type=str, default=str(ACTOR_DIR))
    parser.add_argument("--draft_steps", type=str, default="50,75,100")
    parser.add_argument("--refine_steps", type=str, default="150,200,275")
    parser.add_argument("--prefix_configs", type=str, default=",".join(DEFAULT_PREFIX_CONFIGS))
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--tag", type=str, default="systematic_ablation")
    parser.add_argument("--draft_temperature", type=float, default=0.7)
    parser.add_argument("--draft_top_p", type=float, default=0.9)
    parser.add_argument("--draft_max_new_tokens", type=int, default=192)
    parser.add_argument("--refine_temperature", type=float, default=0.7)
    parser.add_argument("--refine_top_p", type=float, default=0.9)
    parser.add_argument("--refine_max_new_tokens", type=int, default=256)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    return parser.parse_args()


def parse_steps(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_prefix_configs(raw: str) -> List[Tuple[str, int]]:
    configs: List[Tuple[str, int]] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if item.startswith("tokens"):
            configs.append(("tokens", int(item.replace("tokens", ""))))
        else:
            configs.append((item, 0))
    return configs


def checkpoint_path(actor_dir: Path, step: int) -> Path:
    return actor_dir / f"global_step_{step}"


def baseline_raw_path(step: int, n_samples: int) -> Path:
    candidates = [
        RAW_DIR / f"countdown_raw_global_step_{step}_n{n_samples}.jsonl",
        RAW_DIR / f"countdown_raw_global_step_{step}_n320.jsonl",
        RAW_DIR / f"countdown_raw_global_step_{step}_n256.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No raw file found for global_step_{step} with n={n_samples}")


def run_tag(base_tag: str, draft_step: int, refine_step: int, prefix_mode: str, prefix_tokens: int) -> str:
    if prefix_mode == "tokens":
        return f"{base_tag}_d{draft_step}_r{refine_step}_{prefix_mode}{prefix_tokens}"
    return f"{base_tag}_d{draft_step}_r{refine_step}_{prefix_mode}"


def run_config(args, draft_step: int, refine_step: int, prefix_mode: str, prefix_tokens: int) -> str:
    tag = run_tag(args.tag, draft_step, refine_step, prefix_mode, prefix_tokens)
    old_argv = list(sys.argv)
    argv = [old_argv[0],
        "--draft_model_path", str(checkpoint_path(Path(args.actor_dir), draft_step)),
        "--refine_model_path", str(checkpoint_path(Path(args.actor_dir), refine_step)),
        "--draft_ckpt", f"global_step_{draft_step}",
        "--refine_ckpt", f"global_step_{refine_step}",
        "--draft_raw_path", str(baseline_raw_path(draft_step, args.n_samples)),
        "--refine_raw_path", str(baseline_raw_path(refine_step, args.n_samples)),
        "--num_problems", str(args.num_problems), "--n_samples", str(args.n_samples),
        "--gpu_id", args.gpu_id, "--prefix_mode", prefix_mode, "--tag", tag,
        "--save_raw", "--save_per_problem",
        "--draft_temperature", str(args.draft_temperature),
        "--draft_top_p", str(args.draft_top_p),
        "--draft_max_new_tokens", str(args.draft_max_new_tokens),
        "--refine_temperature", str(args.refine_temperature),
        "--refine_top_p", str(args.refine_top_p),
        "--refine_max_new_tokens", str(args.refine_max_new_tokens)]
    if prefix_mode == "tokens":
        argv.extend(["--prefix_tokens", str(prefix_tokens)])
    if args.enforce_eager:
        argv.append("--enforce_eager")
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old_argv
    return tag


def run_pair(args, draft_step: int, refine_step: int, prefix_configs: List[Tuple[str, int]]) -> List[str]:
    tag_prefix = f"{args.tag}_d{draft_step}_r{refine_step}"
    raw_prefix = ",".join(f"{mode}{tokens}" if mode == "tokens" else mode for mode, tokens in prefix_configs)
    old_argv = list(sys.argv)
    argv = [old_argv[0],
        "--draft_model_path", str(checkpoint_path(Path(args.actor_dir), draft_step)),
        "--refine_model_path", str(checkpoint_path(Path(args.actor_dir), refine_step)),
        "--draft_ckpt", f"global_step_{draft_step}",
        "--refine_ckpt", f"global_step_{refine_step}",
        "--draft_raw_path", str(baseline_raw_path(draft_step, args.n_samples)),
        "--refine_raw_path", str(baseline_raw_path(refine_step, args.n_samples)),
        "--prefix_configs", raw_prefix, "--num_problems", str(args.num_problems),
        "--n_samples", str(args.n_samples), "--gpu_id", args.gpu_id,
        "--tag_prefix", tag_prefix, "--draft_temperature", str(args.draft_temperature),
        "--draft_top_p", str(args.draft_top_p), "--draft_max_new_tokens", str(args.draft_max_new_tokens),
        "--refine_temperature", str(args.refine_temperature),
        "--refine_top_p", str(args.refine_top_p), "--refine_max_new_tokens", str(args.refine_max_new_tokens)]
    if args.enforce_eager:
        argv.append("--enforce_eager")
    try:
        sys.argv = argv
        run_pair_ablation()
    finally:
        sys.argv = old_argv
    return [run_tag(args.tag, draft_step, refine_step, mode, tokens) for mode, tokens in prefix_configs]


def budget_rows_for_tag(tag: str, n_samples: int, tokenizer) -> List[dict]:
    summary_path = METRICS_DIR / f"candidate_to_final_c2f_summary_{tag}.csv"
    summary_df = pd.read_csv(summary_path)
    draft_ckpt = summary_df["draft_ckpt"].iloc[0]
    refine_ckpt = summary_df["refine_ckpt"].iloc[0]
    draft_samples = load_raw_samples(baseline_raw_path(int(draft_ckpt.split("_")[-1]), n_samples), tokenizer, n_samples)
    refine_samples = load_raw_samples(baseline_raw_path(int(refine_ckpt.split("_")[-1]), n_samples), tokenizer, n_samples)
    c2f_path = RAW_DIR / f"countdown_raw_c2f_{draft_ckpt}_to_{refine_ckpt}_n{n_samples}_{tag}.jsonl"
    c2f_samples = load_raw_samples(c2f_path, tokenizer, n_samples)
    c2f_total_samples = combine_c2f_tokens(draft_samples, c2f_samples)
    budget_by_pid = per_problem_budgets((draft_samples, refine_samples, c2f_total_samples))
    rows = []
    for variant, samples in (("draft", draft_samples), ("refine", refine_samples), ("c2f", c2f_total_samples)):
        correct_counts, attempt_counts = matched_budget_counts(samples, budget_by_pid)
        summary = compute_budget_summary(
            variant,
            budget_by_pid,
            correct_counts,
            attempt_counts,
            passk_from_problem_counts,
            PASS_KS,
        )
        rows.append({
            "tag": tag,
            "draft_ckpt": draft_ckpt,
            "refine_ckpt": refine_ckpt,
            "variant": variant,
            "budget_tokens_total": summary.budget_tokens_total,
            "budget_tokens_mean": summary.budget_tokens_mean,
            **summary.metrics,
        })
    return rows


def merge_summary_files(tags: Iterable[str]) -> pd.DataFrame:
    frames = []
    for tag in tags:
        frame = pd.read_csv(METRICS_DIR / f"candidate_to_final_c2f_summary_{tag}.csv")
        frame.insert(0, "tag", tag)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_systematic():
    args = parse_args_systematic()
    actor_dir = Path(args.actor_dir)
    prefix_configs = parse_prefix_configs(args.prefix_configs)
    tags: List[str] = []
    for draft_step in parse_steps(args.draft_steps):
        for refine_step in parse_steps(args.refine_steps):
            tags.extend(run_pair(args, draft_step, refine_step, prefix_configs))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path(actor_dir, parse_steps(args.refine_steps)[0])), trust_remote_code=True)
    summary_df = merge_summary_files(tags)
    summary_out = METRICS_DIR / f"candidate_to_final_systematic_ablation_summary_{args.tag}.csv"
    summary_df.to_csv(summary_out, index=False)
    budget_rows = []
    for tag in tags:
        budget_rows.extend(budget_rows_for_tag(tag, args.n_samples, tokenizer))
    budget_out = METRICS_DIR / f"candidate_to_final_systematic_ablation_budget_{args.tag}.csv"
    pd.DataFrame(budget_rows).to_csv(budget_out, index=False)
    manifest = {
        "tags": tags,
        "summary": str(summary_out),
        "budget_summary": str(budget_out),
    }
    manifest_path = METRICS_DIR / f"candidate_to_final_systematic_ablation_manifest_{args.tag}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved systematic summary -> {summary_out}")
    print(f"Saved budget summary -> {budget_out}")
    print(f"Saved manifest -> {manifest_path}")

def _run_selected():
    selector = None
    selector_index = None
    flag = "--mode"
    for index, argument in enumerate(sys.argv):
        if argument == flag:
            selector_index = index
            selector = sys.argv[index + 1] if index + 1 < len(sys.argv) else "single"
            break
        if argument.startswith(flag + "="):
            selector_index = index
            selector = argument.split("=", 1)[1]
            break
    if selector_index is not None:
        del sys.argv[selector_index:selector_index + 2]
        if selector == "pair_sweep":
            return run_pair_ablation()
        if selector == "systematic":
            return run_systematic()
        if selector == "budget":
            return run_systematic()
        if selector not in {"single", "run"}:
            raise ValueError(f"Unknown --mode: {selector}")
    return main()

if __name__ == "__main__":
    _run_selected()
