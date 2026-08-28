#!/usr/bin/env python3
"""math-c2f-run - Standalone coarse-to-fine transfer evaluation.
Hypothesis: a short draft prefix can transfer useful reasoning state to a stronger refine model, improving pass@k without changing the benchmark.
Inputs: dataset/math_eval/<benchmark>/test.jsonl or test.json; draft/refine raw JSONL; local model directories.
Outputs: data/rlvr/outputs/experiments/c2f/metrics/c2f_summary_campaign_math_base_7b_to_math_simple_rl_7b_gsm8k_tokens50.csv; data/rlvr/outputs/experiments/c2f/metrics/c2f_budget_campaign_math_base_7b_to_math_simple_rl_7b_gsm8k_tokens50.csv
Status: paper-appendix
"""
from __future__ import annotations
"""candidate_to_final-math - standalone coarse-to-fine transfer evaluation.

Hypothesis: a short draft prefix can transfer useful reasoning state to a
stronger refine model, improving pass@k without changing the benchmark.
Inputs:  dataset/math_eval/<benchmark>/test.jsonl or test.json, optional raw
         JSONL from an existing draft/refine evaluation, and local model dirs.
Outputs: data/rlvr/outputs/experiments/c2f/{raw,metrics}/c2f_*_<tag>.*
Status:  paper-appendix

This file intentionally contains the small C2F implementation rather than a
runtime dependency on the historical ``rlvr_repro`` package.  Heavy model
libraries are imported only after argument parsing so ``--help`` and registry
checks remain usable on CPU-only analysis machines.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import (  # noqa: E402
    C2F_METRICS_DIR,
    C2F_RAW_DIR,
    MATH_DATASET_DIR as DATA_ROOT,
    MODEL_DIR as MODEL_ROOT,
)

DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64)
PREFIX_MODE_CHOICES = ("tokens", "think_end", "boxed_start", "fraction")
CONCLUSION_RE = re.compile(
    r"(the answer is|final answer|therefore|thus the|hence the|in conclusion)",
    re.IGNORECASE,
)

MODEL_TP = {
    "math_base_7b": 1,
    "math_simple_rl_7b": 1,
    "math_base_14b": 2,
    "math_simple_rl_14b": 2,
    "math_base_32b": 4,
    "math_simple_rl_32b": 4,
    "math_base_qwen_math_7b": 1,
    "math_instruct_qwen_math_7b": 1,
    "math_distill_qwen7b": 1,
    "math_oat_zero_7b": 1,
    "math_base_32b_abel": 4,
    "math_dapo_32b": 4,
    "math_olmo3_sft_7b": 1,
    "math_olmo3_dpo_7b": 1,
    "math_olmo3_rlvr_7b": 1,
}


@dataclass(frozen=True)
class ResolvedModel:
    alias: str
    path: Path
    tensor_parallel_size: int


def rebuild_prefix(draft_completion: str, refine_tokenizer, n_tokens: int) -> str:
    """Reproduce the fixed-token prefix branch used by the legacy campaign."""

    ids = refine_tokenizer.encode(draft_completion, add_special_tokens=False)
    return refine_tokenizer.decode(
        ids[:n_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def find_leak(prefix_text: str, gt_text: str) -> tuple[str, int] | None:
    """Return the earliest registered answer-leak marker in a prefix."""

    candidates: list[tuple[int, int, str]] = []
    boxed = re.search(r"\\boxed", prefix_text)
    if boxed:
        candidates.append((boxed.start(), 0, "boxed"))
    hash_match = prefix_text.find("####")
    if hash_match >= 0:
        candidates.append((hash_match, 1, "gsm_hash"))
    conclusion = CONCLUSION_RE.search(prefix_text)
    if conclusion:
        candidates.append((conclusion.start(), 2, "conclusion"))
    gt_clean = str(gt_text).strip()
    if len(gt_clean) >= 2:
        gt_candidates = [gt_clean]
        without_commas = gt_clean.replace(",", "")
        if without_commas != gt_clean:
            gt_candidates.append(without_commas)
        positions = [(prefix_text.find(value), value) for value in gt_candidates if prefix_text.find(value) >= 0]
        if positions:
            candidates.append((min(positions, key=lambda item: item[0])[0], 3, "gt_string"))
    if not candidates:
        return None
    char_pos, _priority, leak_type = min(candidates, key=lambda item: (item[0], item[1]))
    return leak_type, char_pos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one math C2F transfer experiment")
    parser.add_argument("--draft-model", required=True, help="Model alias or local model path")
    parser.add_argument("--refine-model", required=True, help="Model alias or local model path")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--draft-raw-path", default="")
    parser.add_argument("--refine-raw-path", default="")
    parser.add_argument("--prefix-mode", choices=PREFIX_MODE_CHOICES, default="tokens")
    parser.add_argument("--prefix-tokens", type=int, default=100)
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    parser.add_argument("--max-prefix-tokens", type=int, default=0)
    parser.add_argument("--n-sampling", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draft-temperature", type=float, default=0.6)
    parser.add_argument("--draft-top-p", type=float, default=0.95)
    parser.add_argument("--draft-max-tokens", type=int, default=16000)
    parser.add_argument("--refine-temperature", type=float, default=0.6)
    parser.add_argument("--refine-top-p", type=float, default=0.95)
    parser.add_argument("--refine-max-tokens", type=int, default=16000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-per-problem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--budget-mode", choices=("legacy", "cost-inclusive"), default="legacy", help="Legacy pass@k matching or draft-inclusive empirical token budget.")
    parser.add_argument("--tag", default="")
    return parser.parse_args(argv)


def resolve_model(alias_or_path: str) -> ResolvedModel:
    direct = Path(alias_or_path).expanduser()
    if direct.exists():
        return ResolvedModel(alias=direct.name, path=direct.resolve(), tensor_parallel_size=1)
    path = MODEL_ROOT / alias_or_path
    if not path.exists():
        raise FileNotFoundError(f"Model not found for alias or path: {alias_or_path}")
    return ResolvedModel(
        alias=alias_or_path,
        path=path,
        tensor_parallel_size=MODEL_TP.get(alias_or_path, 1),
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_benchmark_records(benchmark: str, sample_limit: int) -> list[dict]:
    data_dir = DATA_ROOT / benchmark
    jsonl_path = data_dir / "test.jsonl"
    json_path = data_dir / "test.json"
    if jsonl_path.exists():
        records = _read_jsonl(jsonl_path)
    elif json_path.exists():
        records = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Benchmark data not found: {data_dir}")
    if not isinstance(records, list):
        raise TypeError(f"Expected a list of records in {data_dir}")
    return records[:sample_limit] if sample_limit > 0 else records


def record_problem_text(record: dict) -> str:
    parts = []
    if record.get("context") not in (None, ""):
        parts.append(str(record["context"]).strip())
    for key in ("problem", "question"):
        if record.get(key) not in (None, ""):
            parts.append(str(record[key]).strip())
            break
    if not parts:
        raise KeyError(f"No problem/question field in record keys={sorted(record)}")
    return "\n\n".join(parts)


def record_ground_truth(benchmark: str, record: dict) -> str:
    if benchmark == "gsm8k":
        answer = str(record["answer"])
        return answer.split("####", 1)[-1].strip()
    if benchmark == "math500":
        # The benchmark stores the final answer separately from its long
        # worked solution.  Using ``solution`` here silently makes every
        # MATH500 correctness check compare against a proof instead of gold.
        answer = record.get("answer", record.get("solution", ""))
        return str(answer).strip()
    if benchmark == "minerva_math":
        solution = str(record.get("solution", record.get("answer", "")))
        boxed = re.findall(r"\\boxed\s*\{([^{}]*)\}", solution)
        return (boxed[-1] if boxed else solution).strip()
    value = record.get("final_answer", record.get("answer"))
    if isinstance(value, list):
        value = value[0] if value else ""
    text = str(value).strip().strip("$")
    # AMC answers are loaded as floats and AIME answers can contain leading
    # zeroes.  Both representations denote the same integer answer used by
    # the merged evaluation files and by the math grader.
    if benchmark in {"amc23", "aime24"} and re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def build_messages(records: list[dict]) -> list[list[dict]]:
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    return [[{"role": "user", "content": f"{record_problem_text(record)} {instruction}"}] for record in records]


def _raw_problem_index(record: dict) -> int:
    for key in ("problem_index", "index", "idx"):
        if key in record:
            return int(record[key])
    return -1


def load_raw_completions(path: Path, num_problems: int, n_sampling: int) -> dict[int, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Configured raw JSONL does not exist: {path}")
    grouped: dict[int, list[str]] = defaultdict(list)
    for record in _read_jsonl(path):
        pid = _raw_problem_index(record)
        if not 0 <= pid < num_problems or len(grouped[pid]) >= n_sampling:
            continue
        values = []
        for key in ("completion", "response", "continuation", "final_response"):
            if key in record:
                values = [str(record[key])]
                break
        else:
            for key in ("responses", "code"):
                if isinstance(record.get(key), list):
                    values = [str(value) for value in record[key]]
                    break
        grouped[pid].extend(values[: n_sampling - len(grouped[pid])])
    return {pid: grouped.get(pid, [])[:n_sampling] for pid in range(num_problems)}


def _decode_prefix(completion: str, tokenizer, token_count: int) -> str:
    ids = tokenizer.encode(completion, add_special_tokens=False)
    return tokenizer.decode(ids[:token_count], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def extract_prefix(completion: str, mode: str, tokenizer, token_count: int, fraction: float, max_tokens: int) -> str | None:
    if mode == "tokens":
        prefix = _decode_prefix(completion, tokenizer, max(0, token_count))
    elif mode == "fraction":
        ids = tokenizer.encode(completion, add_special_tokens=False)
        prefix = _decode_prefix(completion, tokenizer, max(1, int(len(ids) * fraction)))
    elif mode == "think_end":
        match = re.search(r"</think>", completion, flags=re.IGNORECASE)
        prefix = None if match is None else completion[:match.start()]
    elif mode == "boxed_start":
        match = re.search(r"\\boxed\s*\{", completion)
        prefix = None if match is None else completion[:match.start()]
    else:
        raise ValueError(f"Unknown prefix mode: {mode}")
    if prefix is None:
        return None
    if max_tokens > 0:
        prefix = _decode_prefix(prefix, tokenizer, max_tokens)
    return prefix


def load_tokenizer(model: ResolvedModel):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model.path), trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def prompt_text(tokenizer, message: list[dict]) -> str:
    return tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)


def generate_completions(
    model: ResolvedModel,
    tokenizer,
    messages: list[list[dict]],
    n_sampling: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    args: argparse.Namespace,
) -> dict[int, list[str]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model.path),
        tokenizer=str(model.path),
        tensor_parallel_size=model.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **({"max_model_len": args.max_model_len} if getattr(args, "max_model_len", 0) else {}),
        trust_remote_code=True,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )
    params = SamplingParams(
        n=n_sampling,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    outputs = llm.generate([prompt_text(tokenizer, message) for message in messages], params)
    results = {pid: [item.text or "" for item in output.outputs] for pid, output in enumerate(outputs)}
    if hasattr(llm, "shutdown"):
        llm.shutdown()
    return results


def generate_continuations(
    model: ResolvedModel,
    tokenizer,
    prompt_ids: list[list[int]],
    prefixes: dict[int, list[tuple[str, list[int]]]],
    args: argparse.Namespace,
) -> dict[int, list[str]]:
    from vllm import LLM, SamplingParams

    requests = []
    metadata = []
    for pid, items in sorted(prefixes.items()):
        for prefix_index, (prefix_text_value, prefix_ids) in enumerate(items):
            requests.append({"prompt_token_ids": prompt_ids[pid] + prefix_ids})
            metadata.append((pid, prefix_index, prefix_text_value))
    if not requests:
        return {}
    llm = LLM(
        model=str(model.path),
        tokenizer=str(model.path),
        tensor_parallel_size=model.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **({"max_model_len": args.max_model_len} if getattr(args, "max_model_len", 0) else {}),
        trust_remote_code=True,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )
    params = SamplingParams(
        n=1,
        temperature=args.refine_temperature,
        top_p=args.refine_top_p,
        max_tokens=args.refine_max_tokens,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    results: dict[int, list[str]] = defaultdict(list)
    for start in range(0, len(requests), max(1, args.batch_size)):
        outputs = llm.generate(requests[start:start + args.batch_size], params)
        # C2F raw rows contain the full conditioned completion, matching the
        # historical evaluation files and making old metrics reproducible.
        for offset, output in enumerate(outputs):
            pid, _prefix_index, prefix_text_value = metadata[start + offset]
            for item in output.outputs:
                results[pid].append(prefix_text_value + (item.text or ""))
    if hasattr(llm, "shutdown"):
        llm.shutdown()
    return dict(results)


def normalize_answer(value: object) -> str:
    text = str(value).strip().replace("\\,", "")
    text = text.replace("$", "")
    return re.sub(r"\s+", "", text).lower()


def score_completion(completion: str, ground_truth: str) -> bool:
    boxed = re.findall(r"\\boxed\s*\{([^{}]*)\}", completion)
    predicted = boxed[-1] if boxed else completion.split("\n")[-1]
    try:
        from rllm.rewards.math_utils.utils import grade_answer_verl

        return bool(grade_answer_verl(completion, ground_truth))
    except (ImportError, ModuleNotFoundError):
        return normalize_answer(predicted) == normalize_answer(ground_truth)


def pass_at_k(n: int, correct: int, k: int) -> float:
    if n <= 0 or correct <= 0 or k <= 0:
        return 0.0
    k = min(k, n)
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def evaluate_methods(records: list[dict], completions: dict[str, dict[int, list[str]]], ground_truth: list[str], tokenizer, args, tag: str) -> tuple[list[dict], dict[str, list[dict]]]:
    summary_rows = []
    raw_rows_by_method: dict[str, list[dict]] = {}
    for method, by_pid in completions.items():
        per_problem = []
        raw_rows = []
        for pid, samples in sorted(by_pid.items()):
            correct = 0
            total_tokens = 0
            for sample_index, completion in enumerate(samples):
                tokens = len(tokenizer.encode(completion, add_special_tokens=False))
                ok = score_completion(completion, ground_truth[pid])
                correct += int(ok)
                total_tokens += tokens
                raw_rows.append({
                    "source": method,
                    "problem_index": pid,
                    "sample_index": sample_index,
                    "completion": completion,
                    "completion_tokens": tokens,
                    "correct": ok,
                })
            per_problem.append({
                "problem_index": pid,
                "attempt_count": len(samples),
                "correct_count": correct,
                "correct_mass": correct / len(samples) if samples else 0.0,
                "total_output_tokens": total_tokens,
                "avg_output_tokens": total_tokens / len(samples) if samples else 0.0,
            })
        raw_rows_by_method[method] = raw_rows
        counts = {row["problem_index"]: row["correct_count"] for row in per_problem}
        attempts = {row["problem_index"]: row["attempt_count"] for row in per_problem}
        row = {
            "variant": method,
            "draft_model": args.draft_model,
            "refine_model": args.refine_model,
            "benchmark": args.benchmark,
            "prefix_mode": args.prefix_mode,
            "prefix_tokens": args.prefix_tokens if args.prefix_mode == "tokens" else None,
            "prefix_fraction": args.prefix_fraction if args.prefix_mode == "fraction" else None,
            "n_sampling": args.n_sampling,
            "num_problems": len(records),
            "correct_mass_mean": sum(item["correct_mass"] for item in per_problem) / len(per_problem) if per_problem else 0.0,
            "avg_output_tokens_mean": sum(item["avg_output_tokens"] for item in per_problem) / len(per_problem) if per_problem else 0.0,
            "total_output_tokens": sum(item["total_output_tokens"] for item in per_problem),
            "mean_attempts": sum(attempts.values()) / len(attempts) if attempts else 0.0,
        }
        for k in DEFAULT_KS:
            if k <= args.n_sampling:
                values = [pass_at_k(attempts[pid], count, k) for pid, count in counts.items() if attempts[pid] > 0]
                row[f"pass@{k}"] = sum(values) / len(values) if values else 0.0
        summary_rows.append(row)
        if args.save_raw:
            raw_path = C2F_RAW_DIR / f"{method}_{tag}.jsonl"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("w", encoding="utf-8") as handle:
                for item in raw_rows:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return summary_rows, raw_rows_by_method


def save_results(summary_rows: list[dict], raw_rows_by_method: dict[str, list[dict]], args, tag: str) -> None:
    import pandas as pd

    C2F_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = C2F_METRICS_DIR / f"c2f_summary_{tag}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    if args.save_per_problem:
        rows = []
        for method, raw_rows in raw_rows_by_method.items():
            grouped: dict[int, list[dict]] = defaultdict(list)
            for item in raw_rows:
                grouped[item["problem_index"]].append(item)
            for pid, items in grouped.items():
                rows.append({
                    "variant": method,
                    "problem_index": pid,
                    "attempt_count": len(items),
                    "correct_count": sum(bool(item["correct"]) for item in items),
                    "total_output_tokens": sum(int(item["completion_tokens"]) for item in items),
                })
        pd.DataFrame(rows).to_parquet(C2F_METRICS_DIR / f"c2f_per_problem_{tag}.parquet", index=False)
    if args.budget_mode == "cost-inclusive":
        budget_rows = budget_cost_inclusive_rows(raw_rows_by_method)
        budget_path = C2F_METRICS_DIR / f"c2f_budgetcost_summary_{tag}.csv"
        overview_path = C2F_METRICS_DIR / f"c2f_budgetcost_overview_{tag}.csv"
        frame = pd.DataFrame(budget_rows)
        frame.to_csv(budget_path, index=False)
        frame.groupby("variant", dropna=False).agg(num_problems=("problem_index", "nunique"), mean_n_eff=("n_eff", "mean"), success_at_budget=("success_at_budget", "mean"), correct_rate_at_budget=("correct_rate_at_budget", "mean"), mean_budget=("budget", "mean")).reset_index().to_csv(overview_path, index=False)
    else:
        budget_rows = budget_matched_rows(raw_rows_by_method)
        budget_path = C2F_METRICS_DIR / f"c2f_budget_{tag}.csv"
        pd.DataFrame(budget_rows).to_csv(budget_path, index=False)
    print(f"[C2F] Budget -> {budget_path}")
    print(f"[C2F] Summary -> {summary_path}")


def _token_totals_by_problem(rows: list[dict]) -> dict[int, int]:
    totals: dict[int, int] = defaultdict(int)
    for row in rows:
        totals[int(row["problem_index"])] += int(row.get("completion_tokens", 0))
    return dict(totals)


def _counts_within_budget(rows: list[dict], budget: int) -> tuple[int, int]:
    correct = attempts = used = 0
    for row in sorted(rows, key=lambda item: int(item.get("sample_index", 0))):
        next_used = used + int(row.get("completion_tokens", 0))
        if attempts > 0 and next_used > budget:
            break
        used = next_used
        attempts += 1
        correct += int(bool(row.get("correct", False)))
    return correct, attempts


def budget_matched_rows(raw_rows_by_method: dict[str, list[dict]]) -> list[dict]:
    """Match each method to the smallest per-problem token budget."""
    methods = ("draft", "refine", "c2f")
    rows_by_method = {method: raw_rows_by_method.get(method, []) for method in methods}
    totals = {method: _token_totals_by_problem(rows) for method, rows in rows_by_method.items()}
    problem_ids = set().union(*(values.keys() for values in totals.values()))
    budgets = {}
    for pid in problem_ids:
        positive = [totals[method][pid] for method in methods if totals[method].get(pid, 0) > 0]
        if positive:
            budgets[pid] = min(positive)

    output = []
    for method in methods:
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in rows_by_method[method]:
            grouped[int(row["problem_index"])].append(row)
        correct_counts = {}
        attempt_counts = {}
        for pid, method_rows in grouped.items():
            correct_counts[pid], attempt_counts[pid] = _counts_within_budget(method_rows, budgets.get(pid, 0))
        result = {"variant": method, "budget_mode": "matched"}
        for k in DEFAULT_KS:
            values = [
                pass_at_k(attempt_counts[pid], correct_counts[pid], k)
                for pid in correct_counts
                if attempt_counts.get(pid, 0) > 0
            ]
            result[f"pass@{k}"] = sum(values) / len(values) if values else 0.0
        output.append(result)
    return output


def budget_cost_inclusive_rows(raw_rows_by_method: dict[str, list[dict]]) -> list[dict]:
    """Return per-problem empirical success under draft-inclusive costs.

    The C2F continuation is charged its own generation tokens plus the draft
    generation tokens with the same problem/sample index.  The alignment
    status is retained in every row so a missing sample cannot be hidden.
    """

    methods = ("draft", "refine", "c2f")
    by_method: dict[str, dict[int, list[dict]]] = {method: defaultdict(list) for method in methods}
    for method in methods:
        for row in raw_rows_by_method.get(method, []):
            by_method[method][int(row["problem_index"])].append(row)
    draft_cost = {(int(row["problem_index"]), int(row.get("sample_index", 0))): int(row.get("completion_tokens", 0)) for row in raw_rows_by_method.get("draft", [])}
    problem_ids = sorted(set().union(*(mapping.keys() for mapping in by_method.values())))
    output = []
    for pid in problem_ids:
        costs: dict[str, list[tuple[dict, int]]] = {"draft": [], "refine": [], "c2f": []}
        missing_alignment = 0
        for method in methods:
            for row in sorted(by_method[method].get(pid, []), key=lambda item: int(item.get("sample_index", 0))):
                own = int(row.get("completion_tokens", 0))
                cost = own
                if method == "c2f":
                    key = (pid, int(row.get("sample_index", 0)))
                    if key not in draft_cost:
                        missing_alignment += 1
                    cost += draft_cost.get(key, 0)
                costs[method].append((row, cost))
        totals = {method: sum(cost for _row, cost in values) for method, values in costs.items()}
        positive = [value for value in totals.values() if value > 0]
        budget = min(positive) if positive else 0
        for method in methods:
            used = correct = n_eff = 0
            for row, cost in costs[method]:
                if used + cost > budget:
                    break
                used += cost
                n_eff += 1
                correct += int(bool(row.get("correct", False)))
            output.append({"problem_index": pid, "variant": method, "budget": budget, "method_total_cost": totals[method], "used_cost": used, "n_eff": n_eff, "correct_count_at_budget": correct, "correct_rate_at_budget": correct / n_eff if n_eff else float("nan"), "success_at_budget": float(correct > 0), "draft_alignment_missing": missing_alignment, "budget_mode": "cost-inclusive", "pass_at_k_status": "not_reported"})
    return output


def run(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if args.n_sampling <= 0:
        raise ValueError("--n-sampling must be positive")
    draft_model = resolve_model(args.draft_model)
    refine_model = resolve_model(args.refine_model)
    records = load_benchmark_records(args.benchmark, args.sample_limit)
    ground_truth = [record_ground_truth(args.benchmark, record) for record in records]
    tokenizer = load_tokenizer(refine_model)
    messages = build_messages(records)
    prompt_ids = [tokenizer.encode(prompt_text(tokenizer, message), add_special_tokens=False) for message in messages]

    if args.draft_raw_path:
        draft = load_raw_completions(Path(args.draft_raw_path), len(records), args.n_sampling)
    else:
        draft = generate_completions(draft_model, tokenizer, messages, args.n_sampling, args.draft_temperature, args.draft_top_p, args.draft_max_tokens, args)
    if args.refine_raw_path:
        refine = load_raw_completions(Path(args.refine_raw_path), len(records), args.n_sampling)
    else:
        refine = generate_completions(refine_model, tokenizer, messages, args.n_sampling, args.refine_temperature, args.refine_top_p, args.refine_max_tokens, args)

    prefixes: dict[int, list[tuple[str, list[int]]]] = {}
    for pid, completions in draft.items():
        for completion in completions:
            prefix = extract_prefix(completion, args.prefix_mode, tokenizer, args.prefix_tokens, args.prefix_fraction, args.max_prefix_tokens)
            if prefix is not None:
                prefixes.setdefault(pid, []).append((prefix, tokenizer.encode(prefix, add_special_tokens=False)))
    c2f = generate_continuations(refine_model, tokenizer, prompt_ids, prefixes, args)
    completions = {"draft": draft, "refine": refine, "c2f": c2f}
    tag = args.tag or f"{args.draft_model}_to_{args.refine_model}_{args.benchmark}_{args.prefix_mode}"
    if args.prefix_mode == "tokens":
        tag = f"{tag}{args.prefix_tokens}"
    elif args.prefix_mode == "fraction":
        tag = f"{tag}{args.prefix_fraction}"
    summary_rows, raw_rows = evaluate_methods(records, completions, ground_truth, tokenizer, args, tag)
    save_results(summary_rows, raw_rows, args, tag)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
