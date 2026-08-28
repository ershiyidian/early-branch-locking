"""decoding_controls - Temperature, dead-end, and explicit-diversification controls.
Hypothesis: the observed diversity and continuation effects are not artifacts of temperature,
dead-end formatting, or a failure to explicitly request a different solution method.
Inputs: Countdown raw trajectories; checkpoint model path; dataset/test.parquet; control sampling settings.
Outputs: temperature/dead-end summaries and prompt-diversification raw, per-problem,
summary, and paired-contrast artifacts under data/analysis_results/rlvr_passk/.
Temperature sampling can be issued in chunks so large ``n`` values do not create
an oversized single vLLM request; sample indices are restored before scoring.
Status: paper-appendix
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking.core.countdown_utils import evaluate_countdown_expression, parse_countdown_completion  # noqa: E402
from early_branch_locking.core.countdown_shared import bootstrap_ci_mean, build_prompt_text, canonicalize_expression, entropy_from_counts, enumerate_solution_set, evaluate_countdown_completion, extract_ground_truth, get_prompt_content, load_parquet_sorted, pass_at_k  # noqa: E402
from early_branch_locking._repo import COUNTDOWN_DATA_ROOT as ANALYSIS_ROOT, METRICS_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402

SEED = 42


@dataclass(frozen=True)
class Config:
    model_path: str
    tag: str
    gpu_id: str
    num_problems: int
    n_samples: int
    sample_chunk_size: int
    deadend_samples: int
    temperatures: tuple[float, ...]
    top_p: float
    top_ps: tuple[float, ...]
    min_ps: tuple[float, ...]
    max_new_tokens: int
    max_model_len: int
    dtype: str
    gpu_memory_utilization: float
    controls: tuple[str, ...]
    bootstrap_draws: int
    bootstrap_seed: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tag", default="todo_gap_checks")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--sample_chunk_size", type=int, default=32)
    parser.add_argument("--deadend_samples", type=int, default=16)
    parser.add_argument("--temperatures", default="0.7,1.0,1.3")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_ps", default="", help="Comma-separated top-p grid; empty means only --top_p.")
    parser.add_argument("--min_ps", default="0.0", help="Comma-separated min-p grid; 0.0 disables min-p.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    parser.add_argument(
        "--controls",
        default="temperature,deadend",
        help="Comma-separated controls: temperature,deadend,prompt",
    )
    parser.add_argument("--bootstrap_draws", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=1729)
    args = parser.parse_args()
    temps = tuple(float(item) for item in args.temperatures.split(",") if item.strip())
    if not temps:
        raise ValueError("--temperatures must contain at least one value")
    top_ps = tuple(float(item) for item in args.top_ps.split(",") if item.strip()) or (args.top_p,)
    min_ps = tuple(float(item) for item in args.min_ps.split(",") if item.strip()) or (0.0,)
    controls = tuple(item.strip() for item in args.controls.split(",") if item.strip())
    unknown = set(controls) - {"temperature", "deadend", "prompt"}
    if unknown or not controls:
        raise ValueError(f"Invalid --controls value(s): {sorted(unknown)}")
    if args.sample_chunk_size <= 0:
        raise ValueError("--sample_chunk_size must be positive")
    return Config(
        model_path=args.model_path,
        tag=args.tag,
        gpu_id=args.gpu_id,
        num_problems=args.num_problems,
        n_samples=args.n_samples,
        sample_chunk_size=args.sample_chunk_size,
        deadend_samples=args.deadend_samples,
        temperatures=temps,
        top_p=args.top_p,
        top_ps=top_ps,
        min_ps=min_ps,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        controls=controls,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )


def build_problem_payload(tokenizer, config: Config) -> tuple[list[dict], list[str], list[dict]]:
    records = load_parquet_sorted(TEST_PARQUET, n=config.num_problems, sort_key="sample_id")
    prompts: list[str] = []
    meta: list[dict] = []
    for index, record in enumerate(records):
        prompt_content = get_prompt_content(record)
        numbers, target, feasible_label = extract_ground_truth(record)
        prompts.append(build_prompt_text(prompt_content, tokenizer))
        meta.append({
            "problem_index": index,
            "numbers": numbers,
            "target": target,
            "feasible_label": feasible_label,
            "prompt_content": prompt_content,
        })
    return records, prompts, meta


def build_solution_sets(records: Iterable[dict]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for index, record in enumerate(records):
        numbers, target, _ = extract_ground_truth(record)
        result[index] = enumerate_solution_set(numbers, target)
    return result


def clean_generation(text: str) -> str:
    cleaned = text or ""
    for eos in ("<|endoftext|>", "<|im_end|>"):
        if eos in cleaned:
            cleaned = cleaned.split(eos)[0]
    return cleaned.strip()


def evaluate_samples(outputs, meta: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for problem_output, problem_meta in zip(outputs, meta):
        for sample_index, candidate in enumerate(problem_output.outputs):
            completion = clean_generation(candidate.text)
            ev = evaluate_countdown_completion(
                completion,
                problem_meta["numbers"],
                problem_meta["target"],
                problem_meta["feasible_label"],
                parse_countdown_completion,
                evaluate_countdown_expression,
            )
            rows.append({**problem_meta, "sample_index": sample_index, "completion": completion, "eval": ev})
    return rows


def summarize_temperature(rows: list[dict], solution_sets: dict[int, set[str]], n_samples: int) -> dict:
    by_problem: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[int(row["problem_index"])].append(row)
    pass1: list[float] = []
    pass64: list[float] = []
    coverage: list[float] = []
    entropy: list[float] = []
    for pid, problem_rows in by_problem.items():
        correct = [bool(row["eval"].overall_ok) for row in problem_rows]
        hits = Counter(row["eval"].canonical_expr for row in problem_rows if row["eval"].canonical_expr)
        sol_set = solution_sets[pid]
        filtered_hits = Counter({key: value for key, value in hits.items() if key in sol_set})
        pass1.append(pass_at_k(n_samples, sum(correct), 1))
        pass64.append(pass_at_k(n_samples, sum(correct), min(64, n_samples)))
        coverage.append(len(filtered_hits) / len(sol_set) if sol_set else 0.0)
        entropy.append(entropy_from_counts(filtered_hits))
    return {
        "num_problems": len(by_problem),
        "n_samples": n_samples,
        "pass@1": float(np.mean(pass1)),
        "pass@64": float(np.mean(pass64)),
        "coverage_mean": float(np.mean(coverage)),
        "solution_entropy_mean": float(np.mean(entropy)),
    }


def write_temperature_raw(rows: list[dict], path: Path, decode: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            ev = row["eval"]
            payload = {
                "problem_index": row["problem_index"], "sample_index": row["sample_index"],
                **decode, "completion": row["completion"],
                "overall_ok": ev.overall_ok, "canonical_expr": ev.canonical_expr,
                "parse_status": ev.parse_status, "expr_status": ev.expr_status,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sampling_params_for(
    config: Config, temperature: float, top_p: float, min_p: float, sample_count: int | None = None
):
    """Build sampling parameters, adding min-p only when this vLLM supports it."""
    from vllm import SamplingParams

    kwargs = {
        "n": sample_count if sample_count is not None else config.n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": config.max_new_tokens,
    }
    if min_p > 0.0:
        try:
            return SamplingParams(min_p=min_p, **kwargs), True
        except TypeError:
            return SamplingParams(**kwargs), False
    return SamplingParams(**kwargs), min_p == 0.0


def run_temperature_checks(llm, prompts: list[str], meta: list[dict], solution_sets: dict[int, set[str]], config: Config) -> pd.DataFrame:
    rows: list[dict] = []
    for temperature in config.temperatures:
        for top_p in config.top_ps:
            for min_p in config.min_ps:
                sample_rows: list[dict] = []
                min_p_applied = False
                offset = 0
                while offset < config.n_samples:
                    chunk_size = min(config.sample_chunk_size, config.n_samples - offset)
                    params, min_p_applied = sampling_params_for(
                        config, temperature, top_p, min_p, sample_count=chunk_size
                    )
                    outputs = llm.generate(prompts, params)
                    chunk_rows = evaluate_samples(outputs, meta)
                    for row in chunk_rows:
                        row["sample_index"] = int(row["sample_index"]) + offset
                    sample_rows.extend(chunk_rows)
                    offset += chunk_size
                decode = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "min_p": min_p,
                    "min_p_applied": min_p_applied,
                }
                stem = f"T{temperature:g}_P{top_p:g}_M{min_p:g}"
                write_temperature_raw(
                    sample_rows, RAW_DIR / f"decoding_controls_temperature_raw_{config.tag}_{stem}.jsonl", decode
                )
                summary = summarize_temperature(sample_rows, solution_sets, config.n_samples)
                rows.append({"tag": config.tag, **decode, **summary})
    result = pd.DataFrame(rows)
    result.to_csv(METRICS_DIR / f"decoding_controls_temperature_summary_{config.tag}.csv", index=False)
    return result


def add_diversity_instruction(prompt_content):
    """Add the P2 instruction inside the existing user task without changing its format contract."""
    instruction = (
        "Deliberately seek a valid solution method different from the first approach that comes "
        "to mind. Try to avoid the most obvious arithmetic route."
    )
    needle = "First, reason privately inside <think></think>."
    if not isinstance(prompt_content, list):
        text = str(prompt_content)
        return text.replace(needle, f"{instruction} {needle}") if needle in text else f"{text}\n{instruction}"
    messages = [dict(message) for message in prompt_content]
    for index in range(len(messages) - 1, -1, -1):
        content = str(messages[index].get("content", ""))
        if needle in content:
            messages[index]["content"] = content.replace(needle, f"{instruction} {needle}", 1)
            break
    else:
        messages[-1]["content"] = str(messages[-1].get("content", "")) + "\n" + instruction
    return messages


def prompt_condition_per_problem(
    rows: list[dict], condition: str, solution_sets: dict[int, set[str]], n_samples: int
) -> list[dict]:
    by_problem: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[int(row["problem_index"])].append(row)
    result = []
    for pid, problem_rows in sorted(by_problem.items()):
        correct = [bool(row["eval"].overall_ok) for row in problem_rows]
        hits = Counter(row["eval"].canonical_expr for row in problem_rows if row["eval"].canonical_expr)
        valid_hits = Counter({key: value for key, value in hits.items() if key in solution_sets[pid]})
        result.append(
            {
                "condition": condition,
                "problem_index": pid,
                "n_samples": n_samples,
                "n_correct": sum(correct),
                "pass@1": pass_at_k(n_samples, sum(correct), 1),
                f"pass@{n_samples}": pass_at_k(n_samples, sum(correct), n_samples),
                "coverage": len(valid_hits) / len(solution_sets[pid]) if solution_sets[pid] else 0.0,
                "unique_valid_solutions": len(valid_hits),
                "solution_entropy": entropy_from_counts(valid_hits),
            }
        )
    return result


def summarize_prompt_conditions(per_problem: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "pass@1",
        f"pass@{config.n_samples}",
        "coverage",
        "unique_valid_solutions",
        "solution_entropy",
    ]
    summary_rows = []
    for condition, group in per_problem.groupby("condition", sort=False):
        row = {"tag": config.tag, "condition": condition, "num_problems": len(group), "n_samples": config.n_samples}
        for metric_index, metric in enumerate(metrics):
            mean, lo, hi = bootstrap_ci_mean(
                group[metric].astype(float).tolist(),
                n_boot=config.bootstrap_draws,
                seed=config.bootstrap_seed + metric_index,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        summary_rows.append(row)

    wide = per_problem.pivot(index="problem_index", columns="condition", values=metrics).dropna()
    contrast_rows = []
    for metric_index, metric in enumerate(metrics):
        differences = (wide[(metric, "diversify")] - wide[(metric, "baseline")]).astype(float).tolist()
        mean, lo, hi = bootstrap_ci_mean(
            differences,
            n_boot=config.bootstrap_draws,
            seed=config.bootstrap_seed + metric_index,
        )
        contrast_rows.append(
            {
                "tag": config.tag,
                "contrast": "diversify-baseline",
                "metric": metric,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_problems": len(differences),
                "bootstrap_draws": config.bootstrap_draws,
                "bootstrap_seed": config.bootstrap_seed + metric_index,
                "statistical_unit": "problem",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def run_prompt_diversification(
    llm, tokenizer, prompts: list[str], meta: list[dict], solution_sets: dict[int, set[str]], config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from vllm import SamplingParams

    diversified_prompts = [
        build_prompt_text(add_diversity_instruction(item["prompt_content"]), tokenizer) for item in meta
    ]
    params = SamplingParams(
        n=config.n_samples,
        temperature=0.7,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
    )
    per_problem_rows = []
    raw_path = RAW_DIR / f"decoding_controls_prompt_diversification_raw_{config.tag}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for condition, condition_prompts in (("baseline", prompts), ("diversify", diversified_prompts)):
            sample_rows = evaluate_samples(llm.generate(condition_prompts, params), meta)
            per_problem_rows.extend(
                prompt_condition_per_problem(sample_rows, condition, solution_sets, config.n_samples)
            )
            for row in sample_rows:
                ev = row["eval"]
                handle.write(
                    json.dumps(
                        {
                            "condition": condition,
                            "problem_index": row["problem_index"],
                            "sample_index": row["sample_index"],
                            "completion": row["completion"],
                            "overall_ok": ev.overall_ok,
                            "canonical_expr": ev.canonical_expr,
                            "parse_status": ev.parse_status,
                            "expr_status": ev.expr_status,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    per_problem = pd.DataFrame(per_problem_rows)
    summary, contrasts = summarize_prompt_conditions(per_problem, config)
    per_problem.to_parquet(
        METRICS_DIR / f"decoding_controls_prompt_diversification_per_problem_{config.tag}.parquet", index=False
    )
    summary.to_csv(METRICS_DIR / f"decoding_controls_prompt_diversification_summary_{config.tag}.csv", index=False)
    contrasts.to_csv(METRICS_DIR / f"decoding_controls_prompt_diversification_contrasts_{config.tag}.csv", index=False)
    return summary, contrasts


def eval_simple_expression(numbers: tuple[int, ...], ops: tuple[str, ...]) -> Fraction | None:
    value = Fraction(numbers[0], 1)
    for op, number in zip(ops, numbers[1:]):
        rhs = Fraction(number, 1)
        if op == "+":
            value += rhs
        elif op == "-":
            value -= rhs
        elif op == "*":
            value *= rhs
        elif rhs == 0:
            return None
        else:
            value /= rhs
    return value


def choose_deadend_expression(numbers: list[int], target: int) -> str:
    operators = ("+", "-", "*", "/")
    for perm in itertools.permutations(numbers):
        for ops in itertools.product(operators, repeat=len(numbers) - 1):
            value = eval_simple_expression(perm, ops)
            if value is None or value == Fraction(target, 1):
                continue
            expr = f" {ops[0]} ".join([str(perm[0]), str(perm[1])])
            for op, number in zip(ops[1:], perm[2:]):
                expr = f"{expr} {op} {number}"
            canon, _ = canonicalize_expression(expr)
            if canon is not None:
                return expr
    raise ValueError(f"No dead-end expression found for numbers={numbers}, target={target}")


def build_deadend_prompts(tokenizer, meta: list[dict]) -> tuple[list[str], list[dict]]:
    prompts: list[str] = []
    payload: list[dict] = []
    for item in meta:
        expr = choose_deadend_expression(item["numbers"], item["target"])
        prefix = f"<feasible>yes</feasible>\n<answer>{expr}"
        prompts.append(build_prompt_text(item["prompt_content"], tokenizer) + prefix)
        payload.append({**item, "deadend_prefix": prefix, "deadend_expr": expr})
    return prompts, payload


def run_deadend_check(llm, tokenizer, meta: list[dict], config: Config) -> pd.DataFrame:
    from vllm import SamplingParams

    prompts, payload = build_deadend_prompts(tokenizer, meta)
    params = SamplingParams(n=config.deadend_samples, temperature=0.7, top_p=config.top_p, max_tokens=64)
    outputs = llm.generate(prompts, params)
    rows: list[dict] = []
    for problem_output, item in zip(outputs, payload):
        for sample_index, candidate in enumerate(problem_output.outputs):
            continuation = clean_generation(candidate.text)
            full_text = item["deadend_prefix"] + continuation
            ev = evaluate_countdown_completion(
                full_text,
                item["numbers"],
                item["target"],
                item["feasible_label"],
                parse_countdown_completion,
                evaluate_countdown_expression,
            )
            rows.append({
                "problem_index": item["problem_index"], "sample_index": sample_index,
                "deadend_expr": item["deadend_expr"], "continuation": continuation,
                "strict_success": ev.overall_ok, "expr_ok": ev.expr_ok,
                "closed_answer": "</answer>" in full_text.lower(),
                "parse_status": ev.parse_status, "expr_status": ev.expr_status,
            })
    raw_path = RAW_DIR / f"decoding_controls_deadend_raw_{config.tag}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = pd.DataFrame([{
        "tag": config.tag,
        "num_problems": len(payload),
        "n_samples": config.deadend_samples,
        "strict_success_mean": float(np.mean([row["strict_success"] for row in rows])),
        "expr_ok_mean": float(np.mean([row["expr_ok"] for row in rows])),
        "closed_answer_mean": float(np.mean([row["closed_answer"] for row in rows])),
    }])
    result.to_csv(METRICS_DIR / f"decoding_controls_deadend_summary_{config.tag}.csv", index=False)
    return result


def main() -> None:
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    records, prompts, meta = build_problem_payload(tokenizer, config)
    solution_sets = build_solution_sets(records)
    llm = LLM(
        model=config.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=config.gpu_memory_utilization,
        trust_remote_code=True,
        seed=SEED,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        enforce_eager=True,
    )
    if "temperature" in config.controls:
        temp_summary = run_temperature_checks(llm, prompts, meta, solution_sets, config)
        print(temp_summary.to_string(index=False))
    if "deadend" in config.controls:
        deadend_summary = run_deadend_check(llm, tokenizer, meta, config)
        print(deadend_summary.to_string(index=False))
    if "prompt" in config.controls:
        prompt_summary, prompt_contrasts = run_prompt_diversification(
            llm, tokenizer, prompts, meta, solution_sets, config
        )
        print(prompt_summary.to_string(index=False))
        print(prompt_contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
