#!/usr/bin/env python3
"""Question: Does a controlled early prefix handoff change 7B execution?

Hypothesis: Prefix effects should be measured with source, target, semantic
arm, and no-prefix controls separated at the problem level.
Inputs: branch_decomposition labels/source completions and local Qwen base/RL 7B checkpoints.
Outputs: data/rlvr/outputs/e12/{prefixes,continuations,per_problem,contrasts,config}.
Status: formal training-free prefix-handoff runner; no training or annotation.

The runner deliberately keeps discovery rows (sample_index < 32) separate
from continuation rows.  Prefix construction is conservative: any detected
answer leakage skips the prefix and increments a persisted counter.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking.math_transfer import prefix_handoff_protocol as legacy


PAIR = ("math_base_7b", "math_simple_rl_7b")
DEFAULT_BENCHMARKS = ("gsm8k", "math500", "minerva_math")
ARMS = ("P0", "P1", "R", "S", "W")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "analyze"), default="run")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/rlvr/outputs/e1/labels.jsonl")
    parser.add_argument("--models-root", type=Path, default=ROOT / "model")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/rlvr/outputs/e12")
    parser.add_argument("--continuations", type=Path, default=None)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--problems-per-benchmark", type=int, default=50)
    parser.add_argument("--discovery-samples", type=int, default=32)
    parser.add_argument("--n-continuations", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args(argv)


def read_discovery_labels(path: Path, benchmarks: set[str], limit: int, discovery_samples: int) -> list[dict]:
    rows = []
    wanted_models = set(PAIR)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model") not in wanted_models or row.get("benchmark") not in benchmarks:
                continue
            try:
                problem_index = int(str(row["problem_id"]).rsplit(":", 1)[1])
            except (KeyError, ValueError):
                continue
            if problem_index >= limit or int(row.get("sample_index", 0)) >= discovery_samples:
                continue
            rows.append(row)
    return rows


def group_rows(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["benchmark"], row["problem_id"])].append(row)
    return grouped


def sentence_boundaries(text: str) -> list[int]:
    points = {index + 1 for index, char in enumerate(text) if char == "\n"}
    points.update(match.end() for match in re.finditer(r"[.!?](?:\s+|$)", text))
    return sorted(point for point in points if point > 0)


def safe_prefix(text: str, gt: str, arm: str, source_model: str, row: dict, source_answer: str, tokenizer, target_tokens: int | None = None) -> dict | None:
    milestones = legacy.milestone_record(text, gt)
    p1_end = milestones.get("p1_char_end")
    if p1_end is None:
        return None
    if arm == "P1":
        cut = p1_end
    elif arm == "P0":
        before = [point for point in sentence_boundaries(text[:p1_end]) if point < p1_end]
        cut = before[-1] if before else min(p1_end, max(1, len(text) // 3))
    elif arm == "R":
        boundaries = sentence_boundaries(text[: max(p1_end, 1)])
        if target_tokens is None:
            target_tokens = len(tokenizer.encode(text[:p1_end], add_special_tokens=False))
        viable = []
        for point in boundaries:
            count = len(tokenizer.encode(text[:point], add_special_tokens=False))
            if target_tokens * 0.90 <= count <= target_tokens * 1.10:
                viable.append((point, count))
        if not viable:
            return None
        cut = min(viable, key=lambda item: (abs(item[1] - target_tokens), item[0]))[0]
    elif arm == "S":
        prefix = text[:p1_end]
        lines = [line.strip() for line in prefix.splitlines() if ("=" in line or "\\approx" in line) and line.strip()]
        if not lines:
            return None
        state = lines[0]
        if not legacy.prefix_is_safe(state, gt):
            return None
        cut = len(state)
        text = state
    elif arm == "W":
        cut = p1_end
    else:
        raise ValueError(arm)
    prefix = text[:cut].rstrip()
    if not prefix or not legacy.prefix_is_safe(prefix, gt):
        return None
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    branch = legacy.extract_first_calc_branch(prefix)
    if branch == "<no_calc>":
        branch = legacy.NO_VALID_FIRST_CALC
    return {
        "problem_id": row["problem_id"],
        "benchmark": row["benchmark"],
        "source": source_model,
        "arm": arm,
        "prefix_milestone": "p0_or_boundary" if arm == "P0" else "p1" if arm in ("P1", "W") else arm,
        "prefix_text": prefix,
        "prefix_tokens": prefix_tokens,
        "prefix_branch": branch,
        "prefix_leak_checked": True,
        "source_sample_index": int(row["sample_index"]),
        "source_completion_sha1": row.get("completion_sha1"),
        "source_answer": source_answer,
        "ground_truth": gt,
    }


def choose_prefixes(args: argparse.Namespace, labels: list[dict], source_map: dict[tuple[str, int], dict], tokenizer) -> tuple[list[dict], dict]:
    grouped = group_rows(labels)
    selected: list[dict] = []
    counters = defaultdict(int)
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    for benchmark in benchmarks:
        pids = sorted({key[2] for key in grouped if key[1] == benchmark})
        for pid in pids[: args.problems_per_benchmark]:
            for source_model in PAIR:
                rows = grouped.get((source_model, benchmark, pid), [])
                if not rows:
                    counters["missing_source_rows"] += 1
                    continue
                correct = [row for row in rows if bool(row.get("official_correct"))]
                incorrect = [row for row in rows if not bool(row.get("official_correct"))]
                for arm in ARMS:
                    candidates = correct if arm != "W" else incorrect
                    if not candidates:
                        counters[f"missing_{arm}"] += 1
                        continue
                    candidate = min(candidates, key=lambda row: (int(row.get("completion_chars", 0) or 0), int(row["sample_index"])))
                    completion, raw = legacy.source_completion(candidate, source_map)
                    gt = str(raw.get("gt", raw.get("ground_truth", candidate.get("ground_truth", ""))))
                    source_eval = legacy.evaluate_completion(completion, gt)
                    target_tokens = None
                    if arm == "R":
                        target_tokens = len(tokenizer.encode(completion[: legacy.milestone_record(completion, gt).get("p1_char_end") or 0], add_special_tokens=False))
                    prefix = safe_prefix(completion, gt, arm, source_model, candidate, str(source_eval.answer), tokenizer, target_tokens)
                    if prefix is None:
                        counters[f"skipped_{arm}"] += 1
                        continue
                    prefix["question"] = str(raw.get("question", raw.get("problem", "")))
                    prefix["exact_source_answer"] = str(source_eval.answer)
                    prefix["source_official_correct"] = bool(source_eval.is_correct)
                    selected.append(prefix)
    return selected, dict(counters)


def load_model(alias: str, root: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(root / legacy.MODEL_DIRS[alias], local_files_only=True, trust_remote_code=False, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(root / legacy.MODEL_DIRS[alias], torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device)
    model.eval()
    return tokenizer, model


def generate(args: argparse.Namespace, prefixes: list[dict], output_root: Path) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    if args.gpu:
        torch.cuda.set_device(int(str(args.gpu).split(",")[0]))
    targets = list(PAIR)
    all_rows: list[dict] = []
    max_by_benchmark = {"gsm8k": min(args.max_new_tokens, 1024), "math500": args.max_new_tokens, "minerva_math": args.max_new_tokens}
    tasks = []
    for prefix in prefixes:
        for target in targets:
            task = {**prefix, "target": target, "continuation_count": args.n_continuations}
            tasks.append(task)
    if args.smoke:
        tasks = [task for task in tasks if task["arm"] in ("P0", "P1")][: max(1, 6 * len(PAIR) * 2)]
    for target in targets:
        tokenizer, model = load_model(target, args.models_root, "cuda")
        target_tasks = [task for task in tasks if task["target"] == target]
        for task_index, task in enumerate(target_tasks):
            seed = args.seed + task_index + int(hashlib.sha1(task["problem_id"].encode()).hexdigest()[:8], 16)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            prompt_ids = tokenizer.encode(legacy.prompt_for(task["question"]), add_special_tokens=False)
            prefix_ids = tokenizer.encode(task["prefix_text"], add_special_tokens=False)
            input_ids = torch.tensor([prompt_ids + prefix_ids] * task["continuation_count"], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=True, temperature=args.temperature, top_p=args.top_p, max_new_tokens=max_by_benchmark.get(task["benchmark"], args.max_new_tokens), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            for index, row in enumerate(generated[:, input_ids.shape[1]:].detach().cpu().tolist()):
                if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in row:
                    row = row[: row.index(tokenizer.eos_token_id)]
                continuation = tokenizer.decode(row, skip_special_tokens=True).strip()
                full = task["prefix_text"] + continuation
                ev = legacy.evaluate_completion(full, task["ground_truth"])
                branch = ev.first_calc_branch if ev.first_calc_branch != "<no_calc>" else legacy.NO_VALID_FIRST_CALC
                all_rows.append({**{key: value for key, value in task.items() if key not in ("question", "continuation_count")}, "continuation_index": index, "completion": full, "continuation": continuation, "official_correct": bool(ev.is_correct), "parsed": bool(ev.parsed), "format_valid": bool(ev.parsed and re.search(r"(?:\\boxed|####|final answer|answer)", full, re.IGNORECASE)), "observed_first_calc_branch": branch, "same_branch": branch == task["prefix_branch"] if task["prefix_branch"] not in (legacy.NO_VALID_FIRST_CALC, "NO_VALID_FIRST_CALC") else None, "exact_source_answer_match": str(ev.answer) == str(task["exact_source_answer"]), "prompt_tokens": len(prompt_ids), "prefix_tokens": len(prefix_ids), "continuation_tokens": len(row), "target_seed": seed})
            if (task_index + 1) % 25 == 0:
                print(f"[{target}] tasks {task_index + 1}/{len(target_tasks)}", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    with (output_root / "continuations.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return all_rows


def bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze(args: argparse.Namespace, path: Path, output_root: Path) -> None:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["problem_id"], row["source"], row["target"], row["arm"])].append(row)
    per_problem = []
    for key, values in sorted(grouped.items()):
        per_problem.append({"benchmark": key[0], "problem_id": key[1], "source": key[2], "target": key[3], "arm": key[4], "n": len(values), "success_rate": float(np.mean([row["official_correct"] for row in values])), "any_success": float(any(row["official_correct"] for row in values)), "format_valid_rate": float(np.mean([row["format_valid"] for row in values])), "same_branch_rate": float(np.mean([row["same_branch"] for row in values if row["same_branch"] is not None])) if any(row["same_branch"] is not None for row in values) else float("nan"), "mean_continuation_tokens": float(np.mean([row["continuation_tokens"] for row in values]))})
    frame = pd.DataFrame(per_problem)
    frame.to_parquet(output_root / "per_problem.parquet", index=False)
    contrast_rows = []
    for key, group in frame.groupby(["benchmark", "problem_id", "source", "target"], sort=True):
        pivot = group.set_index("arm")
        for left, right in (("P1", "P0"), ("P1", "R"), ("P1", "S"), ("P1", "W")):
            if left not in pivot.index or right not in pivot.index:
                continue
            contrast_rows.append({"benchmark": key[0], "problem_id": key[1], "source": key[2], "target": key[3], "contrast": f"{left}-{right}", "success_contrast": float(pivot.loc[left, "success_rate"] - pivot.loc[right, "success_rate"]), "any_success_contrast": float(pivot.loc[left, "any_success"] - pivot.loc[right, "any_success"])})
    contrasts = pd.DataFrame(contrast_rows)
    if not contrasts.empty:
        records = []
        for key, group in contrasts.groupby(["benchmark", "source", "target", "contrast"], sort=True):
            for metric in ("success_contrast", "any_success_contrast"):
                mean, low, high = bootstrap(group[metric].to_numpy(), 0, args.bootstrap_draws)
                records.append({"benchmark": key[0], "source": key[1], "target": key[2], "contrast": key[3], "metric": metric, "mean": mean, "ci_low": low, "ci_high": high, "n_problems": int(group["problem_id"].nunique()), "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 0, "statistical_unit": "problem"})
        contrasts = pd.DataFrame(records)
    contrasts.to_csv(output_root / "contrasts.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root
    if args.mode == "analyze":
        path = args.continuations or output_root / "continuations.jsonl"
        analyze(args, path, output_root)
        return 0
    benchmarks = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    tokenizer = AutoTokenizer.from_pretrained(args.models_root / legacy.MODEL_DIRS[PAIR[0]], local_files_only=True, trust_remote_code=False, use_fast=False)
    labels = read_discovery_labels(args.labels, benchmarks, args.problems_per_benchmark, args.discovery_samples)
    source_map = legacy.load_source_rows(labels)
    prefixes, counters = choose_prefixes(args, labels, source_map, tokenizer)
    if args.smoke:
        prefixes = [row for row in prefixes if row["arm"] in ("P0", "P1")]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "prefixes.jsonl").open("w", encoding="utf-8") as handle:
        for row in prefixes:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    (output_root / "selection.json").write_text(json.dumps({"benchmarks": sorted(benchmarks), "labels": len(labels), "prefixes": len(prefixes), "counters": counters, "discovery_samples": args.discovery_samples}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = generate(args, prefixes, output_root)
    config = {"experiment_id": "prefix_handoff", "version": "v2", "base_model": PAIR[0], "rl_model": PAIR[1], "benchmarks": sorted(benchmarks), "problems_per_benchmark": args.problems_per_benchmark, "n_discovery_samples": args.discovery_samples, "n_continuations": args.n_continuations, "arms": ARMS if not args.smoke else ["P0", "P1"], "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens, "seed": args.seed, "statistical_unit": "problem", "prefix_leak_checked": True, "prefix_skip_counters": counters, "continuation_rows": len(rows), "smoke": args.smoke}
    (output_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    analyze(args, output_root / "continuations.jsonl", output_root)
    print(json.dumps({"output": str(output_root), "prefixes": len(prefixes), "continuations": len(rows), "counters": counters}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
