#!/usr/bin/env python3
"""prefix_handoff/semantic_boundary_analysis: semantic prefix handoff with matched counterfactual controls.

Hypothesis: An early, complete strategy prefix from the other model may help
execute the branch, with a potentially asymmetric effect across source/target.
Inputs: branch_decomposition labels and their preserved source completions, plus a compatible
base/RL checkpoint pair.
Procedure: use Beta(1,1) posterior Monte Carlo for lost/shared/gained strata;
select answer-safe P0/P1/P2 prefixes; generate fixed-temperature continuations
for each source/target condition with token-exact prefix ids.
Metrics: recovery lift, empirical any-success, execution retention, branch
adherence/switching, format validity, length, and source/target interaction.
Outputs: data/rlvr/outputs/{e3,e4}/<pair>/{prefixes,continuations}.jsonl,
selection.csv, per_problem.parquet, summary.csv, and config.json.
Statistical unit: problem -> prefix -> continuation.
Known limitations: this entry point uses existing 64-sample discovery data for
selection; a confirmation free-sampling pass is required before causal claims.
Status: formal training-free handoff runner; no model is trained.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking.core.math_trace_utils import evaluate_completion, extract_first_calc_branch, normalize_answer
from early_branch_locking.core.branch_protocol import NO_VALID_FIRST_CALC, milestone_record

MODEL_ROOT = ROOT / "model"
E1_ROOT = ROOT / "data" / "rlvr" / "outputs" / "e1"
PAIR_CONFIG = {
    "qwen_7b_base_rl": ("math_base_7b", "math_simple_rl_7b"),
    "qwen_14b_base_rl": ("math_base_14b", "math_simple_rl_14b"),
    "olmo_sft_rlvr": ("math_olmo3_sft_7b", "math_olmo3_rlvr_7b"),
}
MODEL_DIRS = {
    "math_base_7b": "math_base_7b",
    "math_simple_rl_7b": "math_simple_rl_7b",
    "math_base_14b": "math_base_14b",
    "math_simple_rl_14b": "math_simple_rl_14b",
    "math_olmo3_sft_7b": "Olmo-3-7B-Instruct-SFT",
    "math_olmo3_rlvr_7b": "Olmo-3-7B-Instruct-RLVR",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("prefix_handoff", "semantic_boundary_analysis"), required=True)
    parser.add_argument("--pair", choices=sorted(PAIR_CONFIG), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--e1-root", type=Path, default=E1_ROOT)
    parser.add_argument("--models-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "rlvr" / "outputs")
    parser.add_argument("--benchmarks", default="gsm8k,math500,minerva_math,olympiadbench,amc23,aime24")
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--posterior-draws", type=int, default=20000)
    parser.add_argument("--max-problems", type=int, default=100)
    parser.add_argument("--max-prefixes-per-problem", type=int, default=3)
    parser.add_argument("--n-continuations", type=int, default=8)
    parser.add_argument("--confirmation-free-samples", type=int, default=0, help="Independent no-prefix samples per selected problem and target")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args(argv)


def prompt_for(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalized_prefix(text: str) -> str:
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_labels(root: Path, base: str, rl: str, benchmarks: Sequence[str]) -> list[dict]:
    wanted = {base, rl}
    output = []
    with (root / "labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["model"] in wanted and row["benchmark"] in benchmarks:
                output.append(row)
    return output


def load_source_rows(labels: Sequence[dict]) -> dict[tuple[str, int], dict]:
    needed: dict[str, set[int]] = defaultdict(set)
    for row in labels:
        needed[row["source_file"]].add(int(row["source_row"]))
    output = {}
    for source_file, source_rows in needed.items():
        with (ROOT / source_file).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in source_rows and line.strip():
                    output[(source_file, index)] = json.loads(line)
    return output


def posterior_row(base_rows: Sequence[dict], rl_rows: Sequence[dict], draws: int, delta: float, seed: int) -> dict:
    base_n, base_c = len(base_rows), sum(bool(row["official_correct"]) for row in base_rows)
    rl_n, rl_c = len(rl_rows), sum(bool(row["official_correct"]) for row in rl_rows)
    rng = np.random.default_rng(seed)
    base_samples = rng.beta(base_c + 1, base_n - base_c + 1, size=draws)
    rl_samples = rng.beta(rl_c + 1, rl_n - rl_c + 1, size=draws)
    return {
        "base_n": base_n,
        "base_correct": base_c,
        "rl_n": rl_n,
        "rl_correct": rl_c,
        "base_posterior_mean": float(base_samples.mean()),
        "rl_posterior_mean": float(rl_samples.mean()),
        "posterior_delta_mean": float((base_samples - rl_samples).mean()),
        "p_base_minus_rl_gt_delta": float(np.mean(base_samples - rl_samples > delta)),
        "p_rl_minus_base_gt_delta": float(np.mean(rl_samples - base_samples > delta)),
        "p_both_above_005": float(np.mean((base_samples > 0.05) & (rl_samples > 0.05))),
    }


def select_strata(labels: Sequence[dict], base: str, rl: str, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    by_model_problem: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in labels:
        by_model_problem[(row["model"], row["benchmark"], row["problem_id"])].append(row)
    posteriors = []
    for benchmark in args.benchmarks.split(","):
        pids = sorted({key[2] for key in by_model_problem if key[1] == benchmark and key[0] in (base, rl)})
        for pid in pids:
            item = posterior_row(by_model_problem.get((base, benchmark, pid), []), by_model_problem.get((rl, benchmark, pid), []), args.posterior_draws, args.delta, args.seed + int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16))
            item.update({"benchmark": benchmark, "problem_id": pid})
            if item["p_base_minus_rl_gt_delta"] > 0.95:
                item["stratum"] = "lost"
            elif item["p_rl_minus_base_gt_delta"] > 0.95:
                item["stratum"] = "gained"
            elif item["p_both_above_005"] > 0.95:
                item["stratum"] = "shared_capable"
            else:
                item["stratum"] = "uncertain"
            posteriors.append(item)
    if args.experiment == "prefix_handoff":
        candidates = [row for row in posteriors if row["stratum"] == "lost"]
    else:
        candidates = [row for row in posteriors if row["stratum"] in ("shared_capable", "gained")]
    candidates.sort(key=lambda row: (-abs(row["posterior_delta_mean"]), row["benchmark"], row["problem_id"]))
    limit = 3 if args.smoke else args.max_problems
    selected = candidates[:limit]
    return selected, posteriors


def source_completion(row: dict, source: dict[tuple[str, int], dict]) -> tuple[str, dict]:
    raw = source[(row["source_file"], int(row["source_row"]))]
    completions = raw.get("code") or raw.get("completions") or raw.get("responses") or []
    return str(completions[int(row["sample_index"])]), raw


def prefix_is_safe(prefix: str, gt: str) -> bool:
    if not prefix.strip():
        return False
    gt_norm = normalize_answer(gt).replace(" ", "")
    prefix_norm = normalize_answer(prefix).replace(" ", "")
    if gt_norm and gt_norm not in ("<unparsed>", "nan") and gt_norm in prefix_norm:
        return False
    if "\\boxed" in prefix.lower() or "####" in prefix:
        return False
    return True


def make_prefix(row: dict, source: dict[tuple[str, int], dict], milestone: str, style: str = "raw") -> dict | None:
    completion, raw = source_completion(row, source)
    gt = str(raw.get("gt", raw.get("ground_truth", "")))
    text = normalized_prefix(completion) if style == "minimal_normalized" else completion
    milestones = milestone_record(text, gt)
    char_end = milestones.get(f"{milestone}_char_end")
    if char_end is None:
        return None
    prefix = text[:char_end]
    branch = extract_first_calc_branch(text)
    if branch == "<no_calc>":
        branch = NO_VALID_FIRST_CALC
    if not prefix_is_safe(prefix, gt):
        return None
    return {
        "source_model": row["model"],
        "source_file": row["source_file"],
        "source_row": row["source_row"],
        "source_sample_index": row["sample_index"],
        "benchmark": row["benchmark"],
        "problem_id": row["problem_id"],
        "ground_truth": gt,
        "prefix_milestone": milestone,
        "prefix_style": style,
        "prefix_text": prefix,
        "prefix_chars": len(prefix),
        "prefix_words": len(prefix.split()),
        "prefix_branch": branch,
        "prefix_p1_char_end": milestones.get("p1_char_end"),
        "prefix_p2_char_end": milestones.get("p2_char_end"),
        "prefix_answer_leakage": milestones.get("answer_leakage_before_p2", False),
    }


def choose_source_prefixes(problem: dict, rows: Sequence[dict], source: dict, model: str, max_prefixes: int) -> list[dict]:
    candidates = [row for row in rows if row["model"] == model and row["official_correct"]]
    by_branch: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_branch[row["first_calc_branch"]].append(row)
    chosen_rows = []
    for branch in sorted(by_branch):
        chosen_rows.append(sorted(by_branch[branch], key=lambda item: (item["sample_index"], item["completion_sha1"]))[0])
    prefixes = []
    for row in chosen_rows:
        for milestone in ("p0", "p1", "p2"):
            prefix = make_prefix(row, source, milestone)
            if prefix is not None:
                prefixes.append(prefix)
    prefixes.sort(key=lambda item: (item["prefix_milestone"], item["prefix_branch"], item["source_sample_index"]))
    return prefixes[:max_prefixes]


def choose_wrong_prefix(problem_rows: Sequence[dict], source: dict, correct_branch: str, target_model: str) -> dict | None:
    rows = [row for row in problem_rows if not row["official_correct"] and row["model"] == target_model and row["first_calc_branch"] != correct_branch]
    for row in sorted(rows, key=lambda item: (item["sample_index"], item["completion_sha1"])):
        for milestone in ("p1", "p0", "p2"):
            prefix = make_prefix(row, source, milestone)
            if prefix is not None:
                return prefix
    return None


def build_tasks(args: argparse.Namespace, selected: Sequence[dict], labels: Sequence[dict], source: dict, base: str, rl: str) -> tuple[list[dict], list[dict]]:
    by_problem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in labels:
        by_problem[(row["benchmark"], row["problem_id"])].append(row)
    tasks, prefixes = [], []
    for item in selected:
        key = (item["benchmark"], item["problem_id"])
        problem_rows = by_problem[key]
        if args.experiment == "prefix_handoff":
            bases = choose_source_prefixes(item, problem_rows, source, base, args.max_prefixes_per_problem)
            if not bases:
                continue
            for prefix in bases:
                prefix = {**prefix, "stratum": "lost"}
                prefixes.append(prefix)
                for target, condition in ((rl, "base_prefix_to_rl"), (base, "base_prefix_to_base")):
                    tasks.append({**prefix, "target_model": target, "condition": condition, "data_role": "discovery"})
                normalized = make_prefix(next(row for row in problem_rows if row["source_file"] == prefix["source_file"] and row["source_row"] == prefix["source_row"] and row["sample_index"] == prefix["source_sample_index"]), source, prefix["prefix_milestone"], "minimal_normalized")
                if normalized is not None and extract_first_calc_branch(normalized["prefix_text"]) == prefix["prefix_branch"]:
                    normalized = {**normalized, "stratum": "lost"}
                    prefixes.append(normalized)
                    tasks.append({**normalized, "target_model": rl, "condition": "normalized_base_prefix_to_rl", "data_role": "discovery"})
                wrong = choose_wrong_prefix(problem_rows, source, prefix["prefix_branch"], base)
                if wrong is not None:
                    wrong = {**wrong, "stratum": "lost"}
                    prefixes.append(wrong)
                    tasks.append({**wrong, "target_model": rl, "condition": "wrong_branch_to_rl", "data_role": "discovery"})
            neutral = {**bases[0], "prefix_text": "Let's work through the problem carefully.\n", "prefix_chars": 43, "prefix_words": 7, "prefix_branch": "NEUTRAL", "prefix_milestone": "none", "prefix_style": "neutral"}
            prefixes.append(neutral)
            tasks.append({**neutral, "target_model": rl, "condition": "neutral_prefix_to_rl", "data_role": "discovery"})
            tasks.append({**neutral, "prefix_text": "", "prefix_chars": 0, "prefix_words": 0, "prefix_branch": "NO_PREFIX", "prefix_milestone": "none", "prefix_style": "none", "target_model": rl, "condition": "no_prefix_rl", "data_role": "discovery"})
        else:
            stratum = item["stratum"]
            base_prefixes = choose_source_prefixes(item, problem_rows, source, base, args.max_prefixes_per_problem)
            rl_prefixes = choose_source_prefixes(item, problem_rows, source, rl, args.max_prefixes_per_problem)
            for prefix in base_prefixes:
                prefix = {**prefix, "stratum": stratum}
                prefixes.append(prefix)
                for target, condition in ((base, "base_prefix_to_base"), (rl, "base_prefix_to_rl")):
                    tasks.append({**prefix, "target_model": target, "condition": condition, "data_role": "discovery"})
            for prefix in rl_prefixes:
                prefix = {**prefix, "stratum": stratum}
                prefixes.append(prefix)
                for target, condition in ((rl, "rl_prefix_to_rl"), (base, "rl_prefix_to_base")):
                    tasks.append({**prefix, "target_model": target, "condition": condition, "data_role": "discovery"})
    # No-prefix controls are generated once per problem/target and matched later.
    seen = set()
    for task in list(tasks):
        key = (task["benchmark"], task["problem_id"], task["target_model"])
        if key in seen:
            continue
        seen.add(key)
        tasks.append({**task, "prefix_text": "", "prefix_chars": 0, "prefix_words": 0, "prefix_branch": "NO_PREFIX", "prefix_milestone": "none", "prefix_style": "none", "condition": f"no_prefix_{task['target_model']}", "data_role": "discovery"})
    return tasks, prefixes


def build_confirmation_tasks(selected: Sequence[dict], labels: Sequence[dict], source: dict, base: str, rl: str, count: int) -> list[dict]:
    if count <= 0:
        return []
    by_problem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in labels:
        by_problem[(row["benchmark"], row["problem_id"])].append(row)
    tasks = []
    for item in selected:
        rows = by_problem.get((item["benchmark"], item["problem_id"]), [])
        if not rows:
            continue
        source_row = sorted(rows, key=lambda row: (row["model"], row["sample_index"]))[0]
        raw = source[(source_row["source_file"], int(source_row["source_row"]))]
        common = {
            "source_model": "NONE",
            "source_file": source_row["source_file"],
            "source_row": source_row["source_row"],
            "source_sample_index": source_row["sample_index"],
            "benchmark": item["benchmark"],
            "problem_id": item["problem_id"],
            "ground_truth": str(raw.get("gt", raw.get("ground_truth", ""))),
            "question": str(raw.get("question", raw.get("problem", ""))),
            "prefix_text": "",
            "prefix_chars": 0,
            "prefix_words": 0,
            "prefix_branch": "NO_PREFIX",
            "prefix_milestone": "none",
            "prefix_style": "none",
            "stratum": item["stratum"],
            "data_role": "confirmation",
        }
        for target in (base, rl):
            tasks.append({**common, "target_model": target, "condition": f"confirmation_free_sampling_{target}"})
    return tasks


def load_model(alias: str, root: Path, device: str):
    path = root / MODEL_DIRS[alias]
    # The published SimpleRL tokenizer.json is valid JSON but incompatible with
    # the installed fast-tokenizer backend; the slow Qwen tokenizer is the
    # matching local fallback for both base and RL checkpoints.
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device)
    model.eval()
    return tokenizer, model


def generate_for_task(model, tokenizer, task: dict, n: int, max_new_tokens: int, temperature: float, top_p: float) -> list[dict]:
    prompt_ids = tokenizer.encode(prompt_for(task["question"]), add_special_tokens=False)
    prefix_ids = tokenizer.encode(task["prefix_text"], add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids + prefix_ids] * n, dtype=torch.long, device=model.device)
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, attention_mask=attention, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    start = input_ids.shape[1]
    outputs = []
    for index, row in enumerate(generated[:, start:].detach().cpu().tolist()):
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id)]
        continuation = tokenizer.decode(row, skip_special_tokens=True)
        full = task["prefix_text"] + continuation
        ev = evaluate_completion(full, task["ground_truth"])
        branch = extract_first_calc_branch(full)
        if branch == "<no_calc>":
            branch = NO_VALID_FIRST_CALC
        outputs.append({
            **{key: value for key, value in task.items() if key not in ("question",)},
            "stratum": task.get("stratum", "unknown"),
            "continuation_index": index,
            "continuation": continuation,
            "full_completion": full,
            "continuation_chars": len(continuation),
            "continuation_tokens": len(row),
            "official_correct": bool(ev.is_correct),
            "parsed": bool(ev.parsed),
            "format_valid": bool(ev.parsed and re.search(r"(?:\\boxed|####|final answer|answer)", full, re.IGNORECASE)),
            "observed_first_calc_branch": branch,
            "same_branch": branch == task["prefix_branch"] if task["prefix_branch"] not in ("NO_PREFIX", "NEUTRAL", NO_VALID_FIRST_CALC) else None,
            "branch_switched": branch != task["prefix_branch"] if task["prefix_branch"] not in ("NO_PREFIX", "NEUTRAL", NO_VALID_FIRST_CALC) else None,
        })
    return outputs


def aggregate(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("stratum", "unknown"), row["benchmark"], row["problem_id"], row["condition"], row["prefix_milestone"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        correct = [bool(row["official_correct"]) for row in values]
        item = {"stratum": key[0], "benchmark": key[1], "problem_id": key[2], "condition": key[3], "prefix_milestone": key[4], "n_continuations": len(values), "continuation_success_rate": float(np.mean(correct)), "any_success": float(any(correct)), "format_valid_rate": float(np.mean([row["format_valid"] for row in values])), "mean_continuation_tokens": float(np.mean([row["continuation_tokens"] for row in values]))}
        branch_values = [row["same_branch"] for row in values if row["same_branch"] is not None]
        switch_values = [row["branch_switched"] for row in values if row["branch_switched"] is not None]
        item["same_branch_rate"] = float(np.mean(branch_values)) if branch_values else None
        item["branch_switch_rate"] = float(np.mean(switch_values)) if switch_values else None
        output.append(item)
    return output


def main(argv=None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("prefix_handoff/semantic_boundary_analysis requires CUDA")
    torch.cuda.set_device(int(str(args.gpu).split(",")[0]))
    base, rl = PAIR_CONFIG[args.pair]
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    labels = load_labels(args.e1_root, base, rl, benchmarks)
    source = load_source_rows(labels)
    selected, posteriors = select_strata(labels, base, rl, args)
    tasks, prefixes = build_tasks(args, selected, labels, source, base, rl)
    # Attach questions after task construction so raw source remains the audit source.
    for task in tasks:
        raw = source[(task["source_file"], int(task["source_row"]))]
        task["question"] = str(raw.get("question", raw.get("problem", "")))
    confirmation_tasks = build_confirmation_tasks(selected, labels, source, base, rl, args.confirmation_free_samples)
    print(f"stratum_problems={len(selected)} tasks={len(tasks)} prefixes={len(prefixes)} confirmation_tasks={len(confirmation_tasks)}", flush=True)
    task_results = []
    confirmation_results = []
    targets = sorted({task["target_model"] for task in tasks})
    targets = sorted(set(targets) | {task["target_model"] for task in confirmation_tasks})
    device = torch.device("cuda")
    for target in targets:
        tokenizer, model = load_model(target, args.models_root, device)
        target_tasks = [task for task in tasks if task["target_model"] == target]
        for index, task in enumerate(target_tasks, start=1):
            task_results.extend(generate_for_task(model, tokenizer, task, args.n_continuations, args.max_new_tokens, args.temperature, args.top_p))
            if index % 10 == 0:
                print(f"[{target}] tasks {index}/{len(target_tasks)}", flush=True)
        for task in [task for task in confirmation_tasks if task["target_model"] == target]:
            confirmation_results.extend(generate_for_task(model, tokenizer, task, args.confirmation_free_samples, args.max_new_tokens, args.temperature, args.top_p))
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    output_root = args.output_root / args.experiment.lower() / args.pair
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "prefixes.jsonl").open("w", encoding="utf-8") as handle:
        for row in prefixes:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    with (output_root / "continuations.jsonl").open("w", encoding="utf-8") as handle:
        for row in task_results:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    with (output_root / "confirmation_free_sampling.jsonl").open("w", encoding="utf-8") as handle:
        for row in confirmation_results:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    pd.DataFrame(posteriors).to_csv(output_root / "selection.csv", index=False)
    pd.DataFrame(aggregate(task_results)).to_parquet(output_root / "per_problem.parquet", index=False)
    pd.DataFrame(aggregate(task_results)).to_csv(output_root / "summary.csv", index=False)
    pd.DataFrame(aggregate(confirmation_results)).to_csv(output_root / "confirmation_summary.csv", index=False)
    config = {"experiment_id": args.experiment, "pair": args.pair, "base_model": base, "rl_model": rl, "benchmarks": benchmarks, "delta": args.delta, "posterior_draws": args.posterior_draws, "n_continuations": args.n_continuations, "confirmation_free_samples": args.confirmation_free_samples, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p, "data_role": "discovery", "confirmation_required": True, "confirmation_output": "confirmation_free_sampling.jsonl", "selection_protocol": "Beta(1,1) posterior difference; lost/gained posterior probability > 0.95; shared P(both > .05) > .95", "prefix_milestones": ["P0", "P1", "P2"], "token_exact_prefix": True, "statistical_unit": "problem_prefix_continuation", "seed": args.seed}
    (output_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output_root), "rows": len(task_results), "confirmation_rows": len(confirmation_results), "selected": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
