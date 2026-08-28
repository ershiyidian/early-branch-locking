#!/usr/bin/env python3
"""branch_decomposition: open-math branch-access and observational-outcome decomposition.

Hypothesis: RL may concentrate early branch entry, including among correct
traces; observational within-branch outcome rates remain a separate measure.
Inputs: existing 64-sample merged math JSONL files for Qwen and OLMo pairs.
Procedure: label every trajectory with the frozen branch protocol, compute
problem-level all-sample/correct-only metrics, discovery curves, pairwise JSD,
and branch-definition agreement. No model is trained or sampled here.
Metrics: entry mass/entropy/effective count/top mass, Gini/Herfindahl,
format/no-calc diagnostics, branch@k, Good-Turing-ready frequencies, and
observational branch outcome rates.
Outputs: data/rlvr/outputs/e1/{labels.jsonl,per_problem.parquet,summary.csv,
discovery.csv,agreement.csv,observational_outcomes.csv,protocol.json}.
Statistical unit: problem, with samples nested below problem and 1000-draw
problem bootstrap confidence intervals for aggregate means.
Known limitations: source raw records use the historical evaluation prompt;
strategy labels are conservative frozen rules and cluster labels are a proxy.
Status: formal offline analysis; no training is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from early_branch_locking.core.math_trace_utils import extract_first_calc_branch
from early_branch_locking.core.branch_protocol import (
    NO_VALID_FIRST_CALC,
    assign_early_clusters,
    bootstrap_mean,
    completion_list,
    entropy,
    gini_counts,
    ground_truth,
    herfindahl,
    jensen_shannon,
    label_completion,
    normalized_mutual_information,
    adjusted_rand_index,
    pairwise_agreement,
    protocol_metadata,
    source_problem_id,
    variation_of_information,
)

OUTPUT_ROOT = ROOT / "data" / "rlvr" / "outputs" / "e1"
RAW_ROOT = ROOT / "data" / "rlvr" / "outputs" / "math"
DEFAULT_BENCHMARKS = ("gsm8k", "math500", "minerva_math", "olympiadbench", "amc23", "aime24")
MODEL_ALIASES = (
    "math_base_7b",
    "math_simple_rl_7b",
    "math_distill_qwen7b",
    "math_base_14b",
    "math_simple_rl_14b",
    "math_olmo3_sft_7b",
    "math_olmo3_dpo_7b",
    "math_olmo3_rlvr_7b",
)
PAIR_SPECS = (
    ("qwen_7b_base_rl", "math_base_7b", "math_simple_rl_7b"),
    ("qwen_14b_base_rl", "math_base_14b", "math_simple_rl_14b"),
    ("olmo_sft_rlvr", "math_olmo3_sft_7b", "math_olmo3_rlvr_7b"),
    ("olmo_dpo_rlvr", "math_olmo3_dpo_7b", "math_olmo3_rlvr_7b"),
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODEL_ALIASES))
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--input-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--min-samples", type=int, default=64)
    parser.add_argument("--discovery-repeats", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--allow-variable-samples", action="store_true")
    return parser.parse_args(argv)


def find_raw_path(root: Path, model: str, benchmark: str) -> Path | None:
    candidates = sorted((root / model / benchmark).glob("*_merged.jsonl"))
    candidates = [path for path in candidates if "_1_seed" not in path.name]
    return candidates[-1] if candidates else None


def load_labels(args: argparse.Namespace, models: Sequence[str], benchmarks: Sequence[str]) -> list[dict]:
    records: list[dict] = []
    missing = []
    for model in models:
        for benchmark in benchmarks:
            raw_path = find_raw_path(args.input_root, model, benchmark)
            if raw_path is None:
                missing.append(f"{model}/{benchmark}")
                continue
            source_path = raw_path.resolve()
            source_file = (
                str(source_path.relative_to(ROOT))
                if source_path.is_relative_to(ROOT)
                else str(source_path)
            )
            with source_path.open(encoding="utf-8") as handle:
                for row_index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    completions = completion_list(row)
                    if len(completions) < args.min_samples and not args.allow_variable_samples:
                        raise ValueError(f"{raw_path}:{row_index} has {len(completions)} samples, expected {args.min_samples}")
                    if len(completions) == 0:
                        continue
                    scores = row.get("score", row.get("scores"))
                    if not isinstance(scores, list) or len(scores) != len(completions):
                        scores = [None] * len(completions)
                    gt = ground_truth(row)
                    pid = source_problem_id(benchmark, row)
                    for sample_index, completion in enumerate(completions):
                        official = scores[sample_index]
                        if official is None:
                            from early_branch_locking.core.math_trace_utils import evaluate_completion

                            official = evaluate_completion(completion, gt).is_correct
                        records.append(
                            label_completion(
                                benchmark=benchmark,
                                model=model,
                                source_file=source_file,
                                source_row=row_index,
                                problem_id=pid,
                                sample_index=sample_index,
                                completion=completion,
                                gt=gt,
                                official_correct=bool(official),
                            )
                        )
    if missing:
        print("missing inputs:", ", ".join(missing), file=sys.stderr)
    if not records:
        raise RuntimeError("No branch_decomposition input records found")
    assign_early_clusters(records)
    return records


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _stable_problem_seed(problem_id: str, seed: int, offset: int = 0) -> int:
    digest = hashlib.sha1(problem_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") + seed + offset


def _discovery(labels: Sequence[str], repeats: int, seed: int) -> list[dict]:
    labels = list(labels)
    if not labels:
        return [{"k": k, "observed_richness_mean": 0.0, "p_at_least_1": 0.0, "p_at_least_2": 0.0, "p_at_least_3": 0.0, "p_at_least_5": 0.0} for k in range(1, 65)]
    rng = np.random.default_rng(seed)
    n = len(labels)
    result = []
    for k in range(1, min(64, n) + 1):
        richness = []
        probabilities = defaultdict(list)
        for _ in range(repeats):
            indices = rng.permutation(n)[:k]
            count = len(set(labels[index] for index in indices))
            richness.append(count)
            for threshold in (1, 2, 3, 5):
                probabilities[threshold].append(float(count >= threshold))
        result.append(
            {
                "k": k,
                "observed_richness_mean": float(np.mean(richness)),
                "observed_richness_sd": float(np.std(richness)),
                **{f"p_at_least_{threshold}": float(np.mean(values)) for threshold, values in probabilities.items()},
            }
        )
    return result


def problem_metrics(records: Sequence[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["model"], record["benchmark"], record["problem_id"])].append(record)
    per_problem = []
    discovery = []
    for (model, benchmark, pid), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda item: item["sample_index"])
        labels = [row["first_calc_branch"] for row in rows]
        correct_rows = [row for row in rows if row["official_correct"]]
        correct_labels = [row["first_calc_branch"] for row in correct_rows]
        n = len(rows)
        counts = Counter(labels)
        top = sorted(counts.values(), reverse=True)
        entry_entropy = entropy(labels)
        entry = {
            "model": model,
            "benchmark": benchmark,
            "problem_id": pid,
            "n_samples": n,
            "num_correct": len(correct_rows),
            "correct_rate": len(correct_rows) / n if n else 0.0,
            "format_failure_rate": _safe_mean(not row["format_valid"] for row in rows),
            "no_valid_first_calc_rate": _safe_mean(row["no_valid_first_calc"] for row in rows),
            "malformed_or_empty_rate": _safe_mean(row["malformed_or_empty"] for row in rows),
            "entry_entropy": entry_entropy,
            "entry_effective_branch_count": float(np.exp(entry_entropy)),
            "entry_top1_mass": top[0] / n if top else 0.0,
            "entry_top3_mass": sum(top[:3]) / n if top else 0.0,
            "entry_herfindahl": herfindahl(labels),
            "entry_gini": gini_counts(labels),
            "observed_branch_count": len(set(labels)),
            "correct_branch_entropy": entropy(correct_labels),
            "correct_effective_branch_count": float(np.exp(entropy(correct_labels))),
            "correct_observed_branch_count": len(set(correct_labels)),
            "correct_top1_mass": max(Counter(correct_labels).values()) / len(correct_labels) if correct_labels else 0.0,
            "correct_herfindahl": herfindahl(correct_labels),
            "correct_gini": gini_counts(correct_labels),
            "correct_trace_count": len({row["numeric_trace"] for row in correct_rows}),
            "strategy_observed_count": len({row["strategy_branch"] for row in rows}),
            "cluster_observed_count": len({row["early_cluster_id"] for row in rows}),
            "p0_available_rate": _safe_mean(row["p0_char_end"] is not None for row in rows),
            "p1_available_rate": _safe_mean(row["p1_char_end"] is not None for row in rows),
            "p2_available_rate": _safe_mean(row["p2_char_end"] is not None for row in rows),
            "p2_leakage_rate": _safe_mean(row["answer_leakage_before_p2"] for row in rows),
            "mean_completion_chars": _safe_mean(row["completion_chars"] for row in rows),
            "mean_token_count_proxy": _safe_mean(row["token_count_proxy"] for row in rows),
        }
        per_problem.append(entry)
        for item in _discovery(labels, args.discovery_repeats, _stable_problem_seed(pid, args.seed)):
            discovery.append({"model": model, "benchmark": benchmark, "problem_id": pid, "kind": "all_sample", **item})
        for item in _discovery(correct_labels, args.discovery_repeats, _stable_problem_seed(pid, args.seed, 1)):
            discovery.append({"model": model, "benchmark": benchmark, "problem_id": pid, "kind": "correct_only", **item})
    return per_problem, discovery


def write_labels(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {key: value for key, value in record.items() if key != "early_cluster_text"}
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def summarize(per_problem: Sequence[dict], args: argparse.Namespace) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_problem:
        groups[(row["model"], row["benchmark"])].append(row)
    metrics = [
        "correct_rate", "entry_entropy", "entry_effective_branch_count", "entry_top1_mass", "entry_top3_mass",
        "entry_herfindahl", "entry_gini", "observed_branch_count", "correct_branch_entropy",
        "correct_effective_branch_count", "correct_observed_branch_count", "correct_top1_mass",
        "correct_herfindahl", "correct_gini", "format_failure_rate", "no_valid_first_calc_rate",
        "p0_available_rate", "p1_available_rate", "p2_available_rate", "p2_leakage_rate",
    ]
    output = []
    for (model, benchmark), rows in sorted(groups.items()):
        base = {"model": model, "benchmark": benchmark, "num_problems": len(rows), "bootstrap_draws": args.bootstrap_draws}
        for metric_index, metric in enumerate(metrics):
            mean, lower, upper = bootstrap_mean([float(row[metric]) for row in rows], args.seed + metric_index, args.bootstrap_draws)
            base[f"{metric}_mean"] = mean
            base[f"{metric}_ci95_low"] = lower
            base[f"{metric}_ci95_high"] = upper
        output.append(base)
    return output


def pair_agreement(records: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["model"], row["benchmark"], row["problem_id"])].append(row)
    rows = []
    for (model, benchmark, pid), values in sorted(grouped.items()):
        values = sorted(values, key=lambda item: item["sample_index"])
        columns = {
            "first_calc_vs_strategy": ("first_calc_branch", "strategy_branch"),
            "first_calc_vs_cluster": ("first_calc_branch", "early_cluster_id"),
            "strategy_vs_cluster": ("strategy_branch", "early_cluster_id"),
        }
        for name, (left_name, right_name) in columns.items():
            left = [item[left_name] for item in values]
            right = [item[right_name] for item in values]
            rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "problem_id": pid,
                    "comparison": name,
                    "pairwise_co_membership": pairwise_agreement(left, right),
                    "ari": adjusted_rand_index(left, right),
                    "nmi": normalized_mutual_information(left, right),
                    "vi": variation_of_information(left, right),
                    "n_samples": len(values),
                }
            )
    return rows


def pair_outcomes(per_problem: Sequence[dict], records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    by_model_problem: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        by_model_problem[(row["model"], row["benchmark"], row["problem_id"])].append(row)
    jsd_rows, outcome_rows = [], []
    for pair_name, base, rl in PAIR_SPECS:
        benchmarks = sorted({key[1] for key in by_model_problem if key[0] in (base, rl)})
        for benchmark in benchmarks:
            pids = sorted({key[2] for key in by_model_problem if key[:2] in ((base, benchmark), (rl, benchmark))})
            for pid in pids:
                left_rows = sorted(by_model_problem.get((base, benchmark, pid), []), key=lambda item: item["sample_index"])
                right_rows = sorted(by_model_problem.get((rl, benchmark, pid), []), key=lambda item: item["sample_index"])
                if not left_rows or not right_rows:
                    continue
                for kind, predicate in (("all_sample", lambda row: True), ("correct_only", lambda row: row["official_correct"])):
                    left = [row["first_calc_branch"] for row in left_rows if predicate(row)]
                    right = [row["first_calc_branch"] for row in right_rows if predicate(row)]
                    jsd_rows.append({"pair": pair_name, "base_model": base, "rl_model": rl, "benchmark": benchmark, "problem_id": pid, "kind": kind, "jsd": jensen_shannon(left, right), "base_n": len(left), "rl_n": len(right)})
                    if kind != "all_sample":
                        continue
                    left_counts, right_counts = Counter(left), Counter(right)
                    for branch in sorted(set(left_counts) | set(right_counts)):
                        base_n, rl_n = left_counts[branch], right_counts[branch]
                        outcome_rows.append({
                            "pair": pair_name, "base_model": base, "rl_model": rl, "benchmark": benchmark, "problem_id": pid,
                            "branch": branch, "base_entry_count": base_n, "rl_entry_count": rl_n,
                            "base_entry_rate": base_n / len(left) if left else 0.0, "rl_entry_rate": rl_n / len(right) if right else 0.0,
                            "base_correct_count": sum(row["official_correct"] for row in left_rows if row["first_calc_branch"] == branch),
                            "rl_correct_count": sum(row["official_correct"] for row in right_rows if row["first_calc_branch"] == branch),
                            "base_outcome_rate": sum(row["official_correct"] for row in left_rows if row["first_calc_branch"] == branch) / base_n if base_n else None,
                            "rl_outcome_rate": sum(row["official_correct"] for row in right_rows if row["first_calc_branch"] == branch) / rl_n if rl_n else None,
                        })
    return jsd_rows, outcome_rows


def main(argv=None) -> None:
    args = parse_args(argv)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    records = load_labels(args, models, benchmarks)
    per_problem, discovery = problem_metrics(records, args)
    summaries = summarize(per_problem, args)
    agreement = pair_agreement(records)
    jsd_rows, outcome_rows = pair_outcomes(per_problem, records)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    write_labels(output / "labels.jsonl", records)
    pd.DataFrame(per_problem).to_parquet(output / "per_problem.parquet", index=False)
    pd.DataFrame(discovery).to_csv(output / "discovery.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(agreement).to_csv(output / "agreement.csv", index=False)
    pd.DataFrame(jsd_rows).to_csv(output / "branch_jsd.csv", index=False)
    pd.DataFrame(outcome_rows).to_csv(output / "observational_outcomes.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(protocol_metadata(), indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "experiment_id": "branch_decomposition",
        "protocol": protocol_metadata(),
        "models": models,
        "benchmarks": benchmarks,
        "num_labeled_samples": len(records),
        "num_problem_rows": len(per_problem),
        "source_root": str(args.input_root.relative_to(ROOT) if args.input_root.is_relative_to(ROOT) else args.input_root),
        "output_root": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
        "seed": args.seed,
        "discovery_repeats": args.discovery_repeats,
        "bootstrap_draws": args.bootstrap_draws,
        "source_sha256": hashlib.sha256("\n".join(sorted({row["source_file"] for row in records})).encode()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(records), "problem_rows": len(per_problem), "summaries": len(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()
