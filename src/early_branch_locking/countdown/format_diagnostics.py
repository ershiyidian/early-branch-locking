
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""format_diagnostics - Countdown format diagnostics.
Hypothesis: part of the RLVR trajectory change is a format transition that should be measured separately from expression correctness.
Inputs: Countdown raw JSONL files and parser-compatible completions.
Outputs: data/analysis_results/rlvr_passk/metrics/format_diagnostics_format_diagnostics_n320.csv
Status: paper-main
"""
"""
format_diagnostics_format_diagnostics_countdown.py

格式/长度诊断：统计 tag 出现率、顺序、parse_status 分布、输出长度。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import (  # noqa: E402
    METRICS_DIR,
    METRICS_DIR as OUT_DIR,
    RAW_DIR,
    TEST_PARQUET,
)

METRICS_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_shared import tolerant_parse_completion, load_jsonl, step_of  # noqa: E402

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, required=True)
    p.add_argument("--glob", type=str, default="countdown_raw_*_n{N}.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    pat = args.glob.replace("{N}", str(args.n_samples))
    files = sorted(RAW_DIR.glob(pat))
    if not files:
        raise FileNotFoundError(f"No raw files matched: {RAW_DIR}/{pat}")

    rows = []
    for fp in files:
        name = fp.name
        ckpt = name[len("countdown_raw_") :]
        ckpt = ckpt[: ckpt.rfind(f"_n{args.n_samples}.jsonl")]

        counts = Counter()
        parse_counts = Counter()
        lengths = []
        answer_lengths = []
        order_ok = 0
        order_total = 0

        for rec in load_jsonl(fp):
            counts["total"] += 1
            if rec.get("has_feasible_tag"):
                counts["has_feasible"] += 1
            if rec.get("has_answer_tag"):
                counts["has_answer"] += 1
            if rec.get("parse_status"):
                parse_counts[rec["parse_status"]] += 1
            if rec.get("tag_order_ok") is not None:
                order_total += 1
                if rec.get("tag_order_ok"):
                    order_ok += 1

            comp = rec.get("completion", "") or ""
            lengths.append(len(comp))
            parsed = tolerant_parse_completion(comp)
            ans = parsed.get("answer_block", "") or ""
            if ans:
                answer_lengths.append(len(ans))

        total = counts["total"] if counts["total"] > 0 else 1
        row = dict(
            checkpoint=ckpt,
            n_samples=args.n_samples,
            total=counts["total"],
            has_feasible_rate=counts["has_feasible"] / total,
            has_answer_rate=counts["has_answer"] / total,
            tag_order_ok_rate=(order_ok / order_total) if order_total > 0 else float("nan"),
            parse_ok_rate=(parse_counts.get("OK", 0) / total),
            completion_len_mean=float(np.mean(lengths)) if lengths else 0.0,
            completion_len_median=float(np.median(lengths)) if lengths else 0.0,
            answer_len_mean=float(np.mean(answer_lengths)) if answer_lengths else 0.0,
            answer_len_median=float(np.median(answer_lengths)) if answer_lengths else 0.0,
        )
        # parse_status rates
        for k, v in parse_counts.items():
            row[f"parse_{k}_rate"] = v / total

        rows.append(row)

    df = pd.DataFrame(rows)

    df["step"] = df["checkpoint"].apply(step_of)
    df = df.sort_values("step").reset_index(drop=True)

    out_csv = OUT_DIR / f"format_diagnostics_format_diagnostics_n{args.n_samples}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[format_diagnostics] wrote: {out_csv}")
    print(df[["checkpoint","step","has_feasible_rate","has_answer_rate","tag_order_ok_rate","parse_ok_rate","completion_len_mean","answer_len_mean"]])


# ---- merged error_breakdown mode ----
"""Format-conditioned coverage and error breakdown for Countdown rollouts."""

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from early_branch_locking.core.countdown_shared import enumerate_solution_set, extract_ground_truth, load_jsonl, load_parquet_sorted, step_of  # noqa: E402


def parse_args_error_breakdown() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExpE2: error breakdown and format-conditioned coverage.")
    parser.add_argument("--n_samples", type=int, default=320)
    parser.add_argument("--num_problems", type=int, default=150)
    parser.add_argument("--raw_paths", nargs="*", default=None)
    return parser.parse_args()


def default_raw_paths(n_samples: int) -> List[Path]:
    return sorted(RAW_DIR.glob(f"countdown_raw_global_step_*_n{n_samples}.jsonl"))


def solution_sets(num_problems: int) -> Dict[int, set]:
    records = load_parquet_sorted(TEST_PARQUET, n=num_problems, sort_key="sample_id")
    out = {}
    for pid, record in enumerate(records):
        numbers, target, feasible = extract_ground_truth(record)
        out[pid] = enumerate_solution_set(numbers, target) if feasible == "yes" else set()
    return out


def group_raw(path: Path, num_problems: int) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in load_jsonl(path):
        pid = int(row["problem_index"])
        if pid < num_problems:
            grouped[pid].append(row)
    return grouped


def checkpoint_name(path: Path, n_samples: int) -> str:
    name = path.name.removeprefix("countdown_raw_")
    return name.removesuffix(f"_n{n_samples}.jsonl")


def classify_counts(rows: Sequence[dict], sol_set: set) -> Counter:
    counts = Counter()
    for row in rows:
        if row.get("parse_status") != "OK":
            counts["format_failure"] += 1
            continue
        if row.get("overall_ok") and row.get("canonical_expr") in sol_set:
            counts["in_support_correct"] += 1
        elif row.get("overall_ok"):
            counts["valid_but_wrong_branch"] += 1
        else:
            counts["parseable_wrong"] += 1
    return counts


def per_problem_metrics(checkpoint: str, pid: int, rows: Sequence[dict], sol_set: set) -> dict:
    n = len(rows)
    counts = classify_counts(rows, sol_set)
    parse_ok = n - counts["format_failure"]
    correct_canons = {row.get("canonical_expr") for row in rows if row.get("overall_ok") and row.get("canonical_expr") in sol_set}
    parse_canons = {row.get("canonical_expr") for row in rows if row.get("parse_status") == "OK" and row.get("overall_ok") and row.get("canonical_expr") in sol_set}
    solution_count = len(sol_set)
    correct_hits = counts["in_support_correct"]
    return {
        "checkpoint": checkpoint,
        "problem_index": pid,
        "n": n,
        "solution_count": solution_count,
        "format_failure_rate": rate(counts["format_failure"], n),
        "parseable_wrong_rate": rate(counts["parseable_wrong"], n),
        "valid_but_wrong_branch_rate": rate(counts["valid_but_wrong_branch"], n),
        "in_support_correct_rate": rate(counts["in_support_correct"], n),
        "parse_ok_rate": rate(parse_ok, n),
        "support_mass_format_conditioned": rate(correct_hits, parse_ok),
        "coverage_full": rate(len(correct_canons), solution_count),
        "coverage_format_conditioned": rate(len(parse_canons), solution_count),
        "unique_correct": len(correct_canons),
        "unique_per_correct_sample": rate(len(correct_canons), correct_hits),
    }


def rate(num: float, denom: float) -> float:
    if denom <= 0:
        return math.nan
    return float(num / denom)


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in rows.columns if col not in {"checkpoint", "problem_index"}]
    out = rows.groupby("checkpoint", sort=False)[metric_cols].mean().reset_index()
    out["step"] = out["checkpoint"].apply(step_of)
    return out.sort_values("step")


def run_error_breakdown() -> None:
    args = parse_args_error_breakdown()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = [Path(path) for path in args.raw_paths] if args.raw_paths else default_raw_paths(args.n_samples)
    if not raw_paths:
        raise FileNotFoundError(f"No n{args.n_samples} raw rollout files found in {RAW_DIR}.")
    sol_sets = solution_sets(args.num_problems)
    per_rows = []
    for path in raw_paths:
        checkpoint = checkpoint_name(path, args.n_samples)
        grouped = group_raw(path, args.num_problems)
        for pid in range(args.num_problems):
            per_rows.append(per_problem_metrics(checkpoint, pid, grouped.get(pid, []), sol_sets[pid]))
    per_df = pd.DataFrame(per_rows)
    summary = aggregate(per_df)
    summary.to_csv(METRICS_DIR / f"format_diagnostics2_error_breakdown_n{args.n_samples}.csv", index=False)
    per_df.to_parquet(METRICS_DIR / f"format_diagnostics2_error_breakdown_per_problem_n{args.n_samples}.parquet", index=False)
    print(summary.to_string(index=False), flush=True)

def _run_selected():
    selector = None
    selector_index = None
    flag = "--mode"
    for index, argument in enumerate(sys.argv):
        if argument == flag:
            selector_index = index
            selector = sys.argv[index + 1] if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--") else "__flag__"
            break
        if argument.startswith(flag + "="):
            selector_index = index
            selector = argument.split("=", 1)[1]
            break
    if selector_index is not None:
        if selector == "__flag__":
            sys.argv.pop(selector_index)
        else:
            del sys.argv[selector_index:selector_index + 2]
        if selector == "error_breakdown":
            return run_error_breakdown()
        if selector not in {"format", "diagnostic", "__flag__"}:
            raise ValueError(f"Unknown --mode: {selector}")
    return main()

if __name__ == "__main__":
    _run_selected()
