
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""solution_coverage - Exact solution-space coverage on Countdown.
Hypothesis: later checkpoints concentrate probability on fewer valid solution classes even when pass@1 improves.
Inputs: Countdown raw JSONL files; dataset/test.parquet; exact expression enumerator.
Outputs: data/analysis_results/rlvr_passk/metrics/solution_coverage_solution_coverage_n320.csv; data/analysis_results/rlvr_passk/metrics/solution_coverage_solution_coverage_per_problem_n320.parquet
Status: paper-main
"""
"""
solution_coverage_solution_coverage_countdown.py

旗舰实验 C（解空间覆盖率）：
- 对每题精确枚举可行解集合 S(x)（按 canonical_expr 归并）
- 对 raw 采样统计：
  * Support mass: P(sample in S(x))
  * Coverage: |S_hit| / |S(x)|
  * top1 solution mass, entropy on-solution
  * off-support mass

输出：
- data/analysis_results/rlvr_passk/metrics/solution_coverage_solution_coverage_n{N}.csv
- 可选 per-problem parquet
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from early_branch_locking._repo import DATASET_DIR, METRICS_DIR as OUT_DIR, RAW_DIR, TEST_PARQUET  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)

from early_branch_locking.core.countdown_shared import (  # noqa: E402
    load_parquet_sorted,
    extract_ground_truth,
    entropy_from_counts,
    canonicalize_expression,
    enumerate_solution_set,
    load_jsonl,
    step_of,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, required=True)
    p.add_argument("--num_problems", type=int, default=100)
    p.add_argument("--glob", type=str, default="countdown_raw_*_n{N}.jsonl",
                   help="raw 文件匹配模式，{N} 会被替换为 n_samples")
    p.add_argument("--save_per_problem", action="store_true", default=False)
    return p.parse_args()


def build_solution_sets(records: List[dict]) -> Dict[int, Set[str]]:
    sol_sets: Dict[int, Set[str]] = {}
    for pid, rec in enumerate(records):
        numbers, target, _ = extract_ground_truth(rec)
        sol_sets[pid] = enumerate_solution_set(numbers, target)
    return sol_sets


def main():
    args = parse_args()
    pat = args.glob.replace("{N}", str(args.n_samples))
    files = sorted(RAW_DIR.glob(pat))
    if not files:
        raise FileNotFoundError(f"No raw files matched: {RAW_DIR}/{pat}")

    records = load_parquet_sorted(TEST_PARQUET, n=args.num_problems, sort_key="sample_id")
    sol_sets = build_solution_sets(records)

    all_rows = []
    per_problem_rows = []

    for fp in files:
        name = fp.name
        ckpt_tail = name[len("countdown_raw_") :]
        marker = f"_n{args.n_samples}"
        if marker not in ckpt_tail:
            raise ValueError(f"raw filename does not contain sample marker {marker}: {name}")
        ckpt = ckpt_tail.split(marker, 1)[0]

        by_prob = defaultdict(list)
        for rec in load_jsonl(fp):
            pid = int(rec["problem_index"])
            if pid >= args.num_problems:
                continue
            by_prob[pid].append(rec)

        support_mass_list = []
        off_support_mass_list = []
        coverage_list = []
        top1_mass_list = []
        top1_mass_cond_list = []
        entropy_list = []
        unique_solution_list = []
        solution_count_list = []
        correct_mass_list = []
        solver_feasible_list = []

        for pid, recs in by_prob.items():
            n = len(recs)
            sol_set = sol_sets.get(pid, set())
            solution_count = len(sol_set)
            solver_feasible = solution_count > 0

            sol_counts = Counter()
            correct = 0
            for r in recs:
                if r.get("overall_ok"):
                    correct += 1
                canon = r.get("canonical_expr")
                if canon and canon in sol_set:
                    sol_counts[canon] += 1

            support_hits = sum(sol_counts.values())
            support_mass = support_hits / n if n > 0 else 0.0
            off_support_mass = 1.0 - support_mass
            unique_solutions = len(sol_counts)
            coverage = (unique_solutions / solution_count) if solution_count > 0 else 0.0
            top1_mass = (max(sol_counts.values()) / n) if support_hits > 0 else 0.0
            top1_mass_cond = (max(sol_counts.values()) / support_hits) if support_hits > 0 else 0.0
            sol_entropy = entropy_from_counts(sol_counts) if support_hits > 0 else 0.0

            support_mass_list.append(support_mass)
            off_support_mass_list.append(off_support_mass)
            coverage_list.append(coverage)
            top1_mass_list.append(top1_mass)
            top1_mass_cond_list.append(top1_mass_cond)
            entropy_list.append(sol_entropy)
            unique_solution_list.append(unique_solutions)
            solution_count_list.append(solution_count)
            correct_mass_list.append(correct / n if n > 0 else 0.0)
            solver_feasible_list.append(1.0 if solver_feasible else 0.0)

            if args.save_per_problem:
                gt_numbers, gt_target, gt_feasible = extract_ground_truth(records[pid])
                per_problem_rows.append(dict(
                    checkpoint=ckpt,
                    problem_index=pid,
                    sample_id=records[pid].get("sample_id", -1),
                    numbers=gt_numbers,
                    target=gt_target,
                    feasible_label=gt_feasible,
                    solver_feasible=solver_feasible,
                    n=n,
                    correct_mass=(correct / n if n > 0 else 0.0),
                    solution_count=solution_count,
                    support_mass=support_mass,
                    off_support_mass=off_support_mass,
                    unique_solution=unique_solutions,
                    coverage=coverage,
                    top1_solution_mass=top1_mass,
                    top1_solution_mass_cond=top1_mass_cond,
                    solution_entropy=sol_entropy,
                ))

        nprobs = len(by_prob)
        row = dict(
            checkpoint=ckpt,
            n_samples=args.n_samples,
            num_problems=nprobs,
            solver_feasible_rate=float(np.mean(solver_feasible_list)) if solver_feasible_list else float("nan"),
            solution_count_mean=float(np.mean(solution_count_list)) if solution_count_list else float("nan"),
            support_mass_mean=float(np.mean(support_mass_list)) if support_mass_list else float("nan"),
            off_support_mass_mean=float(np.mean(off_support_mass_list)) if off_support_mass_list else float("nan"),
            coverage_mean=float(np.mean(coverage_list)) if coverage_list else float("nan"),
            unique_solution_mean=float(np.mean(unique_solution_list)) if unique_solution_list else float("nan"),
            top1_solution_mass_mean=float(np.mean(top1_mass_list)) if top1_mass_list else float("nan"),
            top1_solution_mass_cond_mean=float(np.mean(top1_mass_cond_list)) if top1_mass_cond_list else float("nan"),
            solution_entropy_mean=float(np.mean(entropy_list)) if entropy_list else float("nan"),
            correct_mass_mean=float(np.mean(correct_mass_list)) if correct_mass_list else float("nan"),
        )
        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    df["step"] = df["checkpoint"].apply(step_of)
    df = df.sort_values("step").reset_index(drop=True)

    out_csv = OUT_DIR / f"solution_coverage_solution_coverage_n{args.n_samples}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[solution_coverage] wrote: {out_csv}")
    print(df[
        [
            "checkpoint",
            "step",
            "solver_feasible_rate",
            "solution_count_mean",
            "support_mass_mean",
            "coverage_mean",
            "top1_solution_mass_mean",
            "off_support_mass_mean",
            "correct_mass_mean",
        ]
    ])

    if args.save_per_problem:
        out_parq = OUT_DIR / f"solution_coverage_solution_coverage_per_problem_n{args.n_samples}.parquet"
        pd.DataFrame(per_problem_rows).to_parquet(out_parq, index=False)
        print(f"[solution_coverage] wrote per-problem: {out_parq}")


if __name__ == "__main__":
    main()
